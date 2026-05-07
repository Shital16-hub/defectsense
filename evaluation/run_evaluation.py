"""
DefectSense ML Model Evaluation — Azure PdM LSTM Autoencoder.

Evaluates the LSTM Autoencoder against the Azure PdM dataset.
Saves results to evaluation/ml_benchmark.json.

Usage:
    python evaluation/run_evaluation.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "ml" / "models"
DATA_DIR   = ROOT / "data" / "azure_pdm"
OUT_PATH   = Path(__file__).parent / "ml_benchmark.json"

FEATURES = [
    "volt",
    "rotate",
    "pressure",
    "vibration",
]
SEQ_LEN  = 198  # loaded from config if available


def _load_seq_len() -> int:
    config_path = ROOT / "data" / "azure_lstm_config.json"
    try:
        with open(config_path) as f:
            return int(json.load(f)["sequence_length"])
    except Exception:
        return SEQ_LEN


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Azure PdM telemetry and failure tables."""
    telemetry_path = DATA_DIR / "PdM_telemetry.csv"
    failures_path  = DATA_DIR / "PdM_failures.csv"

    if not telemetry_path.exists():
        print(f"ERROR: telemetry not found at {telemetry_path}")
        sys.exit(1)
    if not failures_path.exists():
        print(f"ERROR: failures not found at {failures_path}")
        sys.exit(1)

    telem = pd.read_csv(telemetry_path, parse_dates=["datetime"])
    fails = pd.read_csv(failures_path,  parse_dates=["datetime"])
    print(f"Telemetry : {len(telem):,} rows  |  {telem['machineID'].nunique()} machines")
    print(f"Failures  : {len(fails):,} events")
    return telem, fails


def load_models():
    scaler_path    = MODELS_DIR / "azure_sensor_scaler.pkl"
    threshold_path = MODELS_DIR / "azure_anomaly_threshold.pkl"
    model_path     = MODELS_DIR / "azure_lstm_autoencoder.keras"

    if not scaler_path.exists():
        print(f"ERROR: scaler not found at {scaler_path}")
        sys.exit(1)
    if not threshold_path.exists():
        print(f"ERROR: threshold not found at {threshold_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(threshold_path, "rb") as f:
        thresh_data = pickle.load(f)
    threshold = thresh_data.get("threshold", 0.0)

    import tensorflow as tf
    autoencoder = tf.keras.models.load_model(str(model_path))
    return scaler, autoencoder, threshold


def evaluate_lstm(
    telem: pd.DataFrame,
    fails: pd.DataFrame,
    scaler,
    autoencoder,
    threshold: float,
    seq_len: int,
) -> dict:
    """
    Build pre-failure sequences for machines with failures and clean sequences
    for machines without. Compute reconstruction error and AUC.
    """
    from sklearn.metrics import roc_auc_score

    print("\nEvaluating LSTM Autoencoder on Azure PdM...")

    failure_errors: list[float] = []
    clean_errors:   list[float] = []

    machines_with_failures = set(fails["machineID"].unique())

    for mid in sorted(telem["machineID"].unique()):
        mdf    = telem[telem["machineID"] == mid].sort_values("datetime").reset_index(drop=True)
        scaled = scaler.transform(mdf[FEATURES].values).astype(np.float32)

        if mid in machines_with_failures:
            # Build one pre-failure sequence per failure event
            for _, frow in fails[fails["machineID"] == mid].iterrows():
                idx = mdf[mdf["datetime"] < frow["datetime"]].index
                if len(idx) < seq_len:
                    continue
                seq   = scaled[idx[-seq_len:]][np.newaxis, ...]
                recon = autoencoder.predict(seq, verbose=0)
                failure_errors.append(float(np.mean(np.power(seq - recon, 2))))
        else:
            # Sample up to 5 clean windows per machine
            n_windows = len(scaled) - seq_len
            if n_windows <= 0:
                continue
            step = max(1, n_windows // 5)
            for start in range(0, n_windows, step):
                seq   = scaled[start:start + seq_len][np.newaxis, ...]
                recon = autoencoder.predict(seq, verbose=0)
                clean_errors.append(float(np.mean(np.power(seq - recon, 2))))
                if len(clean_errors) >= 500:
                    break

    if not failure_errors or not clean_errors:
        print("  WARNING: insufficient data for AUC computation")
        return {"auc": 0.0, "detection_rate": 0.0,
                "failure_sequences": 0, "clean_sequences": 0}

    all_errors = np.array(clean_errors + failure_errors)
    labels     = np.array([0] * len(clean_errors) + [1] * len(failure_errors))
    auc = float(roc_auc_score(labels, all_errors))
    detection_rate = float(np.mean(np.array(failure_errors) > threshold))

    print(f"  Failure sequences : {len(failure_errors)}")
    print(f"  Clean sequences   : {len(clean_errors)}")
    print(f"  Detection rate    : {detection_rate:.1%}")
    print(f"  AUC               : {auc:.4f}")
    return {
        "auc":                auc,
        "detection_rate":     detection_rate,
        "failure_sequences":  len(failure_errors),
        "clean_sequences":    len(clean_errors),
    }


def main():
    print("=" * 60)
    print("  DefectSense — ML Model Evaluation (Azure PdM)")
    print("=" * 60)

    seq_len = _load_seq_len()
    print(f"Sequence length: {seq_len}")

    telem, fails     = load_data()
    scaler, autoencoder, threshold = load_models()
    print(f"Threshold: {threshold:.6f}")

    lstm_metrics = evaluate_lstm(telem, fails, scaler, autoencoder, threshold, seq_len)

    results = {
        "dataset":       "Azure PdM Predictive Maintenance",
        "n_machines":    int(telem["machineID"].nunique()),
        "n_telemetry":   len(telem),
        "n_failures":    len(fails),
        "threshold":     round(threshold, 6),
        "sequence_length": seq_len,
        "models": {
            "lstm_autoencoder": lstm_metrics,
        },
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUT_PATH}")
    print("\n=== Summary ===")
    m = lstm_metrics
    print(f"  AUC            : {m['auc']:.4f}")
    print(f"  Detection rate : {m['detection_rate']:.1%}")

    # ── Compare against registered champion ───────────────────────────────────
    try:
        sys.path.insert(0, str(ROOT))
        from ml.model_registry_service import ModelRegistryService
        registry = ModelRegistryService()
        registry.init()

        prod = registry.get_latest_version(
            "defectsense_lstm_autoencoder",
            stage="Production",
        )

        if prod:
            prod_auc    = prod["metrics"].get("auc", 0.0)
            current_auc = lstm_metrics["auc"]
            print("\n=== Champion Comparison ===")
            print(f"  Current AUC  : {current_auc:.4f}")
            print(f"  Champion AUC : {prod_auc:.4f}")
            if current_auc > prod_auc:
                print("RECOMMENDATION: Promote to champion")
                print("Run: python ml/promote_to_production.py --model lstm --version X")
        else:
            print("\nNo champion model registered.")
            print("Run: python ml/promote_to_production.py --model lstm --version 1")
    except Exception as exc:
        print(f"Registry check skipped: {exc}")

    return results


if __name__ == "__main__":
    main()
