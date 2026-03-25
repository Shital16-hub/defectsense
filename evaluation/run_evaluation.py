"""
DefectSense ML Model Evaluation — Session 7.

Evaluates LSTM Autoencoder, Isolation Forest, and Ensemble against the
full AI4I 2020 dataset (held-out test split: last 20%).

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
DATA_PATH  = ROOT / "data" / "ai4i_2020.csv"
OUT_PATH   = Path(__file__).parent / "ml_benchmark.json"

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
TARGET   = "machine_failure"
SEQ_LEN  = 30


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Use last 20% as test set (time-ordered)
    split = int(len(df) * 0.8)
    test  = df.iloc[split:].reset_index(drop=True)
    print(f"Test set: {len(test)} rows  |  failures: {test[TARGET].sum()} ({test[TARGET].mean():.1%})")
    return test


def load_models():
    with open(MODELS_DIR / "sensor_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "anomaly_threshold.pkl", "rb") as f:
        thresh_data = pickle.load(f)
    threshold = thresh_data.get("threshold", 0.0)

    with open(MODELS_DIR / "isolation_forest.pkl", "rb") as f:
        iforest_data = pickle.load(f)
    iforest = iforest_data["model"] if isinstance(iforest_data, dict) else iforest_data

    import tensorflow as tf
    autoencoder = tf.keras.models.load_model(str(MODELS_DIR / "lstm_autoencoder.keras"))

    return scaler, autoencoder, threshold, iforest


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.0
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "auc":       round(auc,  4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_iforest(df: pd.DataFrame, scaler, iforest) -> dict:
    print("\nEvaluating Isolation Forest...")
    X      = scaler.transform(df[FEATURES].values)
    y_true = df[TARGET].values.astype(int)
    preds  = iforest.predict(X)          # 1=normal, -1=anomaly
    y_pred = (preds == -1).astype(int)
    scores = -iforest.decision_function(X)  # higher = more anomalous
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    m = metrics(y_true, y_pred, scores)
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    return m


def evaluate_lstm(df: pd.DataFrame, scaler, autoencoder, threshold: float) -> dict:
    print("\nEvaluating LSTM Autoencoder (sliding windows)...")
    X_all  = scaler.transform(df[FEATURES].values).astype(np.float32)
    y_true_all = df[TARGET].values.astype(int)

    recon_errors = []
    labels       = []

    for i in range(SEQ_LEN, len(X_all)):
        seq    = X_all[i - SEQ_LEN:i][np.newaxis, ...]   # (1, 30, 5)
        recon  = autoencoder.predict(seq, verbose=0)
        mse    = float(np.mean(np.power(seq - recon, 2)))
        recon_errors.append(mse)
        labels.append(y_true_all[i])

    recon_errors = np.array(recon_errors)
    labels       = np.array(labels)
    y_pred       = (recon_errors > threshold).astype(int)
    # Normalise scores to [0,1] for AUC
    scores = np.clip(recon_errors / (threshold * 3.0), 0.0, 1.0)

    m = metrics(labels, y_pred, scores)
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    return m


def evaluate_ensemble(df: pd.DataFrame, scaler, autoencoder, threshold: float, iforest) -> dict:
    print("\nEvaluating Ensemble (LSTM + IForest)...")
    X_all      = scaler.transform(df[FEATURES].values).astype(np.float32)
    y_true_all = df[TARGET].values.astype(int)

    recon_errors = []
    iforest_preds = iforest.predict(X_all)   # 1=normal, -1=anomaly

    for i in range(SEQ_LEN, len(X_all)):
        seq   = X_all[i - SEQ_LEN:i][np.newaxis, ...]
        recon = autoencoder.predict(seq, verbose=0)
        recon_errors.append(float(np.mean(np.power(seq - recon, 2))))

    recon_errors  = np.array(recon_errors)
    iforest_slice = (iforest_preds[SEQ_LEN:] == -1).astype(int)
    labels        = y_true_all[SEQ_LEN:]

    lstm_flag = (recon_errors > threshold).astype(int)

    # Ensemble: high-conf = both flag; medium = either
    y_pred = np.clip(lstm_flag + iforest_slice, 0, 1)

    # Combined score
    lstm_norm    = np.clip(recon_errors / (threshold * 3.0), 0.0, 1.0)
    iforest_raw  = -iforest.decision_function(X_all[SEQ_LEN:])
    iforest_norm = (iforest_raw - iforest_raw.min()) / (iforest_raw.max() - iforest_raw.min() + 1e-8)
    scores       = 0.6 * lstm_norm + 0.4 * iforest_norm

    m = metrics(labels, y_pred, scores)
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    return m


def main():
    print("=" * 60)
    print("  DefectSense — ML Model Evaluation")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df             = load_data()
    scaler, autoencoder, threshold, iforest = load_models()
    print(f"Threshold: {threshold:.6f}")

    iforest_metrics  = evaluate_iforest(df, scaler, iforest)
    lstm_metrics     = evaluate_lstm(df, scaler, autoencoder, threshold)
    ensemble_metrics = evaluate_ensemble(df, scaler, autoencoder, threshold, iforest)

    results = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "test_split": "last 20% (2000 rows)",
        "test_failures": int(df[TARGET].sum()),
        "test_failure_rate": round(float(df[TARGET].mean()), 4),
        "threshold": round(threshold, 6),
        "models": {
            "isolation_forest": iforest_metrics,
            "lstm_autoencoder": lstm_metrics,
            "ensemble":         ensemble_metrics,
        },
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUT_PATH}")
    print("\n=== Summary ===")
    print(f"{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 60)
    for name, m in results["models"].items():
        print(f"{name:<22} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    return results


if __name__ == "__main__":
    main()
