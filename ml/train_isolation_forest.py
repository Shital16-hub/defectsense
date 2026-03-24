"""
Train Isolation Forest as ensemble backup anomaly detector on AI4I 2020.

Approach:
    - Train on NORMAL samples only
    - contamination=0.05 (expect ~5% anomalies in real operation)
    - Output score: decision_function value  (higher = more normal)
    - Anomaly: predict() == -1

Run:
    python ml/train_isolation_forest.py

Outputs:
    ml/models/isolation_forest.pkl
"""
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "ai4i_2020.csv"
MODELS_DIR = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = MODELS_DIR / "sensor_scaler.pkl"

FEATURES = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]

# ── Hyperparameters ───────────────────────────────────────────────────────────
CONTAMINATION = 0.05
N_ESTIMATORS = 200
MAX_SAMPLES = "auto"
RANDOM_STATE = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        print(f"ERROR: Dataset not found at {path}")
        print("Run:  python data/download_data.py  first.")
        sys.exit(1)

    df = pd.read_csv(path)

    col_map = {
        "Air temperature [K]": "air_temperature",
        "Process temperature [K]": "process_temperature",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "machine_failure",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    normal = df[df["machine_failure"] == 0][FEATURES + ["machine_failure"]].dropna()
    anomaly = df[df["machine_failure"] == 1][FEATURES + ["machine_failure"]].dropna()

    print(f"Normal samples  : {len(normal):,}")
    print(f"Anomaly samples : {len(anomaly):,}")
    return normal, anomaly


def get_or_create_scaler(normal_df: pd.DataFrame) -> MinMaxScaler:
    """Reuse LSTM scaler if available (ensures consistent feature scaling)."""
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded existing scaler from {SCALER_PATH}")
    else:
        scaler = MinMaxScaler()
        scaler.fit(normal_df[FEATURES])
        print("Created new scaler (LSTM autoencoder not trained yet — run train_autoencoder.py first)")
    return scaler


def evaluate(model: IsolationForest, scaler: MinMaxScaler, normal_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> dict:
    """Evaluate on full dataset (normal + anomaly) — for reporting only."""
    X_normal = scaler.transform(normal_df[FEATURES])
    X_anomaly = scaler.transform(anomaly_df[FEATURES])

    X_all = np.vstack([X_normal, X_anomaly])
    y_true = np.array([1] * len(X_normal) + [-1] * len(X_anomaly))

    y_pred = model.predict(X_all)
    scores = model.decision_function(X_all)

    # Convert sklearn convention: 1=normal, -1=anomaly
    # to our convention: 0=normal, 1=anomaly for confusion matrix
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "score_mean_normal": float(np.mean(scores[:len(X_normal)])),
        "score_mean_anomaly": float(np.mean(scores[len(X_normal):])),
    }


def main() -> None:
    print("=" * 60)
    print("  DefectSense — Isolation Forest Training")
    print("=" * 60)
    print()

    normal_df, anomaly_df = load_data(DATA_PATH)
    scaler = get_or_create_scaler(normal_df)

    # Train on normal samples only
    X_train = scaler.transform(normal_df[FEATURES])

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(f"\nTraining IsolationForest (n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION})...")
    model.fit(X_train)
    print("Training complete.")

    # Evaluate
    metrics = evaluate(model, scaler, normal_df, anomaly_df)

    # ── MLflow Tracking ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("defectsense_anomaly_detection")

    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_params({
            "model_type": "IsolationForest",
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "max_samples": MAX_SAMPLES,
            "features": str(FEATURES),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "isolation_forest")

    # Save model
    model_path = MODELS_DIR / "isolation_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
            "contamination": CONTAMINATION,
        }, f)

    print()
    print("=" * 60)
    print("  Evaluation Results (on full labeled dataset)")
    print("=" * 60)
    print(f"  Precision           : {metrics['precision']:.4f}")
    print(f"  Recall              : {metrics['recall']:.4f}")
    print(f"  F1 Score            : {metrics['f1_score']:.4f}")
    print(f"  False Positive Rate : {metrics['false_positive_rate']:.4f}")
    print(f"  True Positives      : {metrics['true_positives']}")
    print(f"  False Positives     : {metrics['false_positives']}")
    print(f"  True Negatives      : {metrics['true_negatives']}")
    print(f"  False Negatives     : {metrics['false_negatives']}")
    print(f"  Score (normal mean) : {metrics['score_mean_normal']:.4f}")
    print(f"  Score (anomaly mean): {metrics['score_mean_anomaly']:.4f}")
    print()
    print(f"  Model saved → {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
