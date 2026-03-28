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
import os
import pickle
import sys
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

load_dotenv(Path(__file__).parent.parent / ".env")
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
    # ── Try PostgreSQL first ───────────────────────────────────────────────────
    try:
        sys.path.insert(0, str(ROOT))
        from app.services.postgres_service import PostgresService
        pg = PostgresService(os.getenv("POSTGRES_URL"))
        pg.init()
        if pg.is_connected:
            pg_normal  = pg.get_normal_samples()
            pg_anomaly = pg.get_failure_samples()
            needed = FEATURES + ["machine_failure"]
            if (
                len(pg_normal) > 100
                and all(f in pg_normal.columns for f in needed)
                and all(f in pg_anomaly.columns for f in needed)
            ):
                normal  = pg_normal[needed].dropna()
                anomaly = pg_anomaly[needed].dropna()
                print(f"Training data source: PostgreSQL ({len(normal):,} normal samples)")
                print(f"Normal samples  : {len(normal):,}")
                print(f"Anomaly samples : {len(anomaly):,}")
                pg.close()
                return normal, anomaly
            pg.close()
    except Exception:
        pass  # fall through to CSV

    # ── Fall back to CSV ───────────────────────────────────────────────────────
    print("Training data source: CSV fallback")
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

    normal  = df[df["machine_failure"] == 0][FEATURES + ["machine_failure"]].dropna()
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

    # AUC: negate decision_function so higher value = more anomalous
    try:
        auc = float(roc_auc_score(y_true_bin, -scores))
    except Exception:
        auc = 0.0

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
        "auc": round(auc, 4),
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
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    Path(tracking_uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("defectsense_anomaly_detection")

    registered_version = None
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

        # ── MLflow Model Registry ──────────────────────────────────────────────
        try:
            sys.path.insert(0, str(ROOT))
            from ml.model_registry_service import ModelRegistryService
            registry = ModelRegistryService()
            registry.init()

            model_uri = (
                f"runs:/{mlflow.active_run().info.run_id}"
                f"/isolation_forest"
            )
            result = mlflow.register_model(
                model_uri=model_uri,
                name="defectsense_isolation_forest",
            )

            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name="defectsense_isolation_forest",
                alias="challenger",
                version=result.version,
            )
            registered_version = result.version
        except Exception as exc:
            print(f"Registry warning (non-fatal): {exc}")

    # Save model
    model_path = MODELS_DIR / "isolation_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
            "contamination": CONTAMINATION,
        }, f)

    # ── Azure Blob Upload ──────────────────────────────────────────────────────
    today = date.today().strftime("%Y%m%d")
    print("\nUploading artefacts to Azure Blob Storage...")
    azure_ok = _upload_to_azure([
        (model_path, "isolation_forest_latest.pkl"),
        (model_path, f"isolation_forest_{today}.pkl"),
    ])

    # ── Final Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Isolation Forest Training Complete")
    print("=" * 60)
    reg_str   = f"defectsense_isolation_forest v{registered_version}" if registered_version else "FAILED (see above)"
    azure_str = "isolation_forest_latest.pkl [OK]" if azure_ok else "skipped / failed (see above)"
    print(f"  Local:     ml/models/isolation_forest.pkl")
    print(f"  Azure:     {azure_str}")
    print(f"  Registry:  {reg_str}")
    print(f"  Alias:     challenger")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1_score']:.4f}")
    print("=" * 60)


# ── Azure Blob Upload helper ───────────────────────────────────────────────────

def _upload_to_azure(uploads: list[tuple[Path, str]]) -> bool:
    """Upload model files to Azure Blob. Returns True if all uploads succeeded."""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "defectsense-models")

    if not connection_string:
        print("Azure: AZURE_STORAGE_CONNECTION_STRING not set — skipping upload.")
        return False

    try:
        sys.path.insert(0, str(ROOT))
        from app.services.blob_storage_service import BlobStorageService  # noqa: PLC0415
        blob_service = BlobStorageService(connection_string, container_name)
        if not blob_service.is_available:
            print("Azure: blob storage unavailable — skipping upload.")
            return False
        all_ok = True
        for local_path, blob_name in uploads:
            ok = blob_service.upload_model(local_path, blob_name)
            status = "OK" if ok else "FAILED"
            all_ok = all_ok and ok
            print(f"  Azure upload [{status:6s}]: {blob_name}")
        return all_ok
    except Exception as exc:
        print(f"  Azure upload error (non-fatal): {exc}")
        return False


if __name__ == "__main__":
    main()
