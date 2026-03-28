"""
Train LSTM Autoencoder for anomaly detection on AI4I 2020 sensor data.

Architecture:
    Input (30, 5) → Encoder [LSTM-64 → LSTM-32 → Dense-16]
                  → Decoder [RepeatVector(30) → LSTM-32 → LSTM-64 → TimeDistributed Dense-5]

Training:
    - Trained on NORMAL samples only (no failures)
    - Anomaly threshold = mean + MULTIPLIER * std of validation reconstruction errors

Run:
    python ml/train_autoencoder.py

Outputs:
    ml/models/lstm_autoencoder.keras
    ml/models/sensor_scaler.pkl
    ml/models/anomaly_threshold.pkl
"""
import os
import pickle
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "ai4i_2020.csv"
MODELS_DIR = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
FEATURES = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
N_FEATURES = len(FEATURES)
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
THRESHOLD_MULTIPLIER = 3.0  # mean + N * std


# ── Data Preparation ──────────────────────────────────────────────────────────

def load_and_prepare(path: Path) -> tuple[np.ndarray, np.ndarray, object]:
    """Load CSV, filter normal samples, scale, create sequences."""
    from sklearn.preprocessing import MinMaxScaler

    # ── Try PostgreSQL first ───────────────────────────────────────────────────
    normal = pd.DataFrame()
    try:
        from app.services.postgres_service import PostgresService
        pg = PostgresService(os.getenv("POSTGRES_URL"))
        pg.init()
        if pg.is_connected:
            pg_normal = pg.get_normal_samples()
            if len(pg_normal) > 100 and all(f in pg_normal.columns for f in FEATURES):
                normal = pg_normal[FEATURES].dropna()
                print(f"Training data source: PostgreSQL ({len(normal):,} normal samples)")
            pg.close()
    except Exception as _exc:
        pass  # fall through to CSV

    # ── Fall back to CSV ───────────────────────────────────────────────────────
    if normal.empty:
        print("Training data source: CSV fallback")
        if not path.exists():
            print(f"ERROR: Dataset not found at {path}")
            print("Run:  python data/download_data.py  first.")
            sys.exit(1)

        df = pd.read_csv(path)

        # Handle both original column names and renamed ones
        col_map = {
            "Air temperature [K]": "air_temperature",
            "Process temperature [K]": "process_temperature",
            "Rotational speed [rpm]": "rotational_speed",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
            "Machine failure": "machine_failure",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        normal = df[df["machine_failure"] == 0][FEATURES].dropna()
        print(f"Normal samples : {len(normal):,} / {len(df):,} total")

    if normal.empty:
        print("ERROR: No normal samples available from PostgreSQL or CSV.")
        sys.exit(1)

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    normal_scaled = scaler.fit_transform(normal)

    # Create sliding window sequences
    sequences = []
    for i in range(len(normal_scaled) - SEQUENCE_LENGTH):
        sequences.append(normal_scaled[i : i + SEQUENCE_LENGTH])
    sequences = np.array(sequences, dtype=np.float32)
    print(f"Sequences      : {sequences.shape}  (samples × timesteps × features)")

    return sequences, normal_scaled, scaler


def build_model() -> "tf.keras.Model":
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, N_FEATURES))

    # Encoder
    x = layers.LSTM(LSTM_UNITS_1, return_sequences=True)(inputs)
    x = layers.LSTM(LSTM_UNITS_2, return_sequences=False)(x)
    encoded = layers.Dense(DENSE_UNITS, activation="relu")(x)

    # Decoder
    x = layers.RepeatVector(SEQUENCE_LENGTH)(encoded)
    x = layers.LSTM(LSTM_UNITS_2, return_sequences=True)(x)
    x = layers.LSTM(LSTM_UNITS_1, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(N_FEATURES))(x)

    model = Model(inputs, decoded, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def compute_threshold(model, val_sequences: np.ndarray) -> tuple[float, float, float]:
    """Compute reconstruction errors on validation set, return threshold."""
    reconstructed = model.predict(val_sequences, verbose=0)
    mse_per_sample = np.mean(np.power(val_sequences - reconstructed, 2), axis=(1, 2))
    mean_err = float(np.mean(mse_per_sample))
    std_err = float(np.std(mse_per_sample))
    threshold = mean_err + THRESHOLD_MULTIPLIER * std_err
    return threshold, mean_err, std_err


def compute_post_training_auc(model, scaler, data_path: Path) -> float:
    """Compute ROC-AUC on the held-out test split (last 20% of CSV).

    Uses the same sliding-window approach as evaluation/run_evaluation.py.
    Falls back to 0.0 if the CSV is unavailable or AUC cannot be computed.
    """
    try:
        from sklearn.metrics import roc_auc_score

        if not data_path.exists():
            return 0.0

        df = pd.read_csv(data_path)
        col_map = {
            "Air temperature [K]": "air_temperature",
            "Process temperature [K]": "process_temperature",
            "Rotational speed [rpm]": "rotational_speed",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
            "Machine failure": "machine_failure",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        split = int(len(df) * 0.8)
        test = df.iloc[split:].reset_index(drop=True)

        X_all = scaler.transform(test[FEATURES].values).astype(np.float32)
        y_true = test["machine_failure"].values.astype(int)

        # Build all sequences as a single batch for efficient prediction
        seqs = np.array(
            [X_all[i - SEQUENCE_LENGTH : i] for i in range(SEQUENCE_LENGTH, len(X_all))],
            dtype=np.float32,
        )
        labels = y_true[SEQUENCE_LENGTH:]

        recon  = model.predict(seqs, verbose=0)
        errors = np.mean(np.power(seqs - recon, 2), axis=(1, 2))

        return float(roc_auc_score(labels, errors))
    except Exception:
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import tensorflow as tf

    print("=" * 60)
    print("  DefectSense — LSTM Autoencoder Training")
    print("=" * 60)
    print(f"  TensorFlow version : {tf.__version__}")
    print(f"  Sequence length    : {SEQUENCE_LENGTH}")
    print(f"  Features           : {FEATURES}")
    print()

    sequences, _, scaler = load_and_prepare(DATA_PATH)

    # Train / validation split
    split = int(len(sequences) * (1 - VALIDATION_SPLIT))
    train_seq, val_seq = sequences[:split], sequences[split:]
    print(f"Train sequences    : {len(train_seq):,}")
    print(f"Val   sequences    : {len(val_seq):,}\n")

    model = build_model()
    model.summary()
    print()

    # ── MLflow Tracking ───────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    Path(tracking_uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("defectsense_anomaly_detection")

    with mlflow.start_run(run_name="lstm_autoencoder"):
        mlflow.log_params({
            "sequence_length": SEQUENCE_LENGTH,
            "n_features": N_FEATURES,
            "lstm_units_1": LSTM_UNITS_1,
            "lstm_units_2": LSTM_UNITS_2,
            "dense_units": DENSE_UNITS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "threshold_multiplier": THRESHOLD_MULTIPLIER,
        })

        # Train
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ]

        history = model.fit(
            train_seq, train_seq,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(val_seq, val_seq),
            callbacks=callbacks,
            verbose=1,
        )

        # Compute threshold and post-training AUC
        threshold, mean_err, std_err = compute_threshold(model, val_seq)
        train_loss = float(history.history["loss"][-1])
        val_loss   = float(history.history["val_loss"][-1])

        print("\nComputing post-training AUC on test split...")
        auc = compute_post_training_auc(model, scaler, DATA_PATH)
        print(f"  AUC (test split): {auc:.4f}")

        mlflow.log_metrics({
            "final_train_loss":          train_loss,
            "final_val_loss":            val_loss,
            "reconstruction_error_mean": mean_err,
            "reconstruction_error_std":  std_err,
            "anomaly_threshold":         threshold,
            "auc":                       round(auc, 4),
        })

        # Save model
        model_path = MODELS_DIR / "lstm_autoencoder.keras"
        model.save(str(model_path))
        mlflow.keras.log_model(model, "lstm_autoencoder")
        print(f"\nModel saved      → {model_path}")

        # Save scaler
        scaler_path = MODELS_DIR / "sensor_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved     → {scaler_path}")

        # Save threshold
        threshold_data = {
            "threshold": threshold,
            "mean": mean_err,
            "std": std_err,
            "multiplier": THRESHOLD_MULTIPLIER,
            "sequence_length": SEQUENCE_LENGTH,
            "features": FEATURES,
        }
        threshold_path = MODELS_DIR / "anomaly_threshold.pkl"
        with open(threshold_path, "wb") as f:
            pickle.dump(threshold_data, f)
        print(f"Threshold saved  → {threshold_path}")

        # ── Azure Blob Upload ──────────────────────────────────────────────────
        today = date.today().strftime("%Y%m%d")
        print("\nUploading artefacts to Azure Blob Storage...")
        azure_ok = _upload_to_azure([
            (model_path,     "lstm_autoencoder_latest.keras"),
            (model_path,     f"lstm_autoencoder_{today}.keras"),
            (scaler_path,    "sensor_scaler_latest.pkl"),
            (threshold_path, "anomaly_threshold_latest.pkl"),
        ])

        # ── MLflow Model Registry ──────────────────────────────────────────────
        registered_version = None
        try:
            sys.path.insert(0, str(ROOT))
            from ml.model_registry_service import ModelRegistryService
            registry = ModelRegistryService()
            registry.init()

            model_uri = (
                f"runs:/{mlflow.active_run().info.run_id}"
                f"/lstm_autoencoder"
            )
            result = mlflow.register_model(
                model_uri=model_uri,
                name="defectsense_lstm_autoencoder",
            )

            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name="defectsense_lstm_autoencoder",
                alias="challenger",
                version=result.version,
            )
            registered_version = result.version
        except Exception as exc:
            print(f"Registry warning (non-fatal): {exc}")

        # ── Final Summary ──────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  LSTM Autoencoder Training Complete")
        print("=" * 60)
        reg_str   = f"defectsense_lstm_autoencoder v{registered_version}" if registered_version else "FAILED (see above)"
        azure_str = "lstm_autoencoder_latest.keras [OK]" if azure_ok else "skipped / failed (see above)"
        print(f"  Local:    ml/models/lstm_autoencoder.keras")
        print(f"  Azure:    {azure_str}")
        print(f"  Registry: {reg_str}")
        print(f"  Alias:    challenger")
        print(f"  AUC:      {auc:.4f}")
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
