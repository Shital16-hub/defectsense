"""
Evaluate Azure PdM LSTM Autoencoder -- Steps 8-13.

Loads the model from ml/models/azure_lstm_autoencoder.keras if it exists.
If the model does not exist (training was killed before Step 10), retrains
a fresh model for 3 epochs using the saved scaler and PostgreSQL data, then
runs evaluation.

Primary output: AUC score vs AI4I baseline 0.455.

Run:
    python ml/evaluate_azure_lstm.py
"""

import json
import os
import pickle
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd

# -- Constants (must match train_autoencoder_azure.py) -------------------------
FEATURES             = ["volt", "rotate", "pressure", "vibration"]
N_FEATURES           = len(FEATURES)
LSTM_UNITS_1         = 64
LSTM_UNITS_2         = 32
DENSE_UNITS          = 16
BATCH_SIZE           = int(os.getenv("AZURE_LSTM_BATCH_SIZE", "128"))
VALIDATION_SPLIT     = 0.1
THRESHOLD_MULTIPLIER = 3.0
AI4I_BASELINE_AUC    = 0.455
MAX_SEQUENCES        = int(os.getenv("AZURE_LSTM_MAX_SEQUENCES", "50000"))
RETRAIN_EPOCHS       = 3   # used only when model file is missing

CONFIG_PATH = ROOT / "data" / "azure_lstm_config.json"
MODELS_DIR  = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH    = MODELS_DIR / "azure_sensor_scaler.pkl"
MODEL_PATH     = MODELS_DIR / "azure_lstm_autoencoder.keras"
THRESHOLD_PATH = MODELS_DIR / "azure_anomaly_threshold.pkl"


# -- Helpers -------------------------------------------------------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    seq_len = cfg.get("sequence_length")
    if not seq_len or not isinstance(seq_len, int):
        print(f"ERROR: 'sequence_length' missing or invalid in {CONFIG_PATH}")
        sys.exit(1)
    return cfg


def load_clean_rows() -> pd.DataFrame:
    postgres_url = os.getenv("POSTGRES_URL", "").strip()
    if not postgres_url:
        print("ERROR: POSTGRES_URL not set in .env")
        sys.exit(1)
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
            postgres_url,
            connect_args={"sslmode": "require", "options": "-c statement_timeout=0"},
            pool_pre_ping=True,
        )
        query = text("""
            SELECT machine_id, datetime, volt, rotate, pressure, vibration
            FROM   azure_sensor_readings
            WHERE  is_clean_normal = TRUE
            ORDER  BY machine_id, datetime
        """)
        print("  Fetching 737k rows (this may take 1-2 minutes)...")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, parse_dates=["datetime"])
        engine.dispose()
        if df.empty:
            print("ERROR: No clean normal rows found.")
            sys.exit(1)
        print(f"  Clean rows loaded: {len(df):,}  ({df['machine_id'].nunique()} machines)")
        return df
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: PostgreSQL connection failed -- {exc}")
        sys.exit(1)


def load_failure_rows(engine) -> pd.DataFrame:
    try:
        from sqlalchemy import text
        query = text("""
            SELECT machine_id, datetime, volt, rotate, pressure, vibration, machine_failure
            FROM   azure_sensor_readings
            WHERE  machine_id IN (
                SELECT DISTINCT machine_id FROM azure_sensor_readings
                WHERE  machine_failure = 1
            )
            ORDER  BY machine_id, datetime
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, parse_dates=["datetime"])
        return df
    except Exception as exc:
        print(f"  WARNING: Could not load failure rows -- {exc}")
        return pd.DataFrame()


def build_sequences(df: pd.DataFrame, scaler, sequence_length: int) -> np.ndarray:
    sequences = []
    machine_counts = {}
    for machine_id in sorted(df["machine_id"].unique()):
        mdf = df[df["machine_id"] == machine_id].sort_values("datetime").reset_index(drop=True)
        scaled = scaler.transform(mdf[FEATURES].values).astype(np.float32)
        n = 0
        for i in range(len(scaled) - sequence_length):
            sequences.append(scaled[i: i + sequence_length])
            n += 1
        machine_counts[machine_id] = n
    total = len(sequences)
    avg   = total / len(machine_counts) if machine_counts else 0
    print(f"  Total sequences  : {total:,}  across {len(machine_counts)} machines")
    print(f"  Avg per machine  : {avg:.1f}")
    return np.array(sequences, dtype=np.float32)


def build_model(sequence_length: int):
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inputs  = tf.keras.Input(shape=(sequence_length, N_FEATURES))
    x       = layers.LSTM(LSTM_UNITS_1, return_sequences=True)(inputs)
    x       = layers.LSTM(LSTM_UNITS_2, return_sequences=False)(x)
    encoded = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x       = layers.RepeatVector(sequence_length)(encoded)
    x       = layers.LSTM(LSTM_UNITS_2, return_sequences=True)(x)
    x       = layers.LSTM(LSTM_UNITS_1, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(N_FEATURES))(x)
    model   = tf.keras.Model(inputs, decoded, name="azure_lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def compute_threshold(model, val_sequences: np.ndarray) -> tuple:
    reconstructed  = model.predict(val_sequences, verbose=0)
    mse            = np.mean(np.power(val_sequences - reconstructed, 2), axis=(1, 2))
    mean_err       = float(np.mean(mse))
    std_err        = float(np.std(mse))
    threshold      = mean_err + THRESHOLD_MULTIPLIER * std_err
    return threshold, mean_err, std_err


def evaluate_on_failures(model, scaler, postgres_url, sequence_length, threshold, val_sequences) -> tuple:
    try:
        from sklearn.metrics import roc_auc_score
        from sqlalchemy import create_engine
        engine = create_engine(
            postgres_url,
            connect_args={"sslmode": "require", "options": "-c statement_timeout=0"},
            pool_pre_ping=True,
        )
        all_machine_df = load_failure_rows(engine)
        engine.dispose()

        if all_machine_df.empty:
            print("  WARNING: No failure rows found -- skipping evaluation.")
            return 0.0, 0.0

        failure_rows = all_machine_df[all_machine_df["machine_failure"] == 1]
        print(f"  Failure events found: {len(failure_rows):,}")

        failure_sequences = []
        for _, frow in failure_rows.iterrows():
            mid    = frow["machine_id"]
            f_time = frow["datetime"]
            machine_hist = (
                all_machine_df[
                    (all_machine_df["machine_id"] == mid) &
                    (all_machine_df["datetime"] < f_time)
                ]
                .sort_values("datetime")
                .tail(sequence_length)
            )
            if len(machine_hist) < sequence_length:
                continue
            scaled = scaler.transform(machine_hist[FEATURES].values).astype(np.float32)
            failure_sequences.append(scaled)

        if not failure_sequences:
            print("  WARNING: No complete pre-failure sequences built.")
            return 0.0, 0.0

        fail_seqs  = np.array(failure_sequences, dtype=np.float32)
        fail_recon = model.predict(fail_seqs, verbose=0)
        fail_mse   = np.mean(np.power(fail_seqs - fail_recon, 2), axis=(1, 2))

        detection_rate = float(np.mean(fail_mse > threshold))
        print(f"  Pre-failure sequences used : {len(fail_seqs):,}")
        print(f"  Detection rate (>{THRESHOLD_MULTIPLIER}sd threshold): {detection_rate:.1%}")

        n_clean_sample = min(5000, len(val_sequences))
        idx_sample     = np.random.default_rng(42).choice(len(val_sequences), size=n_clean_sample, replace=False)
        clean_sample   = val_sequences[idx_sample]
        clean_recon    = model.predict(clean_sample, verbose=0)
        clean_mse      = np.mean(np.power(clean_sample - clean_recon, 2), axis=(1, 2))

        errors = np.concatenate([clean_mse, fail_mse])
        labels = np.concatenate([
            np.zeros(len(clean_mse), dtype=int),
            np.ones(len(fail_mse),   dtype=int),
        ])
        auc = float(roc_auc_score(labels, errors))
        print(f"  AUC (clean vs failure) : {auc:.4f}")
        print(f"  vs AI4I baseline {AI4I_BASELINE_AUC:.3f} : {'IMPROVED' if auc > AI4I_BASELINE_AUC else 'NO IMPROVEMENT'}")
        return detection_rate, auc

    except Exception as exc:
        print(f"  WARNING: Failure evaluation failed (non-fatal) -- {exc}")
        import traceback; traceback.print_exc()
        return 0.0, 0.0


def _upload_to_azure(uploads: list) -> bool:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name    = os.getenv("AZURE_STORAGE_CONTAINER", "defectsense-models")
    if not connection_string:
        print("  Azure: AZURE_STORAGE_CONNECTION_STRING not set -- skipping upload.")
        return False
    try:
        sys.path.insert(0, str(ROOT))
        from app.services.blob_storage_service import BlobStorageService
        blob_service = BlobStorageService(connection_string, container_name)
        if not blob_service.is_available:
            print("  Azure: blob storage unavailable -- skipping upload.")
            return False
        all_ok = True
        for local_path, blob_name in uploads:
            ok     = blob_service.upload_model(local_path, blob_name)
            status = "OK" if ok else "FAILED"
            all_ok = all_ok and ok
            print(f"  Azure upload [{status:6s}]: {blob_name}")
        return all_ok
    except Exception as exc:
        print(f"  Azure upload error (non-fatal): {exc}")
        return False


# -- Main ----------------------------------------------------------------------

def main() -> None:
    import tensorflow as tf

    print("=" * 62)
    print("  DefectSense -- Azure LSTM Evaluation (Steps 8-13)")
    print("=" * 62)
    print(f"  TensorFlow version : {tf.__version__}")

    # -- Step 1: Config ---------------------------------------------------------
    print("\n[STEP 1] Reading config from azure_lstm_config.json...")
    cfg             = load_config()
    SEQUENCE_LENGTH = cfg["sequence_length"]
    print(f"  SEQUENCE_LENGTH = {SEQUENCE_LENGTH}")

    postgres_url = os.getenv("POSTGRES_URL", "").strip()

    # -- Step 2: Load / Train model ---------------------------------------------
    if MODEL_PATH.exists():
        print(f"\n[STEP 2] Loading existing model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        model.summary()

        # Load scaler
        print(f"\n[STEP 3] Loading scaler from {SCALER_PATH}...")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        # Need val sequences for threshold + AUC -- rebuild from postgres
        print("\n[STEP 4] Loading clean rows for threshold/AUC computation...")
        clean_df    = load_clean_rows()
        print("\n[STEP 5] Building sequences...")
        sequences   = build_sequences(clean_df, scaler, SEQUENCE_LENGTH)
        n           = len(sequences)
        if n > MAX_SEQUENCES:
            rng       = np.random.default_rng(42)
            idx       = rng.choice(n, size=MAX_SEQUENCES, replace=False)
            idx.sort()
            sequences = sequences[idx]
        split         = int(len(sequences) * (1 - VALIDATION_SPLIT))
        val_seq       = sequences[split:]
        train_loss    = None
        val_loss_hist = None
        epochs_ran    = 0

    else:
        print(f"\n  Model file NOT found at {MODEL_PATH}")
        print(f"  Retraining for {RETRAIN_EPOCHS} epochs from scratch...")

        # Load / fit scaler
        if SCALER_PATH.exists():
            print(f"\n[STEP 3] Loading saved scaler from {SCALER_PATH}...")
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            print("\n[STEP 2] Loading clean rows from PostgreSQL...")
            clean_df = load_clean_rows()
        else:
            print("\n[STEP 2] Loading clean rows from PostgreSQL...")
            clean_df = load_clean_rows()
            print("\n[STEP 3] Fitting scaler on clean rows...")
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.fit(clean_df[FEATURES].values)
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(scaler, f)
            print(f"  Saved -> {SCALER_PATH}")

        print("\n[STEP 4] Building per-machine sequences...")
        sequences = build_sequences(clean_df, scaler, SEQUENCE_LENGTH)
        n         = len(sequences)
        print(f"  Original : {n:,}")
        if n > MAX_SEQUENCES:
            rng       = np.random.default_rng(42)
            idx       = rng.choice(n, size=MAX_SEQUENCES, replace=False)
            idx.sort()
            sequences = sequences[idx]
            print(f"  Sampled  : {len(sequences):,}")

        split   = int(len(sequences) * (1 - VALIDATION_SPLIT))
        train_s = sequences[:split]
        val_seq = sequences[split:]
        print(f"  Train: {len(train_s):,}   Val: {len(val_seq):,}")

        print(f"\n[STEP 5] Building model...")
        model = build_model(SEQUENCE_LENGTH)
        model.summary()

        print(f"\n[STEP 6] Training {RETRAIN_EPOCHS} epochs...")
        history = model.fit(
            train_s, train_s,
            epochs=RETRAIN_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(val_seq, val_seq),
            verbose=1,
        )
        train_loss    = float(history.history["loss"][-1])
        val_loss_hist = float(history.history["val_loss"][-1])
        epochs_ran    = len(history.history["loss"])
        print(f"  Epochs ran       : {epochs_ran}")
        print(f"  Final train loss : {train_loss:.6f}")
        print(f"  Final val loss   : {val_loss_hist:.6f}")

    # -- Step 8: Threshold -------------------------------------------------------
    print("\n[STEP 8] Computing anomaly threshold on validation set...")
    threshold, mean_err, std_err = compute_threshold(model, val_seq)
    print(f"  Mean recon error : {mean_err:.6f}")
    print(f"  Std  recon error : {std_err:.6f}")
    print(f"  Threshold        : {threshold:.6f}  (mean + {THRESHOLD_MULTIPLIER}sd)")

    threshold_data = {
        "threshold":       threshold,
        "mean":            mean_err,
        "std":             std_err,
        "multiplier":      THRESHOLD_MULTIPLIER,
        "sequence_length": SEQUENCE_LENGTH,
        "features":        FEATURES,
    }
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(threshold_data, f)
    print(f"  Saved -> {THRESHOLD_PATH}")

    # -- Step 9: Evaluate on failures -------------------------------------------
    print("\n[STEP 9] Evaluating on pre-failure sequences from PostgreSQL...")
    detection_rate, auc = evaluate_on_failures(
        model, scaler, postgres_url, SEQUENCE_LENGTH, threshold, val_seq
    )

    # -- Step 10: Save model ----------------------------------------------------
    print("\n[STEP 10] Saving model...")
    model.save(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved -> {MODEL_PATH}  ({size_mb:.2f} MB)")

    # -- MLflow logging ---------------------------------------------------------
    mlflow.set_experiment("defectsense_anomaly_detection")
    with mlflow.start_run(run_name="azure_lstm_autoencoder_eval"):
        mlflow.keras.log_model(model, "azure_lstm_autoencoder")
        metrics = {
            "reconstruction_error_mean": mean_err,
            "reconstruction_error_std":  std_err,
            "anomaly_threshold":         threshold,
            "auc":                       round(auc, 4),
            "detection_rate":            round(detection_rate, 4),
        }
        if train_loss is not None:
            metrics["final_train_loss"] = train_loss
            metrics["final_val_loss"]   = val_loss_hist
        mlflow.log_metrics(metrics)

        # -- Step 11: Azure Blob -------------------------------------------------
        today = date.today().strftime("%Y%m%d")
        print("\n[STEP 11] Uploading artefacts to Azure Blob Storage...")
        azure_ok = _upload_to_azure([
            (MODEL_PATH,     "azure_lstm_autoencoder_latest.keras"),
            (MODEL_PATH,     f"azure_lstm_autoencoder_{today}.keras"),
            (SCALER_PATH,    "azure_sensor_scaler_latest.pkl"),
            (THRESHOLD_PATH, "azure_anomaly_threshold_latest.pkl"),
        ])

        # -- Step 12: MLflow registry -------------------------------------------
        print("\n[STEP 12] Registering model in MLflow registry...")
        registered_version = None
        registered_alias   = "challenger"
        try:
            sys.path.insert(0, str(ROOT))
            from ml.model_registry_service import ModelRegistryService
            registry = ModelRegistryService()
            registry.init()

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/azure_lstm_autoencoder"
            result    = mlflow.register_model(model_uri=model_uri, name="defectsense_lstm_autoencoder_azure")

            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name="defectsense_lstm_autoencoder_azure",
                alias=registered_alias,
                version=result.version,
            )
            registered_version = result.version
            print(f"  Registered: defectsense_lstm_autoencoder_azure v{registered_version}")
            print(f"  Alias: {registered_alias}")
        except Exception as exc:
            print(f"  Registry warning (non-fatal): {exc}")

    # -- Step 13: Final summary -------------------------------------------------
    print("\n[STEP 13] Final summary")
    print()
    print("=" * 62)
    print("  AZURE PdM LSTM AUTOENCODER -- EVALUATION COMPLETE")
    print("=" * 62)
    print(f"  Dataset              : Azure PdM (100 machines, 1 yr)")
    print(f"  Sequence length      : {SEQUENCE_LENGTH}  (from config)")
    print(f"  Sensors              : {', '.join(FEATURES)}")
    print(f"  AUC                  : {auc:.4f}  (vs AI4I baseline {AI4I_BASELINE_AUC:.3f})")
    print(f"  Detection rate       : {detection_rate:.1%}")
    print(f"  Model saved          : ml/models/azure_lstm_autoencoder.keras")
    azure_str = "OK" if azure_ok else "FAILED (see above)"
    print(f"  Azure blob           : {azure_str}")
    if registered_version:
        print(f"  MLflow registry      : defectsense_lstm_autoencoder_azure v{registered_version} [{registered_alias}]")
    else:
        print(f"  MLflow registry      : FAILED (see above)")
    improved = auc > AI4I_BASELINE_AUC
    print(f"  Improvement vs AI4I  : {'YES (improved)' if improved else 'NO (no improvement)'}  ({auc:.4f} vs {AI4I_BASELINE_AUC:.3f})")
    print("=" * 62)


if __name__ == "__main__":
    main()
