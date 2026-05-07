"""
Train LSTM Autoencoder for anomaly detection on Azure PdM sensor data.

Architecture:
    Input (SEQ_LEN, 4) -> Encoder [LSTM-64 -> LSTM-32 -> Dense-16]
                       -> Decoder [RepeatVector(SEQ_LEN) -> LSTM-32 -> LSTM-64
                                  -> TimeDistributed Dense-4]

Sequence length is read from data/azure_lstm_config.json (auto-calculated
from the precursor-window analysis -- never hardcoded here).

Training:
    - Trained on CLEAN NORMAL rows only from azure_sensor_readings
    - Sequences built per-machine (no cross-machine contamination)
    - Anomaly threshold = mean + 3 * std of validation reconstruction errors

Run:
    python ml/train_autoencoder_azure.py

Outputs:
    ml/models/azure_lstm_autoencoder.keras
    ml/models/azure_sensor_scaler.pkl
    ml/models/azure_anomaly_threshold.pkl
"""

import json
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
CONFIG_PATH = ROOT / "data" / "azure_lstm_config.json"
MODELS_DIR  = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -- Fixed hyperparameters -----------------------------------------------------
FEATURES           = ["volt", "rotate", "pressure", "vibration"]
N_FEATURES         = len(FEATURES)
LSTM_UNITS_1       = 64
LSTM_UNITS_2       = 32
DENSE_UNITS        = 16
EPOCHS             = 50
BATCH_SIZE         = int(os.getenv("AZURE_LSTM_BATCH_SIZE", "128"))
VALIDATION_SPLIT   = 0.1
THRESHOLD_MULTIPLIER = 3.0
AI4I_BASELINE_AUC  = 0.455   # AUC of the old AI4I LSTM model


# -- STEP 1 -- Read config ------------------------------------------------------

def load_config() -> dict:
    """Read sequence_length and metadata from azure_lstm_config.json."""
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found at {CONFIG_PATH}")
        print("Run:  python data/load_azure_to_postgres.py  first.")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    seq_len = cfg.get("sequence_length")
    if not seq_len or not isinstance(seq_len, int):
        print(f"ERROR: 'sequence_length' missing or invalid in {CONFIG_PATH}")
        sys.exit(1)

    calc_date = cfg.get("calculation_date", "unknown")[:10]
    print(f"  Sequence length  : {seq_len}  (calculated {calc_date})")
    print(f"  Median precursor : {cfg.get('median_precursor_hours')} hrs")
    print(f"  Total failures   : {cfg.get('total_failure_events')}")
    return cfg


# -- STEP 2 -- Load clean rows from PostgreSQL ----------------------------------

def load_clean_rows() -> pd.DataFrame:
    """
    Query azure_sensor_readings for is_clean_normal = TRUE.
    Exits with a clear message if PostgreSQL is unavailable -- no CSV fallback.
    """
    postgres_url = os.getenv("POSTGRES_URL", "").strip()
    if not postgres_url:
        print("ERROR: POSTGRES_URL not set in .env -- cannot continue.")
        sys.exit(1)

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(
            postgres_url,
            connect_args={
                "sslmode": "require",
                "options": "-c statement_timeout=0",  # disable timeout for large bulk fetch
            },
            pool_pre_ping=True,
        )

        query = text("""
            SELECT machine_id, datetime, volt, rotate, pressure, vibration
            FROM   azure_sensor_readings
            WHERE  is_clean_normal = TRUE
            ORDER  BY machine_id, datetime
        """)

        print("  Fetching 737k rows (this may take 1-2 minutes over the network)...")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, parse_dates=["datetime"])

        engine.dispose()

        if df.empty:
            print("ERROR: No clean normal rows found in azure_sensor_readings.")
            print("Run:  python data/load_azure_to_postgres.py  first.")
            sys.exit(1)

        print(f"  Clean rows loaded: {len(df):,}  ({df['machine_id'].nunique()} machines)")
        return df

    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: PostgreSQL connection failed -- {exc}")
        print("Check POSTGRES_URL in .env and that azure_sensor_readings exists.")
        sys.exit(1)


def load_failure_rows(engine) -> pd.DataFrame:
    """Load all rows needed to build pre-failure sequences."""
    try:
        # We need all rows for machines that have failures so we can
        # look back SEQUENCE_LENGTH steps from each failure event.
        query = text("""
            SELECT machine_id, datetime, volt, rotate, pressure, vibration,
                   machine_failure
            FROM   azure_sensor_readings
            WHERE  machine_id IN (
                SELECT DISTINCT machine_id
                FROM   azure_sensor_readings
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


# -- STEP 3 -- Fit and save scaler ----------------------------------------------

def fit_scaler(df: pd.DataFrame):
    """Fit MinMaxScaler on clean normal sensor columns. Returns (scaler, scaled_array)."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES].values).astype(np.float32)

    print("  Scaler fitted (per-sensor min/max on clean normal rows):")
    for i, feat in enumerate(FEATURES):
        print(f"    {feat:<12}: min={scaler.data_min_[i]:.4f}  max={scaler.data_max_[i]:.4f}")

    scaler_path = MODELS_DIR / "azure_sensor_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved -> {scaler_path}")

    return scaler, scaled


# -- STEP 4 -- Build per-machine sequences --------------------------------------

def build_sequences(df: pd.DataFrame, scaler, sequence_length: int) -> np.ndarray:
    """
    Build sliding-window sequences of shape (sequence_length, N_FEATURES)
    separately for each machine to avoid cross-machine contamination.
    """
    sequences = []
    machine_counts = {}

    for machine_id in sorted(df["machine_id"].unique()):
        machine_df = (
            df[df["machine_id"] == machine_id]
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        scaled = scaler.transform(machine_df[FEATURES].values).astype(np.float32)

        n_seqs = 0
        for i in range(len(scaled) - sequence_length):
            sequences.append(scaled[i : i + sequence_length])
            n_seqs += 1

        machine_counts[machine_id] = n_seqs

    total     = len(sequences)
    avg_count = total / len(machine_counts) if machine_counts else 0
    print(f"  Total sequences  : {total:,}  across {len(machine_counts)} machines")
    print(f"  Avg per machine  : {avg_count:.1f}")

    return np.array(sequences, dtype=np.float32)


# -- STEP 6 -- Build model ------------------------------------------------------

def build_model(sequence_length: int):
    """LSTM Autoencoder -- same architecture as train_autoencoder.py, 4 sensors."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = tf.keras.Input(shape=(sequence_length, N_FEATURES))

    # Encoder
    x       = layers.LSTM(LSTM_UNITS_1, return_sequences=True)(inputs)
    x       = layers.LSTM(LSTM_UNITS_2, return_sequences=False)(x)
    encoded = layers.Dense(DENSE_UNITS, activation="relu")(x)

    # Decoder
    x       = layers.RepeatVector(sequence_length)(encoded)
    x       = layers.LSTM(LSTM_UNITS_2, return_sequences=True)(x)
    x       = layers.LSTM(LSTM_UNITS_1, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(N_FEATURES))(x)

    model = Model(inputs, decoded, name="azure_lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# -- STEP 8 -- Compute threshold ------------------------------------------------

def compute_threshold(
    model, val_sequences: np.ndarray
) -> tuple[float, float, float]:
    """Reconstruction MSE on val set -> mean + MULTIPLIER * std."""
    reconstructed    = model.predict(val_sequences, verbose=0)
    mse_per_sample   = np.mean(np.power(val_sequences - reconstructed, 2), axis=(1, 2))
    mean_err         = float(np.mean(mse_per_sample))
    std_err          = float(np.std(mse_per_sample))
    threshold        = mean_err + THRESHOLD_MULTIPLIER * std_err
    return threshold, mean_err, std_err


# -- STEP 9 -- Evaluate on failure sequences ------------------------------------

def evaluate_on_failures(
    model,
    scaler,
    postgres_url: str,
    sequence_length: int,
    threshold: float,
    val_sequences: np.ndarray,
) -> tuple[float, float]:
    """
    Build pre-failure sequences, compute reconstruction errors, return
    (detection_rate, auc).  Both values are 0.0 on any failure.
    """
    try:
        from sklearn.metrics import roc_auc_score
        from sqlalchemy import create_engine

        engine = create_engine(
            postgres_url,
            connect_args={
                "sslmode": "require",
                "options": "-c statement_timeout=0",
            },
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
                continue   # not enough history before this failure

            scaled = scaler.transform(
                machine_hist[FEATURES].values
            ).astype(np.float32)
            failure_sequences.append(scaled)

        if not failure_sequences:
            print("  WARNING: No complete pre-failure sequences built.")
            return 0.0, 0.0

        fail_seqs  = np.array(failure_sequences, dtype=np.float32)
        fail_recon = model.predict(fail_seqs, verbose=0)
        fail_mse   = np.mean(np.power(fail_seqs - fail_recon, 2), axis=(1, 2))

        detection_rate = float(np.mean(fail_mse > threshold))
        print(f"  Pre-failure sequences used: {len(fail_seqs):,}")
        print(f"  Detection rate (>{THRESHOLD_MULTIPLIER}sd threshold): {detection_rate:.1%}")

        # AUC -- sample up to 5000 clean val sequences vs all failure sequences
        n_clean_sample = min(5000, len(val_sequences))
        idx_sample     = np.random.default_rng(42).choice(
            len(val_sequences), size=n_clean_sample, replace=False
        )
        clean_sample   = val_sequences[idx_sample]
        clean_recon    = model.predict(clean_sample, verbose=0)
        clean_mse      = np.mean(np.power(clean_sample - clean_recon, 2), axis=(1, 2))

        errors = np.concatenate([clean_mse, fail_mse])
        labels = np.concatenate([
            np.zeros(len(clean_mse), dtype=int),
            np.ones(len(fail_mse),   dtype=int),
        ])
        auc = float(roc_auc_score(labels, errors))
        print(f"  AUC (clean vs failure): {auc:.4f}")
        print(
            f"  vs AI4I baseline AUC {AI4I_BASELINE_AUC:.3f}: "
            f"{'IMPROVED (improved)' if auc > AI4I_BASELINE_AUC else 'NO IMPROVEMENT (no improvement)'}"
        )
        return detection_rate, auc

    except Exception as exc:
        print(f"  WARNING: Failure evaluation failed (non-fatal) -- {exc}")
        return 0.0, 0.0




# -- Main ----------------------------------------------------------------------

def main() -> None:
    import tensorflow as tf

    # -- STEP 1 -- Config -------------------------------------------------------
    print("=" * 62)
    print("  DefectSense -- Azure PdM LSTM Autoencoder Training")
    print("=" * 62)
    print(f"  TensorFlow version : {tf.__version__}")

    print("\n[STEP 1] Reading config from azure_lstm_config.json...")
    cfg             = load_config()
    SEQUENCE_LENGTH = cfg["sequence_length"]
    print(f"  SEQUENCE_LENGTH = {SEQUENCE_LENGTH}")

    # -- STEP 2 -- Load clean rows ----------------------------------------------
    print("\n[STEP 2] Loading clean normal rows from PostgreSQL...")
    clean_df = load_clean_rows()

    # -- STEP 3 -- Fit scaler ---------------------------------------------------
    print("\n[STEP 3] Fitting MinMaxScaler on clean normal rows...")
    scaler, _ = fit_scaler(clean_df)

    # -- STEP 4 -- Build per-machine sequences ----------------------------------
    print(f"\n[STEP 4] Building per-machine sequences (length={SEQUENCE_LENGTH})...")
    sequences = build_sequences(clean_df, scaler, SEQUENCE_LENGTH)
    n_clean_sequences = len(sequences)

    # -- STEP 4 (sampling) -- Cap sequences if over MAX_TRAIN_SEQUENCES ---------
    max_seqs = int(os.getenv("AZURE_LSTM_MAX_SEQUENCES", "50000"))
    print(f"  Original sequences built  : {n_clean_sequences:,}")
    print(f"  MAX_TRAIN_SEQUENCES limit  : {max_seqs:,}")
    if n_clean_sequences > max_seqs:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_clean_sequences, size=max_seqs, replace=False)
        idx.sort()
        sequences = sequences[idx]
        n_clean_sequences = len(sequences)
    print(f"  Final sequences after sampling: {n_clean_sequences:,}")

    # Estimated training time
    steps_per_epoch = int(n_clean_sequences * (1 - VALIDATION_SPLIT) / BATCH_SIZE)
    mins_per_epoch  = steps_per_epoch * 0.161 / 60
    mins_total_5ep  = mins_per_epoch * 5
    print(f"\n  Steps per epoch            : {steps_per_epoch:,}")
    print(f"  Time per epoch at 161ms/step: {mins_per_epoch:.1f} min")
    print(f"  Estimated total for 5 epochs: {mins_total_5ep:.1f} min")
    print(f"  Ready to train. Expected time roughly {mins_total_5ep:.0f} minutes on CPU.")

    # -- STEP 5 -- Train / validation split -------------------------------------
    print("\n[STEP 5] Train / validation split (last 10% -> validation)...")
    split    = int(len(sequences) * (1 - VALIDATION_SPLIT))
    train_seq, val_seq = sequences[:split], sequences[split:]
    print(f"  Train sequences : {len(train_seq):,}")
    print(f"  Val   sequences : {len(val_seq):,}")

    # -- STEP 6 -- Build model --------------------------------------------------
    print("\n[STEP 6] Building LSTM Autoencoder model...")
    model = build_model(SEQUENCE_LENGTH)
    model.summary()

    # -- MLflow setup (wraps steps 7-12) ---------------------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    tracking_db  = Path(tracking_uri.replace("sqlite:///", ""))
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("defectsense_anomaly_detection")

    with mlflow.start_run(run_name="azure_lstm_autoencoder"):

        mlflow.log_params({
            "sequence_length":       SEQUENCE_LENGTH,
            "n_sensors":             N_FEATURES,
            "n_clean_sequences":     n_clean_sequences,
            "features":              ",".join(FEATURES),
            "dataset":               "azure_pdm",
            "lstm_units_1":          LSTM_UNITS_1,
            "lstm_units_2":          LSTM_UNITS_2,
            "dense_units":           DENSE_UNITS,
            "epochs":                EPOCHS,
            "batch_size":            BATCH_SIZE,
            "threshold_multiplier":  THRESHOLD_MULTIPLIER,
        })

        # -- STEP 7 -- Train ----------------------------------------------------
        print("\n[STEP 7] Training...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, min_delta=0.0001,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, verbose=1,
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

        train_loss = float(history.history["loss"][-1])
        val_loss   = float(history.history["val_loss"][-1])
        epochs_ran = len(history.history["loss"])
        print(f"\n  Epochs run       : {epochs_ran}")
        print(f"  Final train loss : {train_loss:.6f}")
        print(f"  Final val loss   : {val_loss:.6f}")

        # -- STEP 8 -- Compute threshold ----------------------------------------
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
        threshold_path = MODELS_DIR / "azure_anomaly_threshold.pkl"
        with open(threshold_path, "wb") as f:
            pickle.dump(threshold_data, f)
        print(f"  Saved -> {threshold_path}")

        # -- STEP 9 -- Evaluate on failure sequences ----------------------------
        print("\n[STEP 9] Evaluating on pre-failure sequences from PostgreSQL...")
        postgres_url   = os.getenv("POSTGRES_URL", "").strip()
        detection_rate, auc = evaluate_on_failures(
            model, scaler, postgres_url, SEQUENCE_LENGTH, threshold, val_seq
        )

        # -- STEP 10 -- Save model ----------------------------------------------
        print("\n[STEP 10] Saving model...")
        model_path = MODELS_DIR / "azure_lstm_autoencoder.keras"
        model.save(str(model_path))
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Saved -> {model_path}  ({size_mb:.2f} MB)")

        mlflow.keras.log_model(model, "azure_lstm_autoencoder")

        # -- STEP 11 -- Local storage (blob storage disabled) ------------------
        print("\n[STEP 11] Models saved locally to ml/models/ — blob storage disabled.")

        # -- Log all metrics to MLflow -----------------------------------------
        mlflow.log_metrics({
            "final_train_loss":          train_loss,
            "final_val_loss":            val_loss,
            "reconstruction_error_mean": mean_err,
            "reconstruction_error_std":  std_err,
            "anomaly_threshold":         threshold,
            "auc":                       round(auc, 4),
            "detection_rate":            round(detection_rate, 4),
        })

        # -- STEP 12 -- MLflow model registration ------------------------------
        print("\n[STEP 12] Registering model in MLflow registry...")
        registered_version = None
        registered_alias   = "challenger"
        try:
            sys.path.insert(0, str(ROOT))
            from ml.model_registry_service import ModelRegistryService

            registry = ModelRegistryService()
            registry.init()

            model_uri = (
                f"runs:/{mlflow.active_run().info.run_id}"
                f"/azure_lstm_autoencoder"
            )
            result = mlflow.register_model(
                model_uri=model_uri,
                name="defectsense_lstm_autoencoder_azure",
            )

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

        # -- STEP 13 -- Final summary -------------------------------------------
        print("\n[STEP 13] Final summary")
        print()
        print("=" * 62)
        print("  AZURE PdM LSTM AUTOENCODER -- TRAINING COMPLETE")
        print("=" * 62)
        print(f"  Dataset              : Azure PdM (100 machines, 1 yr)")
        print(f"  Clean sequences      : {n_clean_sequences:,}")
        print(f"  Sequence length      : {SEQUENCE_LENGTH}  (from config)")
        print(f"  Sensors              : {', '.join(FEATURES)}")
        print(f"  AUC                  : {auc:.4f}  (vs AI4I baseline {AI4I_BASELINE_AUC:.3f})")
        print(f"  Detection rate       : {detection_rate:.1%}")
        print(f"  Model saved          : ml/models/azure_lstm_autoencoder.keras")
        print(f"  Blob storage         : disabled (local storage only)")
        if registered_version:
            print(f"  MLflow registry      : defectsense_lstm_autoencoder_azure v{registered_version} [{registered_alias}]")
        else:
            print(f"  MLflow registry      : FAILED (see above)")
        improved = auc > AI4I_BASELINE_AUC
        print(f"  Improvement vs AI4I  : {'YES (improved)' if improved else 'NO (no improvement)'}  ({auc:.4f} vs {AI4I_BASELINE_AUC:.3f})")
        print("=" * 62)


if __name__ == "__main__":
    main()
