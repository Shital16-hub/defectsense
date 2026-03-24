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
from pathlib import Path

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

    # Filter normal samples only for training
    normal = df[df["machine_failure"] == 0][FEATURES].dropna()
    print(f"Normal samples : {len(normal):,} / {len(df):,} total")

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
    mlflow.set_tracking_uri("./mlruns")
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

        # Compute threshold
        threshold, mean_err, std_err = compute_threshold(model, val_seq)
        train_loss = float(history.history["loss"][-1])
        val_loss = float(history.history["val_loss"][-1])

        mlflow.log_metrics({
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "reconstruction_error_mean": mean_err,
            "reconstruction_error_std": std_err,
            "anomaly_threshold": threshold,
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

        print()
        print("=" * 60)
        print("  Training Complete")
        print("=" * 60)
        print(f"  Final train loss : {train_loss:.6f}")
        print(f"  Final val loss   : {val_loss:.6f}")
        print(f"  Recon error mean : {mean_err:.6f}")
        print(f"  Recon error std  : {std_err:.6f}")
        print(f"  Anomaly threshold: {threshold:.6f}  (mean + {THRESHOLD_MULTIPLIER}×std)")
        print()
        print(f"  Model trained. Threshold: {threshold:.4f}. Saved to ml/models/")
        print("=" * 60)


if __name__ == "__main__":
    main()
