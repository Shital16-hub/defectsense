"""
ML Inference Service — loads trained LSTM Autoencoder and runs anomaly detection.

Key design decisions:
  - Sequence is built OUTSIDE this service (by AnomalyDetectorAgent from Redis cache)
    so this service is stateless and reusable across workers.
  - All CPU-bound inference runs in a thread-pool executor to avoid blocking the
    async event loop.
  - Each prediction is logged to an MLflow run kept open for the lifetime of the app.
  - is_anomaly = reconstruction_error > threshold  (LSTM only)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json as _json
import pickle
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.models.anomaly import AnomalyResult, FailureType
from app.models.sensor import SensorReading

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "ml" / "models"

AUTOENCODER_PATH = MODELS_DIR / "azure_lstm_autoencoder.keras"
SCALER_PATH      = MODELS_DIR / "azure_sensor_scaler.pkl"
THRESHOLD_PATH   = MODELS_DIR / "azure_anomaly_threshold.pkl"

FEATURES = [
    "volt",
    "rotate",
    "pressure",
    "vibration",
]

N_FEATURES = 4

# ── Sequence length — read from config if available ───────────────────────────
_config_path = Path(__file__).parent.parent.parent / "data" / "azure_lstm_config.json"
try:
    with open(_config_path) as _f:
        SEQUENCE_LENGTH = int(_json.load(_f)["sequence_length"])
except Exception:
    SEQUENCE_LENGTH = 30


class MLService:
    """
    Stateless ML inference service — no internal rolling buffer.
    The caller (AnomalyDetectorAgent) supplies the pre-built sequence from Redis.

    If blob_service is provided and local model files are missing, the service
    will attempt to download them from Azure Blob Storage before loading.
    """

    def __init__(self, blob_service=None) -> None:
        self._autoencoder    = None
        self._scaler         = None
        self._threshold_data: dict = {}
        self._loaded         = False
        self._executor       = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._blob_service   = blob_service  # optional BlobStorageService

        # MLflow prediction tracking
        self._mlflow_run_id: Optional[str] = None
        self._pred_step      = 0
        self._mlflow_lock    = threading.Lock()

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all model artefacts from disk. Call once at app startup.

        If a local file is missing and a BlobStorageService was supplied, the
        method attempts to download the 'latest' blob variant before loading.
        """
        if self._loaded:
            return

        logger.info("MLService: loading artefacts from {}", MODELS_DIR)

        # ── helper: check local file exists (blob storage is disabled) ──────
        def _ensure_local(path: Path, blob_name: str) -> bool:
            if path.exists():
                return True
            logger.error(
                "  MLService: model file not found at '{}' and blob storage is "
                "disabled — run ml/train_autoencoder_azure.py to generate this file.",
                path.name,
            )
            return False

        # ── LSTM Autoencoder ─────────────────────────────────────────────────
        if _ensure_local(AUTOENCODER_PATH, "azure_lstm_autoencoder_latest.keras"):
            import tensorflow as tf  # deferred — slow import
            self._autoencoder = tf.keras.models.load_model(str(AUTOENCODER_PATH))
            logger.info("  ✓ Azure LSTM Autoencoder loaded")
        else:
            logger.warning("  ✗ Autoencoder not found — run ml/train_autoencoder_azure.py")

        # ── Sensor scaler ────────────────────────────────────────────────────
        if _ensure_local(SCALER_PATH, "azure_sensor_scaler_latest.pkl"):
            with open(SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("  ✓ Sensor scaler loaded")

        # ── Anomaly threshold ────────────────────────────────────────────────
        if _ensure_local(THRESHOLD_PATH, "azure_anomaly_threshold_latest.pkl"):
            with open(THRESHOLD_PATH, "rb") as f:
                self._threshold_data = pickle.load(f)
            logger.info(
                "  ✓ Threshold loaded: {:.6f}",
                self._threshold_data.get("threshold", 0.0),
            )

        self._loaded = True
        self._init_mlflow()
        logger.info("MLService ready.")

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._autoencoder is not None

    @property
    def is_blob_available(self) -> bool:
        return (
            self._blob_service is not None
            and self._blob_service.is_available
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def predict_anomaly(
        self,
        reading: SensorReading,
        sequence: Optional[list[SensorReading]] = None,
    ) -> AnomalyResult:
        """
        Run LSTM anomaly detection.

        Args:
            reading:  The current SensorReading to evaluate.
            sequence: Last N SensorReadings for this machine (from Redis cache).
                      If len(sequence) < SEQUENCE_LENGTH, LSTM is skipped and
                      no detection is performed.

        Returns:
            AnomalyResult with scores and sensor deltas.
        """
        if not self._loaded:
            self.load()

        loop = asyncio.get_event_loop()
        raw    = self._to_raw(reading)
        scaled = self._scale(raw)

        lstm_recon_error: Optional[float] = None
        lstm_above_threshold              = False
        ml_model_used                     = "none"

        # ── LSTM Autoencoder (CPU-bound → thread pool) ────────────────────────
        if (
            self._autoencoder is not None
            and sequence is not None
            and len(sequence) >= SEQUENCE_LENGTH
        ):
            seq_array = self._build_sequence_array(sequence[-SEQUENCE_LENGTH:])
            lstm_recon_error = await loop.run_in_executor(
                self._executor, self._run_lstm, seq_array
            )
            threshold = self._threshold_data.get("threshold", 0.0)
            lstm_above_threshold = lstm_recon_error > threshold
            ml_model_used = "lstm_autoencoder"
        elif sequence is not None and len(sequence) < SEQUENCE_LENGTH:
            # Buffer is filling up — model is warming up, not absent
            ml_model_used = "buffering"
            logger.debug(
                "MLService: buffering {}/{} readings for machine sequence",
                len(sequence),
                SEQUENCE_LENGTH,
            )

        is_anomaly  = lstm_above_threshold
        anomaly_score = self._compute_anomaly_score(lstm_recon_error)
        failure_probability = float(np.clip(anomaly_score * 1.15, 0.0, 1.0))

        # ── Sensor deltas ─────────────────────────────────────────────────────
        sensor_deltas = self._compute_deltas(scaled, sequence)

        # ── Failure type classification (rule-based on z-score deltas) ────────
        failure_type = self._classify_failure_type(sensor_deltas) if is_anomaly else None

        result = AnomalyResult(
            machine_id=reading.machine_id,
            timestamp=reading.timestamp,
            anomaly_score=round(anomaly_score, 4),
            failure_probability=round(failure_probability, 4),
            is_anomaly=is_anomaly,
            failure_type_prediction=failure_type,
            sensor_deltas=sensor_deltas,
            ml_model_used=ml_model_used,
            reconstruction_error=lstm_recon_error,
            isolation_score=None,
        )

        # Log to MLflow in background thread
        loop.run_in_executor(self._executor, self._log_to_mlflow, result)

        return result

    # ── Thread-pool workers (synchronous, called via executor) ─────────────────

    def _run_lstm(self, sequence: np.ndarray) -> float:
        """Run LSTM autoencoder and return reconstruction MSE."""
        reconstructed = self._autoencoder.predict(sequence, verbose=0)
        mse = float(np.mean(np.power(sequence - reconstructed, 2)))
        return mse

    # ── MLflow prediction tracking ─────────────────────────────────────────────

    def _init_mlflow(self) -> None:
        try:
            import os
            import mlflow
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("defectsense_live_predictions")
            run = mlflow.start_run(run_name="live_inference")
            self._mlflow_run_id = run.info.run_id
            logger.info("  ✓ MLflow prediction tracking run started: {}", self._mlflow_run_id)
        except Exception as exc:
            logger.warning("MLflow init failed (non-fatal): {}", exc)

    def _log_to_mlflow(self, result: AnomalyResult) -> None:
        if self._mlflow_run_id is None:
            return
        try:
            import mlflow
            with self._mlflow_lock:
                self._pred_step += 1
                step = self._pred_step
            with mlflow.start_run(run_id=self._mlflow_run_id):
                mlflow.log_metrics(
                    {
                        "anomaly_score":        result.anomaly_score,
                        "failure_probability":  result.failure_probability,
                        "is_anomaly":           int(result.is_anomaly),
                        "reconstruction_error": result.reconstruction_error or 0.0,
                    },
                    step=step,
                )
        except Exception as exc:
            logger.debug("MLflow log skipped: {}", exc)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _to_raw(self, reading: SensorReading) -> np.ndarray:
        return np.array([getattr(reading, f) for f in FEATURES], dtype=np.float32)

    def _scale(self, raw: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.transform(raw.reshape(1, -1))[0].astype(np.float32)
        # Fallback: approximate Azure PdM ranges [volt, rotate, pressure, vibration]
        mins = np.array([97.3,  160.3, 51.2, 14.9], dtype=np.float32)
        maxs = np.array([250.9, 683.5, 182.1, 72.3], dtype=np.float32)
        return np.clip((raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)

    def _build_sequence_array(self, readings: list[SensorReading]) -> np.ndarray:
        """Convert list of SensorReadings → (1, SEQUENCE_LENGTH, N_FEATURES) array."""
        rows = [self._scale(self._to_raw(r)) for r in readings]
        return np.array(rows, dtype=np.float32)[np.newaxis, ...]

    def _compute_anomaly_score(self, recon_error: Optional[float]) -> float:
        threshold = self._threshold_data.get("threshold", 1.0) or 1.0

        if recon_error is None:
            return 0.0

        lstm_norm = float(np.clip(recon_error / (threshold * 2.0), 0.0, 1.0))
        return round(lstm_norm, 4)

    def _classify_failure_type(self, sensor_deltas: dict[str, float]) -> Optional[str]:
        """
        Rule-based failure type classification from sensor z-score deltas.

        Azure PdM failure types and their dominant sensor signatures:
          BWF — Bearing Wear Failure:      vibration spike  (vibration z > 2.0)
          RSF — Rotor Speed Failure:       rotation drop    (rotate z < -2.0)
          PBF — Pressure Blockage Failure: pressure spike   (pressure z > 2.0)
          EVF — Electrical Overload Failure: voltage spike  (volt z > 2.0)

        Rules are evaluated in priority order; the first match wins.
        Returns None if no sensor exceeds the threshold (anomaly without
        a clear single-sensor signature).
        """
        vib  = sensor_deltas.get("vibration", 0.0)
        rot  = sensor_deltas.get("rotate",    0.0)
        pres = sensor_deltas.get("pressure",  0.0)
        volt = sensor_deltas.get("volt",      0.0)

        # Priority order matches Azure PdM failure frequency (BWF most common)
        if vib  >  2.0:
            return "BWF"
        if rot  < -2.0:
            return "RSF"
        if pres >  2.0:
            return "PBF"
        if volt >  2.0:
            return "EVF"

        # Secondary thresholds — dominant sensor wins by magnitude
        candidates = {
            "BWF": vib,
            "RSF": -rot,   # negative because RSF is a drop
            "PBF": pres,
            "EVF": volt,
        }
        dominant, magnitude = max(candidates.items(), key=lambda x: x[1])
        if magnitude > 1.0:
            return dominant

        return None

    def _compute_deltas(
        self,
        scaled: np.ndarray,
        sequence: Optional[list[SensorReading]],
    ) -> dict[str, float]:
        if not sequence or len(sequence) < 5:
            return {f: 0.0 for f in FEATURES}
        history = np.array(
            [self._scale(self._to_raw(r)) for r in sequence], dtype=np.float32
        )
        mean    = history.mean(axis=0)
        std     = history.std(axis=0) + 1e-8
        z       = (scaled - mean) / std
        return {f: round(float(v), 3) for f, v in zip(FEATURES, z)}
