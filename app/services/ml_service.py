"""
ML Inference Service — loads trained LSTM Autoencoder + Isolation Forest and runs
ensemble anomaly detection on a single SensorReading.

Usage:
    from app.services.ml_service import MLService
    service = MLService()          # loads models once at startup
    result  = await service.predict(sensor_reading)
"""
from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.models.anomaly import AnomalyResult, FailureType
from app.models.sensor import SensorReading

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "ml" / "models"

AUTOENCODER_PATH = MODELS_DIR / "lstm_autoencoder.keras"
SCALER_PATH      = MODELS_DIR / "sensor_scaler.pkl"
THRESHOLD_PATH   = MODELS_DIR / "anomaly_threshold.pkl"
IFOREST_PATH     = MODELS_DIR / "isolation_forest.pkl"

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

# Rule-based failure-type heuristics (mirrors AI4I domain knowledge)
# Each entry: (feature, direction, threshold_z) → failure_type
_FAILURE_RULES: list[tuple[str, str, float, str]] = [
    ("tool_wear",          "high",  2.0, "TWF"),  # Tool Wear Failure
    ("air_temperature",    "high",  2.5, "HDF"),  # Heat Dissipation Failure
    ("process_temperature","high",  2.5, "HDF"),
    ("rotational_speed",   "low",   2.0, "PWF"),  # Power Failure (low speed)
    ("torque",             "high",  2.5, "PWF"),  # Power Failure (high torque)
    ("rotational_speed",   "high",  2.5, "OSF"),  # Overstrain Failure
]


class MLService:
    """
    Singleton-style ML inference service.

    Maintains a rolling window of the last SEQUENCE_LENGTH sensor readings
    per machine to build LSTM input sequences on the fly.
    """

    SEQUENCE_LENGTH: int = 30

    def __init__(self) -> None:
        self._autoencoder = None
        self._scaler      = None
        self._threshold_data: dict = {}
        self._iforest     = None
        self._loaded      = False

        # Per-machine rolling buffers: machine_id → deque of scaled feature vectors
        self._buffers: dict[str, deque] = {}

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all model artefacts from disk (call once at startup)."""
        if self._loaded:
            return

        logger.info("MLService: loading model artefacts from {}", MODELS_DIR)

        # LSTM Autoencoder (TensorFlow / Keras)
        if not AUTOENCODER_PATH.exists():
            logger.warning(
                "Autoencoder not found at {}. Run ml/train_autoencoder.py first.", AUTOENCODER_PATH
            )
        else:
            import tensorflow as tf  # deferred import — avoids slow startup when not needed
            self._autoencoder = tf.keras.models.load_model(str(AUTOENCODER_PATH))
            logger.info("  ✓ LSTM Autoencoder loaded")

        # Scaler
        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("  ✓ Sensor scaler loaded")

        # Anomaly threshold
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH, "rb") as f:
                self._threshold_data = pickle.load(f)
            logger.info(
                "  ✓ Threshold loaded: {:.6f}",
                self._threshold_data.get("threshold", 0.0),
            )

        # Isolation Forest
        if not IFOREST_PATH.exists():
            logger.warning(
                "Isolation Forest not found at {}. Run ml/train_isolation_forest.py first.",
                IFOREST_PATH,
            )
        else:
            with open(IFOREST_PATH, "rb") as f:
                self._iforest = pickle.load(f)
            logger.info("  ✓ Isolation Forest loaded")

        self._loaded = True
        logger.info("MLService: all artefacts loaded.")

    @property
    def is_ready(self) -> bool:
        return self._loaded and (
            self._autoencoder is not None or self._iforest is not None
        )

    # ── Inference ──────────────────────────────────────────────────────────────

    async def predict(self, reading: SensorReading) -> AnomalyResult:
        """
        Run ensemble inference on a single SensorReading.

        Returns an AnomalyResult with anomaly_score, failure_probability,
        failure_type_prediction, and sensor_deltas.
        """
        if not self._loaded:
            self.load()

        raw = np.array(
            [getattr(reading, f) for f in FEATURES], dtype=np.float32
        )

        # Scale
        scaled = self._scale(raw)

        # Update rolling buffer for this machine
        buf = self._get_buffer(reading.machine_id)
        buf.append(scaled)

        lstm_score: Optional[float] = None
        lstm_recon_error: Optional[float] = None
        iforest_score: Optional[float] = None

        # ── LSTM Autoencoder ──────────────────────────────────────────────────
        if self._autoencoder is not None and len(buf) == self.SEQUENCE_LENGTH:
            sequence = np.array(list(buf), dtype=np.float32)[np.newaxis, ...]  # (1, 30, 5)
            reconstructed = self._autoencoder.predict(sequence, verbose=0)
            mse = float(np.mean(np.power(sequence - reconstructed, 2)))
            lstm_recon_error = mse
            threshold = self._threshold_data.get("threshold", 0.0)
            # Normalise: 0 = perfect, 1 = far above threshold
            if threshold > 0:
                lstm_score = min(mse / (threshold * 2.0), 1.0)
            else:
                lstm_score = 0.0

        # ── Isolation Forest ─────────────────────────────────────────────────
        if self._iforest is not None:
            decision = float(self._iforest.decision_function(scaled.reshape(1, -1))[0])
            iforest_score = decision
            # Isolation Forest: negative = anomalous; normalise to [0,1]
            # Typical range [-0.5, 0.5]; flip and clip
            iforest_norm = float(np.clip((-decision + 0.5) / 1.0, 0.0, 1.0))
        else:
            iforest_norm = 0.0

        # ── Ensemble ─────────────────────────────────────────────────────────
        if lstm_score is not None and iforest_norm:
            anomaly_score = 0.6 * lstm_score + 0.4 * iforest_norm
            ml_model_used = "ensemble"
        elif lstm_score is not None:
            anomaly_score = lstm_score
            ml_model_used = "lstm_autoencoder"
        else:
            anomaly_score = iforest_norm
            ml_model_used = "isolation_forest"

        is_anomaly = anomaly_score >= 0.5
        failure_probability = float(np.clip(anomaly_score * 1.2, 0.0, 1.0))

        # ── Sensor Deltas (z-scores vs dataset stats) ─────────────────────────
        sensor_deltas = self._compute_deltas(raw, reading.machine_id, buf)

        # ── Failure Type Heuristic ─────────────────────────────────────────────
        failure_type: Optional[FailureType] = None
        if is_anomaly:
            failure_type = self._infer_failure_type(sensor_deltas)

        return AnomalyResult(
            machine_id=reading.machine_id,
            timestamp=reading.timestamp,
            anomaly_score=round(anomaly_score, 4),
            failure_probability=round(failure_probability, 4),
            is_anomaly=is_anomaly,
            failure_type_prediction=failure_type,
            sensor_deltas=sensor_deltas,
            ml_model_used=ml_model_used,
            reconstruction_error=lstm_recon_error,
            isolation_score=iforest_score,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _scale(self, raw: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.transform(raw.reshape(1, -1))[0].astype(np.float32)
        # Fallback: min-max using approximate AI4I ranges
        mins = np.array([295.0, 305.0, 1168.0, 3.8, 0.0], dtype=np.float32)
        maxs = np.array([304.0, 314.0, 2886.0, 76.6, 253.0], dtype=np.float32)
        return np.clip((raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)

    def _get_buffer(self, machine_id: str) -> deque:
        if machine_id not in self._buffers:
            self._buffers[machine_id] = deque(maxlen=self.SEQUENCE_LENGTH)
        return self._buffers[machine_id]

    def _compute_deltas(
        self, raw: np.ndarray, machine_id: str, buf: deque
    ) -> dict[str, float]:
        """Compute per-sensor z-scores against the rolling buffer mean/std."""
        if len(buf) < 5:
            return {f: 0.0 for f in FEATURES}

        history = np.array(list(buf))
        mean = history.mean(axis=0)
        std  = history.std(axis=0) + 1e-8
        scaled = self._scale(raw)
        z_scores = (scaled - mean) / std
        return {f: round(float(z), 3) for f, z in zip(FEATURES, z_scores)}

    def _infer_failure_type(self, deltas: dict[str, float]) -> Optional[FailureType]:
        """Simple rule-based heuristic to predict most likely failure type."""
        candidates: list[tuple[float, str]] = []
        for feature, direction, z_thresh, ftype in _FAILURE_RULES:
            z = deltas.get(feature, 0.0)
            if direction == "high" and z >= z_thresh:
                candidates.append((z, ftype))
            elif direction == "low" and z <= -z_thresh:
                candidates.append((-z, ftype))

        if not candidates:
            return None
        # Return failure type with highest triggering z-score
        candidates.sort(reverse=True)
        return candidates[0][1]  # type: ignore[return-value]
