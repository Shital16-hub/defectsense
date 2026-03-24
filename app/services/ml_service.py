"""
ML Inference Service — loads trained LSTM Autoencoder + Isolation Forest and runs
ensemble anomaly detection.

Key design decisions vs the previous version:
  - Sequence is built OUTSIDE this service (by AnomalyDetectorAgent from Redis cache)
    so this service is stateless and reusable across workers.
  - All CPU-bound inference runs in a thread-pool executor to avoid blocking the
    async event loop.
  - Each prediction is logged to an MLflow run kept open for the lifetime of the app.
  - High-confidence detection: reconstruction_error > threshold AND iforest == -1.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
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

SEQUENCE_LENGTH = 30

# Rule-based failure-type heuristics (AI4I domain knowledge)
_FAILURE_RULES: list[tuple[str, str, float, str]] = [
    ("tool_wear",           "high", 2.0, "TWF"),
    ("air_temperature",     "high", 2.5, "HDF"),
    ("process_temperature", "high", 2.5, "HDF"),
    ("rotational_speed",    "low",  2.0, "PWF"),
    ("torque",              "high", 2.5, "PWF"),
    ("rotational_speed",    "high", 2.5, "OSF"),
]


class MLService:
    """
    Stateless ML inference service — no internal rolling buffer.
    The caller (AnomalyDetectorAgent) supplies the pre-built sequence from Redis.
    """

    def __init__(self) -> None:
        self._autoencoder   = None
        self._scaler        = None
        self._threshold_data: dict = {}
        self._iforest       = None
        self._loaded        = False
        self._executor      = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # MLflow prediction tracking
        self._mlflow_run_id: Optional[str] = None
        self._pred_step     = 0
        self._mlflow_lock   = threading.Lock()

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all model artefacts from disk. Call once at app startup."""
        if self._loaded:
            return

        logger.info("MLService: loading artefacts from {}", MODELS_DIR)

        if AUTOENCODER_PATH.exists():
            import tensorflow as tf  # deferred — slow import
            self._autoencoder = tf.keras.models.load_model(str(AUTOENCODER_PATH))
            logger.info("  ✓ LSTM Autoencoder loaded")
        else:
            logger.warning("  ✗ Autoencoder not found — run ml/train_autoencoder.py")

        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("  ✓ Sensor scaler loaded")

        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH, "rb") as f:
                self._threshold_data = pickle.load(f)
            logger.info(
                "  ✓ Threshold loaded: {:.6f}",
                self._threshold_data.get("threshold", 0.0),
            )

        if IFOREST_PATH.exists():
            with open(IFOREST_PATH, "rb") as f:
                iforest_data = pickle.load(f)
            # pkl was saved as a dict: {"model": ..., "scaler": ..., ...}
            if isinstance(iforest_data, dict):
                self._iforest = iforest_data["model"]
            else:
                self._iforest = iforest_data
            logger.info("  ✓ Isolation Forest loaded")
        else:
            logger.warning("  ✗ Isolation Forest not found — run ml/train_isolation_forest.py")

        self._loaded = True
        self._init_mlflow()
        logger.info("MLService ready.")

    @property
    def is_ready(self) -> bool:
        return self._loaded and (
            self._autoencoder is not None or self._iforest is not None
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def predict_anomaly(
        self,
        reading: SensorReading,
        sequence: Optional[list[SensorReading]] = None,
    ) -> AnomalyResult:
        """
        Run ensemble anomaly detection.

        Args:
            reading:  The current SensorReading to evaluate.
            sequence: Last N SensorReadings for this machine (from Redis cache).
                      If len(sequence) < SEQUENCE_LENGTH, LSTM is skipped and
                      only Isolation Forest runs.

        Returns:
            AnomalyResult with scores, confidence flags, failure type, and deltas.
        """
        if not self._loaded:
            self.load()

        loop = asyncio.get_event_loop()
        raw    = self._to_raw(reading)
        scaled = self._scale(raw)

        lstm_recon_error: Optional[float] = None
        lstm_above_threshold              = False
        iforest_prediction: Optional[int] = None
        iforest_score: Optional[float]    = None
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

        # ── Isolation Forest (CPU-bound → thread pool) ───────────────────────
        if self._iforest is not None:
            iforest_prediction, iforest_score = await loop.run_in_executor(
                self._executor, self._run_iforest, scaled
            )
            ml_model_used = (
                "ensemble" if ml_model_used == "lstm_autoencoder" else "isolation_forest"
            )

        # ── Confidence classification ─────────────────────────────────────────
        #   HIGH:   both models flag anomaly
        #   MEDIUM: only one model flags anomaly
        #   NONE:   neither
        high_confidence = lstm_above_threshold and iforest_prediction == -1
        medium_confidence = lstm_above_threshold or iforest_prediction == -1
        is_anomaly = high_confidence or medium_confidence

        anomaly_score = self._compute_anomaly_score(
            lstm_recon_error, iforest_score, high_confidence
        )
        failure_probability = float(np.clip(anomaly_score * 1.15, 0.0, 1.0))

        # ── Sensor deltas ─────────────────────────────────────────────────────
        sensor_deltas = self._compute_deltas(scaled, sequence)

        # ── Failure type ──────────────────────────────────────────────────────
        failure_type: Optional[FailureType] = None
        if is_anomaly:
            failure_type = self._infer_failure_type(sensor_deltas)

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
            isolation_score=iforest_score,
        )

        # Log to MLflow in background thread (run_in_executor returns a Future, not a coroutine)
        loop.run_in_executor(self._executor, self._log_to_mlflow, result)

        return result

    # ── Thread-pool workers (synchronous, called via executor) ─────────────────

    def _run_lstm(self, sequence: np.ndarray) -> float:
        """Run LSTM autoencoder and return reconstruction MSE."""
        reconstructed = self._autoencoder.predict(sequence, verbose=0)
        mse = float(np.mean(np.power(sequence - reconstructed, 2)))
        return mse

    def _run_iforest(self, scaled: np.ndarray) -> tuple[int, float]:
        """Run Isolation Forest; return (prediction, decision_score)."""
        pred  = int(self._iforest.predict(scaled.reshape(1, -1))[0])    # 1 or -1
        score = float(self._iforest.decision_function(scaled.reshape(1, -1))[0])
        return pred, score

    # ── MLflow prediction tracking ─────────────────────────────────────────────

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri("./mlruns")
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
        # Fallback: approximate AI4I ranges
        mins = np.array([295.0, 305.0, 1168.0, 3.8, 0.0],   dtype=np.float32)
        maxs = np.array([304.0, 314.0, 2886.0, 76.6, 253.0], dtype=np.float32)
        return np.clip((raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)

    def _build_sequence_array(self, readings: list[SensorReading]) -> np.ndarray:
        """Convert list of SensorReadings → (1, SEQUENCE_LENGTH, N_FEATURES) array."""
        rows = [self._scale(self._to_raw(r)) for r in readings]
        return np.array(rows, dtype=np.float32)[np.newaxis, ...]

    def _compute_anomaly_score(
        self,
        recon_error: Optional[float],
        iforest_score: Optional[float],
        high_confidence: bool,
    ) -> float:
        threshold = self._threshold_data.get("threshold", 1.0) or 1.0

        lstm_norm = 0.0
        if recon_error is not None:
            lstm_norm = float(np.clip(recon_error / (threshold * 2.0), 0.0, 1.0))

        iforest_norm = 0.0
        if iforest_score is not None:
            # IForest decision_function: negative = anomalous, range ~ [-0.5, 0.5]
            iforest_norm = float(np.clip((-iforest_score + 0.5), 0.0, 1.0))

        if lstm_norm and iforest_norm:
            score = 0.6 * lstm_norm + 0.4 * iforest_norm
        elif lstm_norm:
            score = lstm_norm
        elif iforest_norm:
            score = iforest_norm
        else:
            score = 0.0

        # Boost score slightly for confirmed high-confidence detections
        if high_confidence:
            score = min(score * 1.1, 1.0)

        return round(score, 4)

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

    def _infer_failure_type(self, deltas: dict[str, float]) -> Optional[FailureType]:
        candidates: list[tuple[float, str]] = []
        for feature, direction, z_thresh, ftype in _FAILURE_RULES:
            z = deltas.get(feature, 0.0)
            if direction == "high" and z >= z_thresh:
                candidates.append((z, ftype))
            elif direction == "low" and z <= -z_thresh:
                candidates.append((-z, ftype))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]  # type: ignore[return-value]
