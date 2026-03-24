"""
Anomaly Detector Agent — orchestrates sequence retrieval → ML inference → storage.

Flow per SensorReading:
  1. Store reading in Redis cache (always)
  2. Fetch last 30 readings for this machine from Redis → build LSTM sequence
  3. Call MLService.predict_anomaly(reading, sequence)
  4. If anomaly detected:
       - Log AnomalyResult to MongoDB
       - Cache in Redis anomaly list
       - Publish to Redis pub/sub (WebSocket broadcast)
  5. Return AnomalyResult to caller (API layer / LangGraph)

Handles gracefully:
  - Model not loaded → returns placeholder with is_anomaly=False
  - Redis unavailable → runs IForest-only (no LSTM sequence)
  - Insufficient sequence length (< 30) → runs IForest-only
  - MongoDB unavailable → logs warning, continues
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from loguru import logger

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading

if TYPE_CHECKING:
    from app.services.ml_service import MLService
    from app.services.redis_service import RedisService

SEQUENCE_LENGTH = 30
MONGODB_COLLECTION = "anomalies"


class AnomalyDetectorAgent:
    """
    Stateless agent — all state lives in Redis and MongoDB.
    Inject services at construction; call `await run(reading)`.
    """

    def __init__(
        self,
        ml_service: "MLService",
        redis_service: "RedisService",
        mongo_db=None,   # motor AsyncIOMotorDatabase or None
    ) -> None:
        self._ml      = ml_service
        self._redis   = redis_service
        self._mongo   = mongo_db

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self, reading: SensorReading) -> AnomalyResult:
        """
        Process one SensorReading end-to-end.

        Returns:
            AnomalyResult — always returned even if downstream steps fail.
        """
        if not self._ml.is_ready:
            logger.warning(
                "AnomalyDetector: ML models not loaded — returning safe default for {}",
                reading.machine_id,
            )
            return self._safe_default(reading)

        # Step 1: store reading in Redis (non-blocking — fire-and-forget errors)
        await self._redis.store_reading(reading)

        # Step 2: fetch sequence for LSTM
        sequence = await self._get_sequence(reading.machine_id)
        if len(sequence) < SEQUENCE_LENGTH:
            logger.debug(
                "AnomalyDetector: only {}/{} readings for {} — LSTM skipped, IForest only",
                len(sequence),
                SEQUENCE_LENGTH,
                reading.machine_id,
            )

        # Step 3: run inference
        result = await self._ml.predict_anomaly(reading, sequence or None)

        # Step 4: post-process anomaly
        if result.is_anomaly:
            logger.warning(
                "ANOMALY | machine={} score={:.3f} prob={:.3f} type={} model={}",
                result.machine_id,
                result.anomaly_score,
                result.failure_probability,
                result.failure_type_prediction,
                result.ml_model_used,
            )
            await self._store_anomaly(result)
        else:
            logger.debug(
                "Normal  | machine={} score={:.3f}",
                result.machine_id,
                result.anomaly_score,
            )

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _get_sequence(self, machine_id: str) -> list[SensorReading]:
        try:
            return await self._redis.get_recent_readings(machine_id, n=SEQUENCE_LENGTH)
        except Exception as exc:
            logger.warning("AnomalyDetector: Redis unavailable for sequence fetch — {}", exc)
            return []

    async def _store_anomaly(self, result: AnomalyResult) -> None:
        """Persist anomaly to Redis cache + MongoDB + pub/sub."""
        # Redis cache (for dashboard)
        await self._redis.cache_anomaly(result)

        # Redis pub/sub (WebSocket broadcast)
        await self._redis.publish_anomaly(result)

        # MongoDB (durable audit log)
        if self._mongo is not None:
            await self._log_to_mongo(result)

    async def _log_to_mongo(self, result: AnomalyResult) -> None:
        try:
            doc = result.model_dump(mode="json")
            doc["logged_at"] = datetime.now(tz=timezone.utc).isoformat()
            await self._mongo[MONGODB_COLLECTION].insert_one(doc)
            logger.debug("AnomalyDetector: logged anomaly to MongoDB for {}", result.machine_id)
        except Exception as exc:
            logger.warning("AnomalyDetector: MongoDB write failed (non-fatal) — {}", exc)

    @staticmethod
    def _safe_default(reading: SensorReading) -> AnomalyResult:
        return AnomalyResult(
            machine_id=reading.machine_id,
            timestamp=reading.timestamp,
            anomaly_score=0.0,
            failure_probability=0.0,
            is_anomaly=False,
            sensor_deltas={},
            ml_model_used="none",
        )
