"""
Redis Service — async Redis client wrapping sensor reading cache and pub/sub.

Responsibilities:
  - publish_sensor(reading)         → pub/sub channel "sensor:feed"
  - store_reading(reading)          → per-machine list for sequence building
  - get_recent_readings(machine_id) → last N SensorReadings (for LSTM sequence)
  - cache_anomaly(result)           → per-machine anomaly list (dashboard)
  - get_recent_anomalies(machine_id)→ last N AnomalyResults

Redis key schema:
  sensor:{machine_id}:readings   — LPUSH list, trimmed to MAX_READINGS
  sensor:{machine_id}:anomalies  — LPUSH list, trimmed to MAX_ANOMALIES
  sensor:feed                    — pub/sub channel
  defectsense:anomaly_results    — pub/sub channel (for WebSocket broadcast)
"""
from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading

CHANNEL_SENSOR_FEED     = "sensor:feed"
CHANNEL_ANOMALY_RESULTS = "defectsense:anomaly_results"

MAX_READINGS  = 50   # keep last 50 per machine (need 30 for LSTM)
MAX_ANOMALIES = 100  # keep last 100 anomalies per machine


class RedisService:
    """
    Async Redis client. Call `await init()` at app startup and `await close()`
    at shutdown. Handles connection failures gracefully (returns empty results
    rather than crashing the pipeline).
    """

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url    = url
        self._client: Optional[aioredis.Redis] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def init(self) -> None:
        self._client = aioredis.from_url(self._url, decode_responses=True)
        try:
            await self._client.ping()
            logger.info("RedisService: connected to {}", self._url)
        except Exception as exc:
            logger.warning("RedisService: connection failed — {} (will retry on use)", exc)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("RedisService: connection closed")

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ── Pub/Sub ────────────────────────────────────────────────────────────────

    async def publish_sensor(self, reading: SensorReading) -> None:
        """Publish a SensorReading JSON to the 'sensor:feed' channel."""
        if not self._client:
            return
        try:
            await self._client.publish(CHANNEL_SENSOR_FEED, reading.model_dump_json())
        except Exception as exc:
            logger.warning("RedisService.publish_sensor failed: {}", exc)

    async def publish_anomaly(self, result: AnomalyResult) -> None:
        """Publish an AnomalyResult to the anomaly_results channel."""
        if not self._client:
            return
        try:
            await self._client.publish(CHANNEL_ANOMALY_RESULTS, result.model_dump_json())
        except Exception as exc:
            logger.warning("RedisService.publish_anomaly failed: {}", exc)

    # ── Reading cache ──────────────────────────────────────────────────────────

    async def store_reading(self, reading: SensorReading) -> None:
        """
        Push SensorReading JSON into the per-machine readings list.
        Trims the list to MAX_READINGS so memory stays bounded.
        """
        if not self._client:
            return
        try:
            key = f"sensor:{reading.machine_id}:readings"
            payload = reading.model_dump_json()
            pipe = self._client.pipeline()
            pipe.lpush(key, payload)
            pipe.ltrim(key, 0, MAX_READINGS - 1)
            await pipe.execute()
        except Exception as exc:
            logger.warning("RedisService.store_reading failed: {}", exc)

    async def get_recent_readings(
        self, machine_id: str, n: int = 30
    ) -> list[SensorReading]:
        """
        Return the last `n` SensorReadings for a machine (newest first from LPUSH,
        reversed here to return oldest→newest for LSTM sequence order).
        Returns empty list if Redis unavailable or insufficient history.
        """
        if not self._client:
            return []
        try:
            key  = f"sensor:{machine_id}:readings"
            raw  = await self._client.lrange(key, 0, n - 1)
            # lrange returns newest→oldest (LPUSH), reverse for time-order
            readings = []
            for item in reversed(raw):
                try:
                    readings.append(SensorReading(**json.loads(item)))
                except Exception:
                    continue
            return readings
        except Exception as exc:
            logger.warning("RedisService.get_recent_readings failed: {}", exc)
            return []

    # ── Anomaly cache ──────────────────────────────────────────────────────────

    async def cache_anomaly(self, result: AnomalyResult) -> None:
        """Store anomaly result in per-machine list for dashboard queries."""
        if not self._client:
            return
        try:
            key = f"sensor:{result.machine_id}:anomalies"
            pipe = self._client.pipeline()
            pipe.lpush(key, result.model_dump_json())
            pipe.ltrim(key, 0, MAX_ANOMALIES - 1)
            await pipe.execute()
        except Exception as exc:
            logger.warning("RedisService.cache_anomaly failed: {}", exc)

    async def get_recent_anomalies(
        self, machine_id: str, n: int = 10
    ) -> list[AnomalyResult]:
        """Return the last `n` anomalies for a machine (newest first)."""
        if not self._client:
            return []
        try:
            key = f"sensor:{machine_id}:anomalies"
            raw = await self._client.lrange(key, 0, n - 1)
            results = []
            for item in raw:
                try:
                    results.append(AnomalyResult(**json.loads(item)))
                except Exception:
                    continue
            return results
        except Exception as exc:
            logger.warning("RedisService.get_recent_anomalies failed: {}", exc)
            return []

    # ── History (for API endpoint) ─────────────────────────────────────────────

    async def get_history(self, machine_id: str, n: int = 50) -> list[SensorReading]:
        """Alias for get_recent_readings with larger default — used by history endpoint."""
        return await self.get_recent_readings(machine_id, n=n)
