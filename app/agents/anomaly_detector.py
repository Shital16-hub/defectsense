"""
Anomaly Detector Agent — wraps MLService inside an AIOS-compatible agent interface.

AIOS (agiresearch/AIOS) schedules agents as OS-like processes with resource management.
This agent:
  1. Receives a SensorReading (via Redis pub/sub or direct call)
  2. Runs MLService ensemble inference
  3. Publishes AnomalyResult back to Redis channel `defectsense:anomaly_results`
  4. Returns the result to the LangGraph orchestrator (AgentState)

Designed to run under AIOS scheduler — implements the expected `AgentProcess` interface
while remaining usable standalone (direct `run()` call) for testing.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from loguru import logger

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading
from app.services.ml_service import MLService

# ── AIOS shim ─────────────────────────────────────────────────────────────────
# AIOS is installed from source; guard import so the agent still works without it.
try:
    from aios.agent.base import BaseAgent  # type: ignore[import]
    _AIOS_AVAILABLE = True
except ImportError:
    _AIOS_AVAILABLE = False
    logger.warning(
        "AIOS not installed — AnomalyDetectorAgent will run in standalone mode. "
        "Install via: pip install git+https://github.com/agiresearch/AIOS"
    )

    class BaseAgent:  # type: ignore[no-redef]
        """Minimal shim so the class definition works without AIOS."""
        def __init__(self, agent_name: str, task_input: str, *args: Any, **kwargs: Any) -> None:
            self.agent_name = agent_name
            self.task_input = task_input

        def run(self) -> Any:
            raise NotImplementedError


# ── Redis channel names ────────────────────────────────────────────────────────
CHANNEL_SENSOR_READINGS = "defectsense:sensor_readings"
CHANNEL_ANOMALY_RESULTS = "defectsense:anomaly_results"


class AnomalyDetectorAgent(BaseAgent):
    """
    AIOS-compatible agent that performs ensemble ML anomaly detection.

    Can be used in two modes:
    1. **Standalone** — call `await detect(reading)` directly.
    2. **AIOS scheduled** — AIOS calls `run()` which processes `task_input`
       (a JSON-serialised SensorReading).
    """

    _ml_service: Optional[MLService] = None  # shared across instances

    def __init__(
        self,
        agent_name: str = "anomaly_detector",
        task_input: str = "",
        redis_client=None,
    ) -> None:
        super().__init__(agent_name=agent_name, task_input=task_input)
        self._redis = redis_client
        # Lazy-init shared ML service
        if AnomalyDetectorAgent._ml_service is None:
            AnomalyDetectorAgent._ml_service = MLService()
            AnomalyDetectorAgent._ml_service.load()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def detect(self, reading: SensorReading) -> AnomalyResult:
        """
        Run inference on a SensorReading and optionally publish result to Redis.

        Args:
            reading: Validated SensorReading from sensor ingestion service.

        Returns:
            AnomalyResult with ensemble score, failure probability, type prediction.
        """
        logger.debug(
            "AnomalyDetector: processing machine={} ts={}", reading.machine_id, reading.timestamp
        )

        result = await self._ml_service.predict(reading)

        if result.is_anomaly:
            logger.warning(
                "ANOMALY DETECTED — machine={} score={:.3f} prob={:.3f} type={}",
                result.machine_id,
                result.anomaly_score,
                result.failure_probability,
                result.failure_type_prediction,
            )
        else:
            logger.debug(
                "Normal — machine={} score={:.3f}",
                result.machine_id,
                result.anomaly_score,
            )

        # Publish to Redis so other agents / WebSocket subscribers can consume
        if self._redis is not None:
            await self._publish_result(result)

        return result

    # ── AIOS interface ─────────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """
        AIOS entry-point. Deserialises task_input (JSON SensorReading),
        runs detect(), returns serialised AnomalyResult.
        """
        try:
            data = json.loads(self.task_input)
            reading = SensorReading(**data)
        except Exception as exc:
            logger.error("AnomalyDetectorAgent.run: failed to parse task_input: {}", exc)
            return {"error": str(exc)}

        # AIOS is synchronous; bridge to async with a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running inside an existing loop (e.g. FastAPI) — create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.detect(reading))
                    result = future.result()
            else:
                result = loop.run_until_complete(self.detect(reading))
        except RuntimeError:
            result = asyncio.run(self.detect(reading))

        return result.model_dump(mode="json")

    # ── Redis helpers ──────────────────────────────────────────────────────────

    async def _publish_result(self, result: AnomalyResult) -> None:
        try:
            payload = result.model_dump_json()
            await self._redis.publish(CHANNEL_ANOMALY_RESULTS, payload)
            logger.debug("Published AnomalyResult for {} to Redis", result.machine_id)
        except Exception as exc:
            logger.error("Failed to publish anomaly result to Redis: {}", exc)


# ── Convenience factory ────────────────────────────────────────────────────────

def create_anomaly_detector(redis_client=None) -> AnomalyDetectorAgent:
    """Create a ready-to-use AnomalyDetectorAgent (ML models loaded)."""
    return AnomalyDetectorAgent(redis_client=redis_client)
