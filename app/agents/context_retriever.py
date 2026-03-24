"""
Context Retriever Agent — RAG layer for the DefectSense pipeline.

Given an AnomalyResult, this agent:
  1. Builds a semantic query from the anomaly (failure type + sensor deviations)
  2. Searches Qdrant for the top 3 most similar past incidents
  3. Computes a sensor trend summary from the recent Redis reading history
  4. Returns similar_incidents (list[MaintenanceLog]) and sensor_context (str)
     for injection into the Root Cause Reasoner prompt

Standalone test:
    python -c "
    import asyncio
    from app.agents.context_retriever import ContextRetrieverAgent, _mock_anomaly
    asyncio.run(ContextRetrieverAgent.standalone_test())
    "
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from loguru import logger

from app.models.anomaly import AnomalyResult
from app.models.maintenance import MaintenanceLog

if TYPE_CHECKING:
    from app.services.qdrant_service import QdrantService
    from app.services.redis_service import RedisService

# Sensor display names for human-readable trend summaries
SENSOR_LABELS = {
    "air_temperature":     "air temperature",
    "process_temperature": "process temperature",
    "rotational_speed":    "rotational speed",
    "torque":              "torque",
    "tool_wear":           "tool wear",
}


class ContextRetrieverAgent:
    """
    Stateless RAG agent. Inject QdrantService and RedisService at construction.
    Call `await retrieve(anomaly_result)` to get context for root-cause reasoning.
    """

    def __init__(
        self,
        qdrant: "QdrantService",
        redis: Optional["RedisService"] = None,
    ) -> None:
        self._qdrant = qdrant
        self._redis  = redis

    # ── Public API ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        anomaly: AnomalyResult,
        n_history: int = 10,
    ) -> tuple[list[MaintenanceLog], str]:
        """
        Retrieve context for a detected anomaly.

        Args:
            anomaly:   The AnomalyResult from the anomaly detector.
            n_history: Number of past readings to use for trend summary.

        Returns:
            (similar_incidents, sensor_context)
            - similar_incidents: up to 3 MaintenanceLogs from Qdrant
            - sensor_context:    plain-English sensor trend summary string
        """
        if not self._qdrant.is_ready:
            logger.warning("ContextRetriever: Qdrant not ready — returning empty context")
            return [], "No historical context available."

        # Build semantic query from anomaly
        query = self._build_query(anomaly)
        logger.debug("ContextRetriever: query = '{}'", query)

        # Qdrant similarity search
        similar = await self._qdrant.search_similar_incidents(
            query=query,
            failure_type=anomaly.failure_type_prediction,
            limit=3,
        )

        # If typed search returned nothing, retry without type filter
        if not similar:
            logger.debug(
                "ContextRetriever: no results for type={}, retrying without filter",
                anomaly.failure_type_prediction,
            )
            similar = await self._qdrant.search_similar_incidents(
                query=query, failure_type=None, limit=3
            )

        logger.info(
            "ContextRetriever: found {} similar incidents for machine={}",
            len(similar),
            anomaly.machine_id,
        )

        # Sensor trend summary from Redis history
        sensor_context = await self._build_sensor_context(anomaly, n_history)

        return similar, sensor_context

    # ── Query builder ──────────────────────────────────────────────────────────

    def _build_query(self, anomaly: AnomalyResult) -> str:
        """
        Build a natural-language query that captures the anomaly signature.
        Example:
          "Machine M0042 showing HDF failure. High process_temperature deviation (+3.8σ),
           high air_temperature deviation (+2.1σ). Anomaly score: 0.87."
        """
        parts = [f"Machine {anomaly.machine_id}"]

        if anomaly.failure_type_prediction and anomaly.failure_type_prediction != "NONE":
            parts.append(f"showing {anomaly.failure_type_prediction} failure")
        else:
            parts.append("showing anomalous sensor readings")

        # Add top deviating sensors (|z| > 1.0)
        significant = sorted(
            [(abs(v), k, v) for k, v in anomaly.sensor_deltas.items() if abs(v) > 1.0],
            reverse=True,
        )[:3]
        for _, sensor, z in significant:
            direction = "high" if z > 0 else "low"
            label     = SENSOR_LABELS.get(sensor, sensor)
            parts.append(f"{direction} {label} deviation ({z:+.1f} std)")

        parts.append(f"Anomaly score: {anomaly.anomaly_score:.2f}")
        return ". ".join(parts) + "."

    # ── Sensor trend summary ───────────────────────────────────────────────────

    async def _build_sensor_context(
        self,
        anomaly: AnomalyResult,
        n: int,
    ) -> str:
        """
        Pull recent readings from Redis and summarise sensor trends.
        Falls back to delta-based summary if Redis is unavailable.
        """
        readings = []
        if self._redis is not None:
            try:
                readings = await self._redis.get_recent_readings(anomaly.machine_id, n=n)
            except Exception as exc:
                logger.warning("ContextRetriever: Redis unavailable for trend — {}", exc)

        if len(readings) >= 3:
            return self._summarise_from_readings(anomaly, readings)
        else:
            return self._summarise_from_deltas(anomaly)

    def _summarise_from_readings(self, anomaly: AnomalyResult, readings) -> str:
        """Compute % change over the reading window for each sensor."""
        import numpy as np

        lines = [f"Sensor trends over last {len(readings)} readings for {anomaly.machine_id}:"]

        sensors = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
        for sensor in sensors:
            try:
                vals = [getattr(r, sensor) for r in readings]
                arr  = [v for v in vals if v is not None]
                if len(arr) < 2:
                    continue
                first, last = arr[0], arr[-1]
                pct = ((last - first) / (abs(first) + 1e-8)) * 100
                mean_val = float(np.mean(arr))
                label = SENSOR_LABELS.get(sensor, sensor)

                if abs(pct) >= 2.0:
                    direction = "rose" if pct > 0 else "fell"
                    lines.append(
                        f"  - {label.capitalize()} {direction} {abs(pct):.1f}% "
                        f"(mean: {mean_val:.1f}, latest: {last:.1f})"
                    )
                else:
                    lines.append(f"  - {label.capitalize()} stable (mean: {mean_val:.1f})")
            except Exception:
                continue

        return "\n".join(lines)

    def _summarise_from_deltas(self, anomaly: AnomalyResult) -> str:
        """Build summary from z-score deltas when no reading history available."""
        if not anomaly.sensor_deltas:
            return f"No sensor history available for {anomaly.machine_id}."

        lines = [f"Current sensor deviations for {anomaly.machine_id} (z-scores vs recent baseline):"]
        for sensor, z in sorted(anomaly.sensor_deltas.items(), key=lambda x: -abs(x[1])):
            label = SENSOR_LABELS.get(sensor, sensor)
            if abs(z) >= 0.5:
                flag = " [HIGH]" if abs(z) >= 2.0 else ""
                lines.append(f"  - {label.capitalize()}: {z:+.2f} std{flag}")
        return "\n".join(lines)

    # ── Standalone test ────────────────────────────────────────────────────────

    @classmethod
    async def standalone_test(cls) -> None:
        """Quick end-to-end test without starting the full FastAPI app."""
        import os
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(Path(__file__).parent.parent.parent / ".env")

        from app.services.qdrant_service import QdrantService
        from datetime import datetime, timezone

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key    = os.getenv("QDRANT_API_KEY") or None

        print("=" * 55)
        print("  ContextRetrieverAgent — Standalone Test")
        print("=" * 55)

        qdrant = QdrantService(url=qdrant_url, api_key=api_key)
        await qdrant.init()

        count = await qdrant.collection_count("maintenance_logs")
        print(f"\nMaintenance logs in Qdrant: {count}")
        if count == 0:
            print("WARNING: No logs indexed. Run: python data/index_maintenance_logs.py")

        agent = cls(qdrant=qdrant, redis=None)

        # Mock anomaly — HDF (Heat Dissipation Failure)
        mock_anomaly = AnomalyResult(
            machine_id="M0042",
            timestamp=datetime.now(tz=timezone.utc),
            anomaly_score=0.87,
            failure_probability=0.74,
            is_anomaly=True,
            failure_type_prediction="HDF",
            sensor_deltas={
                "air_temperature":     2.1,
                "process_temperature": 3.8,
                "rotational_speed":   -0.4,
                "torque":              1.2,
                "tool_wear":           0.9,
            },
            ml_model_used="ensemble",
        )

        print(f"\nQuery anomaly: machine={mock_anomaly.machine_id} type={mock_anomaly.failure_type_prediction}")
        print(f"Built query: {agent._build_query(mock_anomaly)}\n")

        similar, sensor_ctx = await agent.retrieve(mock_anomaly)

        print(f"Similar incidents found: {len(similar)}")
        for i, log in enumerate(similar, 1):
            print(f"\n  [{i}] machine={log.machine_id} type={log.failure_type}")
            print(f"       symptoms   : {log.symptoms[:80]}")
            print(f"       root_cause : {log.root_cause[:80]}")
            print(f"       action     : {log.action_taken[:80]}")

        print(f"\nSensor context:\n{sensor_ctx}")
        print("\n" + "=" * 55)
        print("  Test complete.")
        print("=" * 55)
