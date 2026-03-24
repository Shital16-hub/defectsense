"""
MongoDB Service — async motor client for durable storage.

Collections:
  - sensor_readings  : raw ingested readings (optional, high-volume)
  - anomalies        : every detected anomaly (AnomalyResult)
  - alerts           : approved/pending MaintenanceAlerts
  - sessions         : LangGraph agent session metadata

Usage:
    svc = MongoDBService(url, db_name)
    await svc.init()
    await svc.save_anomaly(result)
    stats = await svc.get_anomaly_stats()
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from app.models.alert import MaintenanceAlert
from app.models.anomaly import AnomalyResult

COLL_READINGS  = "sensor_readings"
COLL_ANOMALIES = "anomalies"
COLL_ALERTS    = "alerts"
COLL_SESSIONS  = "sessions"


class MongoDBService:
    """
    Async MongoDB service via motor.
    Degrades gracefully if unavailable — methods return None / empty rather than raising.
    """

    def __init__(self, url: str, db_name: str = "defectsense") -> None:
        self._url     = url
        self._db_name = db_name
        self._client  = None
        self._db      = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def init(self) -> None:
        try:
            import motor.motor_asyncio as motor
            self._client = motor.AsyncIOMotorClient(
                self._url, serverSelectionTimeoutMS=3000
            )
            self._db = self._client[self._db_name]
            await self._client.admin.command("ping")
            await self._ensure_indexes()
            logger.info("MongoDBService: connected to '{}' at {}", self._db_name, self._url)
        except Exception as exc:
            logger.warning("MongoDBService: connection failed — {} (will degrade gracefully)", exc)
            self._db = None

    async def close(self) -> None:
        if self._client:
            self._client.close()

    @property
    def is_connected(self) -> bool:
        return self._db is not None

    async def _ensure_indexes(self) -> None:
        """Create indexes for common query patterns."""
        try:
            await self._db[COLL_ANOMALIES].create_index(
                [("machine_id", 1), ("timestamp", -1)]
            )
            await self._db[COLL_ALERTS].create_index(
                [("machine_id", 1), ("created_at", -1)]
            )
            await self._db[COLL_ALERTS].create_index([("approved", 1)])
        except Exception as exc:
            logger.warning("MongoDBService: index creation failed — {}", exc)

    # ── Anomalies ──────────────────────────────────────────────────────────────

    async def save_anomaly(self, result: AnomalyResult) -> Optional[str]:
        """Persist an AnomalyResult. Returns inserted_id or None on failure."""
        if not self._db:
            return None
        try:
            doc = result.model_dump(mode="json")
            doc["logged_at"] = datetime.now(tz=timezone.utc).isoformat()
            res = await self._db[COLL_ANOMALIES].insert_one(doc)
            return str(res.inserted_id)
        except Exception as exc:
            logger.warning("MongoDBService.save_anomaly failed: {}", exc)
            return None

    async def get_anomalies(
        self,
        machine_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent anomalies, optionally filtered by machine_id."""
        if not self._db:
            return []
        try:
            query  = {"machine_id": machine_id} if machine_id else {}
            cursor = self._db[COLL_ANOMALIES].find(
                query, {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as exc:
            logger.warning("MongoDBService.get_anomalies failed: {}", exc)
            return []

    async def get_anomaly_stats(self) -> dict:
        """
        Aggregate stats for the dashboard:
          - total_anomalies
          - by_failure_type: {TWF: N, HDF: N, ...}
          - by_machine: {M001: N, ...} (top 10)
          - recent_24h: count in last 24 hours
        """
        if not self._db:
            return {}
        try:
            from datetime import timedelta

            now     = datetime.now(tz=timezone.utc)
            cutoff  = (now - timedelta(hours=24)).isoformat()

            total   = await self._db[COLL_ANOMALIES].count_documents({})
            recent  = await self._db[COLL_ANOMALIES].count_documents(
                {"timestamp": {"$gte": cutoff}}
            )

            # By failure type
            by_type_cursor = self._db[COLL_ANOMALIES].aggregate([
                {"$match": {"failure_type_prediction": {"$ne": None}}},
                {"$group": {"_id": "$failure_type_prediction", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ])
            by_type = {d["_id"]: d["count"] async for d in by_type_cursor}

            # By machine (top 10)
            by_machine_cursor = self._db[COLL_ANOMALIES].aggregate([
                {"$group": {"_id": "$machine_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10},
            ])
            by_machine = {d["_id"]: d["count"] async for d in by_machine_cursor}

            return {
                "total_anomalies": total,
                "recent_24h":      recent,
                "by_failure_type": by_type,
                "by_machine":      by_machine,
            }
        except Exception as exc:
            logger.warning("MongoDBService.get_anomaly_stats failed: {}", exc)
            return {}

    # ── Alerts ─────────────────────────────────────────────────────────────────

    async def save_alert(self, alert: MaintenanceAlert) -> Optional[str]:
        """Persist a MaintenanceAlert. Returns inserted_id or None."""
        if not self._db:
            return None
        try:
            doc = alert.model_dump(mode="json")
            doc["created_at"] = datetime.now(tz=timezone.utc).isoformat()
            res = await self._db[COLL_ALERTS].insert_one(doc)
            return str(res.inserted_id)
        except Exception as exc:
            logger.warning("MongoDBService.save_alert failed: {}", exc)
            return None

    async def get_alerts(
        self,
        machine_id: Optional[str] = None,
        pending_only: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        """Return alerts, optionally filtered by machine or pending status."""
        if not self._db:
            return []
        try:
            query: dict = {}
            if machine_id:
                query["machine_id"] = machine_id
            if pending_only:
                query["approved"] = None
            cursor = self._db[COLL_ALERTS].find(
                query, {"_id": 0}
            ).sort("created_at", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as exc:
            logger.warning("MongoDBService.get_alerts failed: {}", exc)
            return []

    async def update_alert_approval(
        self,
        alert_id: str,
        approved: bool,
        approved_by: str = "human",
        rejection_reason: Optional[str] = None,
    ) -> bool:
        """Update approval status of an alert. Returns True on success."""
        if not self._db:
            return False
        try:
            update: dict = {
                "$set": {
                    "approved":       approved,
                    "approved_by":    approved_by,
                    "approved_at":    datetime.now(tz=timezone.utc).isoformat(),
                }
            }
            if rejection_reason:
                update["$set"]["rejection_reason"] = rejection_reason
            res = await self._db[COLL_ALERTS].update_one(
                {"alert_id": alert_id}, update
            )
            return res.modified_count > 0
        except Exception as exc:
            logger.warning("MongoDBService.update_alert_approval failed: {}", exc)
            return False

    # ── Sessions ───────────────────────────────────────────────────────────────

    async def save_session(self, session_id: str, metadata: dict) -> None:
        """Save or update a LangGraph agent session record."""
        if not self._db:
            return
        try:
            await self._db[COLL_SESSIONS].update_one(
                {"session_id": session_id},
                {"$set": {**metadata, "updated_at": datetime.now(tz=timezone.utc).isoformat()}},
                upsert=True,
            )
        except Exception as exc:
            logger.warning("MongoDBService.save_session failed: {}", exc)
