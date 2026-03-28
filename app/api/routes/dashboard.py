"""
Dashboard API — Session 6.

Endpoints:
  GET /api/dashboard/stats    — aggregate stats across all machines
  GET /api/dashboard/machines — per-machine health summary
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Request
from loguru import logger

router = APIRouter()


def _get_mongo(request: Request):
    return getattr(request.app.state, "mongo_db", None)

def _get_redis(request: Request):
    return getattr(request.app.state, "redis", None)


@router.get("/stats")
async def dashboard_stats(request: Request):
    """
    Real-time aggregated stats:
      - total_machines_monitored
      - anomalies_last_24h
      - alerts_pending_approval
      - avg_resolution_time_minutes
      - failure_type_distribution
      - alerts_by_severity
    """
    db    = _get_mongo(request)
    redis = _get_redis(request)

    now      = datetime.now(tz=timezone.utc)
    since_24 = now - timedelta(hours=24)

    stats: dict = {
        "total_machines_monitored":   0,
        "anomalies_last_24h":         0,
        "alerts_pending_approval":    0,
        "avg_resolution_time_minutes": None,
        "failure_type_distribution":  {},
        "alerts_by_severity":         {},
        "drift_detected":             None,
        "last_drift_check":           None,
        "as_of":                      now.isoformat(),
    }

    # ── Machine count from Redis keys ──────────────────────────────────────────
    if redis and redis.is_connected:
        try:
            keys = await redis._client.keys("sensor:*:readings")
            machine_ids = set()
            for k in keys:
                raw = k.decode() if isinstance(k, bytes) else k
                parts = raw.split(":")
                if len(parts) >= 2:
                    machine_ids.add(parts[1])
            stats["total_machines_monitored"] = len(machine_ids)
        except Exception as exc:
            logger.warning("Dashboard stats: Redis keys error — {}", exc)

    if db is None:
        return stats

    # ── Anomalies last 24h ─────────────────────────────────────────────────────
    try:
        stats["anomalies_last_24h"] = await db["anomalies"].count_documents(
            {"timestamp": {"$gte": since_24.isoformat()}}
        )
    except Exception:
        pass

    # ── Pending alerts ─────────────────────────────────────────────────────────
    try:
        stats["alerts_pending_approval"] = await db["alerts"].count_documents({"approved": None})
    except Exception:
        pass

    # ── Avg resolution time (approved alerts only) ─────────────────────────────
    try:
        resolved = await db["alerts"].find(
            {"approved": True, "approved_at": {"$exists": True}, "created_at": {"$exists": True}},
            {"created_at": 1, "approved_at": 1},
        ).to_list(200)
        if resolved:
            from datetime import datetime as _dt, timezone as _tz
            deltas = []
            for doc in resolved:
                try:
                    def _parse(v):
                        if isinstance(v, _dt):
                            dt = v
                        else:
                            dt = _dt.fromisoformat(str(v).replace("Z", "+00:00"))
                        return dt if dt.tzinfo else dt.replace(tzinfo=_tz.utc)
                    delta = (_parse(doc["approved_at"]) - _parse(doc["created_at"])).total_seconds()
                    if delta > 0:
                        deltas.append(delta)
                except Exception:
                    pass
            if deltas:
                stats["avg_resolution_time_minutes"] = round(sum(deltas) / len(deltas) / 60, 1)
    except Exception:
        pass

    # ── Failure type distribution (anomalies collection) ──────────────────────
    try:
        pipeline = [
            {"$match": {"failure_type_prediction": {"$ne": None}}},
            {"$group": {"_id": "$failure_type_prediction", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        rows = await db["anomalies"].aggregate(pipeline).to_list(10)
        stats["failure_type_distribution"] = {r["_id"]: r["count"] for r in rows}
    except Exception:
        pass

    # ── Alerts by severity ─────────────────────────────────────────────────────
    try:
        pipeline = [
            {"$group": {"_id": "$root_cause_report.severity", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        rows = await db["alerts"].aggregate(pipeline).to_list(10)
        stats["alerts_by_severity"] = {r["_id"]: r["count"] for r in rows if r["_id"]}
    except Exception:
        pass

    # ── Latest drift report ────────────────────────────────────────────────────
    try:
        drift_docs = await db["drift_reports"].find(
            {}, {"_id": 0, "is_drifted": 1, "run_at": 1}
        ).sort("run_at", -1).limit(1).to_list(length=1)
        if drift_docs:
            stats["drift_detected"]  = drift_docs[0].get("is_drifted")
            stats["last_drift_check"] = drift_docs[0].get("run_at")
    except Exception:
        pass

    return stats


@router.get("/machines")
async def dashboard_machines(request: Request, limit: int = 50):
    """
    Per-machine health summary.
    Combines Redis (latest reading) + MongoDB (last anomaly, open alerts).
    """
    db    = _get_mongo(request)
    redis = _get_redis(request)

    machine_ids: list[str] = []

    # Discover machines from Redis history keys
    if redis and redis.is_connected:
        try:
            keys = await redis._client.keys("sensor:*:readings")
            for k in keys:
                raw = k.decode() if isinstance(k, bytes) else k
                parts = raw.split(":")
                if len(parts) >= 2:
                    machine_ids.append(parts[1])
        except Exception:
            pass

    # Fall back to MongoDB anomalies if Redis has no keys
    if not machine_ids and db is not None:
        try:
            machine_ids = await db["anomalies"].distinct("machine_id")
        except Exception:
            pass

    machine_ids = sorted(set(machine_ids))[:limit]

    machines = []
    for mid in machine_ids:
        entry: dict = {"machine_id": mid, "status": "NORMAL", "last_seen": None,
                       "last_anomaly": None, "open_alerts": 0, "failure_probability": 0.0}

        # Latest reading from Redis
        if redis and redis.is_connected:
            try:
                raw = await redis._client.lindex(f"sensor:{mid}:history", 0)
                if raw:
                    import json
                    reading = json.loads(raw)
                    entry["last_seen"] = reading.get("timestamp")
            except Exception:
                pass

        if db is not None:
            # Last anomaly
            try:
                doc = await db["anomalies"].find_one(
                    {"machine_id": mid},
                    sort=[("timestamp", -1)],
                    projection={"_id": 0, "timestamp": 1, "failure_probability": 1,
                                "failure_type_prediction": 1, "is_anomaly": 1},
                )
                if doc:
                    entry["last_anomaly"]        = doc.get("timestamp")
                    entry["failure_probability"] = doc.get("failure_probability", 0.0)
                    fprob = entry["failure_probability"]
                    entry["status"] = (
                        "CRITICAL" if fprob > 0.9 else
                        "WARNING"  if fprob > 0.5 else
                        "NORMAL"
                    )
            except Exception:
                pass

            # Open (pending) alerts
            try:
                entry["open_alerts"] = await db["alerts"].count_documents(
                    {"machine_id": mid, "approved": None}
                )
            except Exception:
                pass

        machines.append(entry)

    return {"machines": machines, "count": len(machines)}
