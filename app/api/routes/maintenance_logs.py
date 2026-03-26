"""
Maintenance Logs API — manage the RAG knowledge base.

Endpoints:
  POST /api/maintenance-logs/add        upsert a single log into Qdrant + MongoDB
  POST /api/maintenance-logs/bulk-add   upsert up to 100 logs in batch
  GET  /api/maintenance-logs            list logs from MongoDB (pagination + filter by failure_type, machine_id)
  GET  /api/maintenance-logs/count      count logs in MongoDB and Qdrant
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.models.maintenance import MaintenanceLog

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class BulkAddRequest(BaseModel):
    logs: list[MaintenanceLog] = Field(..., max_length=100)


class AddResponse(BaseModel):
    log_id: str
    mongo_saved: bool
    qdrant_upserted: bool
    message: str


class BulkAddResponse(BaseModel):
    count: int
    mongo_saved: int
    qdrant_upserted: int
    message: str


class CountResponse(BaseModel):
    mongodb_count: int
    qdrant_count: int
    in_sync: bool


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_mongo(request: Request):
    db = getattr(request.app.state, "mongo_db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")
    return db


def _get_qdrant(request: Request):
    qdrant = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant unavailable")
    return qdrant


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/add", response_model=AddResponse)
async def add_maintenance_log(log: MaintenanceLog, request: Request):
    """
    Embed and upsert a single MaintenanceLog into Qdrant, then save to MongoDB.
    Both services must be available; returns 503 if either is down.
    """
    from loguru import logger
    from app.services.mongodb_service import MongoDBService

    db     = _get_mongo(request)
    qdrant = _get_qdrant(request)

    # Upsert to Qdrant
    qdrant_count = await qdrant.upsert_logs([log])
    qdrant_ok    = qdrant_count == 1

    # Save to MongoDB directly via the motor db handle
    from datetime import datetime, timezone
    doc = {**log.model_dump(mode="json"), "saved_at": datetime.now(tz=timezone.utc).isoformat()}
    try:
        await db["maintenance_logs"].insert_one(doc)
        mongo_ok = True
    except Exception as exc:
        logger.warning("add_maintenance_log: MongoDB insert failed — {}", exc)
        mongo_ok = False

    logger.info(
        "add_maintenance_log: log_id={} qdrant={} mongo={}",
        log.log_id, qdrant_ok, mongo_ok,
    )

    return AddResponse(
        log_id=log.log_id,
        mongo_saved=mongo_ok,
        qdrant_upserted=qdrant_ok,
        message="Log added successfully",
    )


@router.post("/bulk-add", response_model=BulkAddResponse)
async def bulk_add_maintenance_logs(body: BulkAddRequest, request: Request):
    """
    Embed and upsert up to 100 MaintenanceLogs into Qdrant, then save to MongoDB.
    Pydantic enforces the 100-item maximum via max_length on the list field.
    """
    from loguru import logger
    from datetime import datetime, timezone

    db     = _get_mongo(request)
    qdrant = _get_qdrant(request)

    logs = body.logs

    # Upsert to Qdrant (single batch call)
    qdrant_count = await qdrant.upsert_logs(logs)

    # Save to MongoDB
    now  = datetime.now(tz=timezone.utc).isoformat()
    docs = [{**log.model_dump(mode="json"), "saved_at": now} for log in logs]
    try:
        result     = await db["maintenance_logs"].insert_many(docs)
        mongo_count = len(result.inserted_ids)
    except Exception as exc:
        logger.warning("bulk_add_maintenance_logs: MongoDB insert_many failed — {}", exc)
        mongo_count = 0

    logger.info(
        "bulk_add_maintenance_logs: {} logs — qdrant={} mongo={}",
        len(logs), qdrant_count, mongo_count,
    )

    return BulkAddResponse(
        count=len(logs),
        mongo_saved=mongo_count,
        qdrant_upserted=qdrant_count,
        message=f"{len(logs)} logs added successfully",
    )


@router.get("", response_model=dict)
async def list_maintenance_logs(
    request:      Request,
    failure_type: Optional[str] = Query(None, description="Filter by failure type: TWF | HDF | PWF | OSF | RNF"),
    machine_id:   Optional[str] = Query(None, description="Filter by machine ID"),
    limit:        int           = Query(50, ge=1, le=500),
    skip:         int           = Query(0, ge=0),
):
    """Return maintenance logs from MongoDB with optional filters and pagination."""
    db = _get_mongo(request)

    query: dict = {}
    if failure_type:
        query["failure_type"] = failure_type
    if machine_id:
        query["machine_id"] = machine_id

    cursor = (
        db["maintenance_logs"]
        .find(query, {"_id": 0})
        .sort("saved_at", -1)
        .skip(skip)
        .limit(limit)
    )
    logs = await cursor.to_list(length=limit)

    return {"logs": logs, "count": len(logs), "skip": skip, "limit": limit}


@router.get("/count", response_model=CountResponse)
async def count_maintenance_logs(request: Request):
    """
    Return the number of maintenance logs in MongoDB and Qdrant.
    Useful for verifying the two stores are in sync.
    """
    db     = _get_mongo(request)
    qdrant = _get_qdrant(request)

    mongo_count  = await db["maintenance_logs"].count_documents({})
    qdrant_count = await qdrant.collection_count()

    return CountResponse(
        mongodb_count=mongo_count,
        qdrant_count=qdrant_count,
        in_sync=mongo_count == qdrant_count,
    )
