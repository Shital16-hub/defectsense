"""
Evaluation API — retrieve and trigger LLM-as-judge evaluation results.

Endpoints:
  GET /api/evaluation/latest   — latest result per eval_type
  GET /api/evaluation/history  — last 30 results (optional ?eval_type=rag|llm_judge)
  GET /api/evaluation/run      — manually trigger both evaluations (background task)
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Query, Request

router = APIRouter()

COLL_EVAL = "evaluation_results"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_mongo(request: Request):
    db = getattr(request.app.state, "mongo_db", None)
    return db


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/latest")
async def get_latest_results(request: Request):
    """Return the most recent result for each eval_type (rag and llm_judge)."""
    db = _get_mongo(request)
    if db is None:
        return {"rag": None, "llm_judge": None}

    async def _latest(eval_type: str):
        try:
            cursor = (
                db[COLL_EVAL]
                .find({"eval_type": eval_type}, {"_id": 0})
                .sort("run_at", -1)
                .limit(1)
            )
            docs = await cursor.to_list(length=1)
            return docs[0] if docs else None
        except Exception:
            return None

    rag, llm = await asyncio.gather(_latest("rag"), _latest("llm_judge"))
    return {"rag": rag, "llm_judge": llm}


@router.get("/history")
async def get_evaluation_history(
    request:   Request,
    eval_type: Optional[str] = Query(None, description="Filter: 'rag' or 'llm_judge'"),
    limit:     int           = Query(30, ge=1, le=100),
):
    """Return evaluation history, newest first."""
    db = _get_mongo(request)
    if db is None:
        return {"results": [], "count": 0}

    try:
        filt: dict = {}
        if eval_type:
            filt["eval_type"] = eval_type

        cursor = (
            db[COLL_EVAL]
            .find(filt, {"_id": 0, "sample_scores": 0})
            .sort("run_at", -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        return {"results": results, "count": len(results)}
    except Exception as exc:
        return {"results": [], "count": 0, "error": str(exc)}


@router.get("/run")
async def trigger_evaluation(request: Request):
    """Manually trigger both evaluations as a background asyncio task."""
    from app.services.evaluation_service import run_nightly_evaluation

    asyncio.create_task(run_nightly_evaluation(request.app))
    return {
        "status":  "started",
        "message": "Evaluation started in background — check /api/evaluation/latest in ~2 minutes",
    }
