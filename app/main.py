"""
DefectSense — FastAPI application entry point.

Startup sequence:
  1. Load ML models (LSTM Autoencoder + Isolation Forest)
  2. Connect to Redis
  3. Connect to MongoDB (optional — degrades gracefully if unavailable)
  4. Connect to Qdrant + load embedding model (optional)
  5. Wire up AnomalyDetectorAgent + ContextRetrieverAgent
  6. Mount API routers

Run:
    uvicorn app.main:app --port 8080 --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and teardown all services around the app lifetime."""
    logger.info("=" * 60)
    logger.info("  DefectSense — starting up")
    logger.info("=" * 60)

    # ── ML Service ─────────────────────────────────────────────────────────────
    from app.services.ml_service import MLService
    ml_service = MLService()
    ml_service.load()   # synchronous — loads models once
    app.state.ml = ml_service

    # ── Redis Service ──────────────────────────────────────────────────────────
    from app.services.redis_service import RedisService
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_service = RedisService(url=redis_url)
    await redis_service.init()
    app.state.redis = redis_service

    # ── MongoDB (optional) ─────────────────────────────────────────────────────
    mongo_db = None
    mongo_url = os.getenv("MONGODB_URL", "")
    if mongo_url:
        try:
            import motor.motor_asyncio as motor
            mongo_client = motor.AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=3000)
            db_name = os.getenv("MONGODB_DB_NAME", "defectsense")
            mongo_db = mongo_client[db_name]
            # Verify connection
            await mongo_client.admin.command("ping")
            logger.info("MongoDB: connected to '{}'", db_name)
        except Exception as exc:
            logger.warning("MongoDB unavailable — anomalies won't be persisted: {}", exc)
            mongo_db = None
    else:
        logger.info("MongoDB: MONGODB_URL not set — skipping (set it in .env)")
    app.state.mongo_db = mongo_db

    # ── Qdrant Service (optional) ──────────────────────────────────────────────
    from app.services.qdrant_service import QdrantService
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.getenv("QDRANT_API_KEY") or None
    qdrant_service = QdrantService(url=qdrant_url, api_key=qdrant_key)
    try:
        await qdrant_service.init()
        logger.info("Qdrant: connected and embedding model loaded")
    except Exception as exc:
        logger.warning("Qdrant unavailable — RAG context disabled: {}", exc)
        qdrant_service = None
    app.state.qdrant = qdrant_service

    # ── Anomaly Detector Agent ─────────────────────────────────────────────────
    from app.agents.anomaly_detector import AnomalyDetectorAgent
    detector = AnomalyDetectorAgent(
        ml_service=ml_service,
        redis_service=redis_service,
        mongo_db=mongo_db,
    )
    app.state.detector = detector

    # ── Context Retriever Agent ────────────────────────────────────────────────
    from app.agents.context_retriever import ContextRetrieverAgent
    if qdrant_service is not None:
        context_retriever = ContextRetrieverAgent(qdrant=qdrant_service, redis=redis_service)
        logger.info("ContextRetrieverAgent: ready")
    else:
        context_retriever = None
        logger.warning("ContextRetrieverAgent: disabled (Qdrant unavailable)")
    app.state.context_retriever = context_retriever

    # ── A-MEM Service ──────────────────────────────────────────────────────────
    from app.services.amem_service import AMEMService
    amem_service = AMEMService(db=mongo_db)
    try:
        await amem_service.init()
        count = await amem_service.memory_count()
        logger.info("AMEMService: ready ({} notes in memory)", count)
    except Exception as exc:
        logger.warning("AMEMService init failed (non-fatal): {}", exc)
        amem_service = None
    app.state.amem = amem_service

    # ── Letta Service ──────────────────────────────────────────────────────────
    from app.services.letta_service import LettaService
    letta_service = LettaService(db=mongo_db)
    await letta_service.init()
    app.state.letta = letta_service

    # ── Root Cause Reasoner Agent ──────────────────────────────────────────────
    from app.agents.root_cause_reasoner import RootCauseReasonerAgent
    reasoner = RootCauseReasonerAgent(
        amem=amem_service,
        letta=letta_service,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        reasoning_model=os.getenv("GROQ_MODEL_REASONING", "deepseek-r1-distill-llama-70b"),
        fast_model=os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant"),
    )
    app.state.reasoner = reasoner
    logger.info("RootCauseReasonerAgent: ready")

    # ── WebSocket Connection Manager ───────────────────────────────────────────
    from app.api.routes.sensors import ConnectionManager
    app.state.ws_manager = ConnectionManager()

    logger.info("DefectSense: all services ready — listening for sensor data")
    logger.info("=" * 60)

    yield  # ← application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("DefectSense: shutting down")
    await redis_service.close()
    logger.info("DefectSense: goodbye")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="DefectSense",
        description="Manufacturing Defect Root-Cause Agent — Hybrid ML + GenAI",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    from app.api.routes.sensors import router as sensors_router
    app.include_router(sensors_router, prefix="/api/sensors", tags=["sensors"])

    # WebSocket endpoint: ws://localhost:8080/api/sensors/stream

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {
            "status":           "ok",
            "ml_ready":         app.state.ml.is_ready,
            "redis_connected":  app.state.redis.is_connected,
            "mongo_connected":  app.state.mongo_db is not None,
            "qdrant_connected": app.state.qdrant is not None,
            "rag_ready":        app.state.context_retriever is not None,
            "amem_ready":       app.state.amem is not None,
            "letta_ready":      app.state.letta.is_ready,
            "reasoner_ready":   True,
        }

    return app


app = create_app()
