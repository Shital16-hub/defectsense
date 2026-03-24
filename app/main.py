"""
DefectSense — FastAPI application entry point.

Startup sequence:
  1. Load ML models (LSTM Autoencoder + Isolation Forest)
  2. Connect to Redis
  3. Connect to MongoDB (optional — degrades gracefully if unavailable)
  4. Wire up AnomalyDetectorAgent with all services
  5. Mount API routers

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

    # ── Anomaly Detector Agent ─────────────────────────────────────────────────
    from app.agents.anomaly_detector import AnomalyDetectorAgent
    detector = AnomalyDetectorAgent(
        ml_service=ml_service,
        redis_service=redis_service,
        mongo_db=mongo_db,
    )
    app.state.detector = detector

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
            "status": "ok",
            "ml_ready": app.state.ml.is_ready,
            "redis_connected": app.state.redis.is_connected,
            "mongo_connected": app.state.mongo_db is not None,
        }

    return app


app = create_app()
