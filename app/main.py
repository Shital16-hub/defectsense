"""
DefectSense — FastAPI application entry point.

Startup sequence:
  1. Load ML models (LSTM Autoencoder + Isolation Forest)
  2. Connect to Redis
  3. Connect to MongoDB (optional — degrades gracefully if unavailable)
  4. Connect to Qdrant + load embedding model (optional)
  5. Wire up agents: AnomalyDetector, ContextRetriever, A-MEM, Letta,
                     RootCauseReasoner, AlertGenerator, Orchestrator
  6. Mount API routers + WebSocket hub
  7. Start background timeout checker

Run:
    uvicorn app.main:app --port 8080 --reload
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

# Disable LangSmith tracing if quota exceeded or not needed
import os as _os
if not _os.getenv("LANGCHAIN_TRACING_FORCE", ""):
    _os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Background task: approval timeout checker ──────────────────────────────────

_pending_threads: dict[str, str] = {}  # thread_id → ISO timeout datetime


async def _approval_timeout_loop(app: FastAPI) -> None:
    """Check pending alerts every 60 s; auto-approve if past timeout."""
    from datetime import datetime, timezone
    timeout_minutes = int(os.getenv("HUMAN_APPROVAL_TIMEOUT_MINUTES", "15"))

    while True:
        await asyncio.sleep(60)
        now = datetime.now(tz=timezone.utc)
        orch = getattr(app.state, "orchestrator", None)
        db   = getattr(app.state, "mongo_db",    None)

        if orch is None or db is None:
            continue

        try:
            # Find pending alerts older than timeout
            cutoff = now.isoformat()
            cursor = db["alerts"].find(
                {"approved": None},
                {"alert_id": 1, "session_id": 1, "machine_id": 1, "created_at": 1},
            )
            async for doc in cursor:
                created_str = doc.get("created_at", "")
                try:
                    created = datetime.fromisoformat(str(created_str).replace("Z", "+00:00"))
                    if created.tzinfo is None:
                        from datetime import timezone as _tz
                        created = created.replace(tzinfo=_tz.utc)
                except Exception:
                    continue

                age_minutes = (now - created).total_seconds() / 60
                if age_minutes >= timeout_minutes:
                    thread_id = f"{doc['machine_id']}:{doc['session_id']}"
                    logger.warning(
                        "Timeout: alert {} pending {:.0f} min — auto-approving as CRITICAL",
                        doc["alert_id"][:8], age_minutes,
                    )
                    try:
                        await orch.resume(
                            thread_id,
                            approved=True,
                            approved_by="timeout_auto",
                            auto=True,
                        )
                    except Exception as exc:
                        logger.warning("Timeout auto-approve failed: {}", exc)
                        # Fallback: update MongoDB directly
                        from datetime import timezone as _tz
                        await db["alerts"].update_one(
                            {"alert_id": doc["alert_id"]},
                            {"$set": {
                                "approved":      True,
                                "approved_by":   "timeout_auto",
                                "auto_approved": True,
                                "approved_at":   now.isoformat(),
                            }},
                        )
        except Exception as exc:
            logger.warning("Timeout checker error: {}", exc)


# ── Scheduled task: nightly drift check ────────────────────────────────────────

async def run_drift_check(app: FastAPI) -> None:
    """APScheduler job — runs at 03:00 UTC."""
    logger.info("Nightly drift check: starting")
    try:
        drift_svc = getattr(app.state, "drift_monitor", None)
        redis_svc = getattr(app.state, "redis", None)
        if drift_svc and redis_svc:
            result = await drift_svc.run_full_drift_check(redis_svc)
            logger.info(
                "Drift check complete: is_drifted={} share={:.0%}",
                result.get("is_drifted"),
                result.get("drift_share", 0),
            )
    except Exception as exc:
        logger.warning("Drift check failed (non-fatal): {}", exc)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and teardown all services around the app lifetime."""
    logger.info("=" * 60)
    logger.info("  DefectSense — starting up")
    logger.info("=" * 60)

    # ── Azure Blob Storage Service ─────────────────────────────────────────────
    from app.services.blob_storage_service import BlobStorageService
    blob_service = BlobStorageService(
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        container_name=os.getenv("AZURE_STORAGE_CONTAINER", "defectsense-models"),
    )
    app.state.blob_storage = blob_service

    # ── ML Service ─────────────────────────────────────────────────────────────
    from app.services.ml_service import MLService
    ml_service = MLService(blob_service=blob_service)
    ml_service.load()
    app.state.ml = ml_service

    # ── Redis Service ──────────────────────────────────────────────────────────
    from app.services.redis_service import RedisService
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_service = RedisService(url=redis_url)
    await redis_service.init()
    app.state.redis = redis_service

    # ── PostgreSQL (optional) ──────────────────────────────────────────────────
    from app.services.postgres_service import PostgresService
    postgres_service = PostgresService(os.getenv("POSTGRES_URL"))
    postgres_service.init()
    app.state.postgres = postgres_service
    if postgres_service.is_connected:
        row_count = postgres_service.get_row_count()
        logger.info("PostgreSQL: connected ({:,} rows)", row_count)
    else:
        logger.warning("PostgreSQL: unavailable (degraded)")

    # ── MongoDB (optional) ─────────────────────────────────────────────────────
    mongo_db = None
    mongo_url = os.getenv("MONGODB_URL", "")
    if mongo_url:
        try:
            import motor.motor_asyncio as motor
            mongo_client = motor.AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=3000)
            db_name = os.getenv("MONGODB_DB_NAME", "defectsense")
            mongo_db = mongo_client[db_name]
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
    qdrant_service = None
    for _attempt in range(3):
        try:
            _svc = QdrantService(url=qdrant_url, api_key=qdrant_key)
            await _svc.init()
            qdrant_service = _svc
            logger.info("Qdrant: connected and embedding model loaded")
            break
        except Exception as exc:
            if _attempt < 2:
                logger.warning("Qdrant connect attempt {} failed — retrying in 3s: {}", _attempt + 1, exc)
                await asyncio.sleep(3)
            else:
                logger.warning("Qdrant unavailable after 3 attempts — RAG context disabled: {}", exc)
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
        reasoning_model=os.getenv("GROQ_MODEL_REASONING", "llama-3.3-70b-versatile"),
        fast_model=os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant"),
    )
    app.state.reasoner = reasoner
    logger.info("RootCauseReasonerAgent: ready")

    # ── Alert Generator Agent ──────────────────────────────────────────────────
    from app.agents.alert_generator import AlertGeneratorAgent
    alert_generator = AlertGeneratorAgent(
        mongo_db=mongo_db,
        redis_service=redis_service,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        fast_model=os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant"),
    )
    app.state.alert_generator = alert_generator
    logger.info("AlertGeneratorAgent: ready")

    # ── Orchestrator ───────────────────────────────────────────────────────────
    from app.agents.orchestrator import DefectSenseOrchestrator
    orchestrator = DefectSenseOrchestrator(
        detector=detector,
        context_retriever=context_retriever,
        amem=amem_service,
        reasoner=reasoner,
        alert_generator=alert_generator,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        auto_approve_threshold=float(os.getenv("AUTO_APPROVE_CONFIDENCE_THRESHOLD", "0.95")),
        approval_timeout_minutes=int(os.getenv("HUMAN_APPROVAL_TIMEOUT_MINUTES", "15")),
        app_base_url=os.getenv("APP_BASE_URL", "http://localhost:8080"),
        mongo_db=mongo_db,
    )
    orchestrator.build()
    app.state.orchestrator = orchestrator
    logger.info("Orchestrator: LangGraph pipeline ready")

    # ── WebSocket Connection Manager ───────────────────────────────────────────
    from app.api.routes.sensors import ConnectionManager
    app.state.ws_manager = ConnectionManager()

    # ── Evaluation Services ────────────────────────────────────────────────────
    from app.services.evaluation_service import (
        RAGEvaluationService, LLMJudgeEvaluationService, run_nightly_evaluation,
    )
    rag_eval_service = RAGEvaluationService(
        mongo_db=mongo_db,
        qdrant_service=qdrant_service,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
    )
    llm_judge_service = LLMJudgeEvaluationService(
        mongo_db=mongo_db,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
    )
    app.state.rag_eval_service  = rag_eval_service
    app.state.llm_judge_service = llm_judge_service
    logger.info("Evaluation services: RAG + LLM-Judge ready")

    # ── Drift Monitoring Service ───────────────────────────────────────────────
    from app.services.drift_monitoring_service import DriftMonitoringService
    drift_service = DriftMonitoringService(
        mongo_db=mongo_db,
        postgres_url=os.getenv("POSTGRES_URL"),
    )
    try:
        await drift_service.init()
        logger.info("DriftMonitoringService: ready")
    except Exception as exc:
        logger.warning("DriftMonitoringService init failed: {}", exc)
    app.state.drift_monitor = drift_service

    # ── Evaluation Scheduler ───────────────────────────────────────────────────
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        run_nightly_evaluation,
        "cron",
        hour=2,
        minute=0,
        args=[app],
        id="nightly_evaluation",
        replace_existing=True,
    )
    scheduler.add_job(
        run_drift_check,
        "cron",
        hour=3,
        minute=0,
        args=[app],
        id="nightly_drift_check",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Evaluation scheduler: nightly job scheduled at 02:00 UTC")
    logger.info("Drift check scheduler: nightly job scheduled at 03:00 UTC")

    # ── Background: approval timeout checker ───────────────────────────────────
    timeout_task = asyncio.create_task(_approval_timeout_loop(app))

    logger.info("DefectSense: all services ready — listening for sensor data")
    logger.info("=" * 60)

    yield  # ← application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    timeout_task.cancel()
    scheduler.shutdown()
    logger.info("DefectSense: shutting down")
    await redis_service.close()
    postgres_service.close()
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
    from app.api.routes.sensors           import router as sensors_router
    from app.api.routes.alerts            import router as alerts_router
    from app.api.routes.dashboard         import router as dashboard_router
    from app.api.routes.maintenance_logs  import router as maintenance_logs_router
    from app.api.routes.evaluation        import router as evaluation_router
    from app.api.websocket                import router as ws_router

    app.include_router(sensors_router,          prefix="/api/sensors",           tags=["sensors"])
    app.include_router(alerts_router,           prefix="/api/alerts",            tags=["alerts"])
    app.include_router(dashboard_router,        prefix="/api/dashboard",         tags=["dashboard"])
    app.include_router(maintenance_logs_router, prefix="/api/maintenance-logs",  tags=["maintenance-logs"])
    app.include_router(evaluation_router,       prefix="/api/evaluation",        tags=["evaluation"])
    app.include_router(ws_router,               prefix="/ws",                    tags=["websocket"])

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {
            "status":              "ok",
            "ml_ready":            app.state.ml.is_ready,
            "blob_storage_ready":  getattr(app.state, "blob_storage", None) is not None
                                   and app.state.blob_storage.is_available,
            "redis_connected":     app.state.redis.is_connected,
            "mongo_connected":     app.state.mongo_db is not None,
            "qdrant_connected":    app.state.qdrant is not None,
            "rag_ready":           app.state.context_retriever is not None,
            "amem_ready":          app.state.amem is not None,
            "letta_ready":         app.state.letta.is_ready,
            "reasoner_ready":      True,
            "alert_generator_ready": True,
            "orchestrator_ready":  app.state.orchestrator is not None,
            "postgres_ready":      getattr(app.state, "postgres", None) is not None
                                   and app.state.postgres.is_connected,
            "drift_monitor_ready": getattr(app.state, "drift_monitor", None) is not None
                                   and app.state.drift_monitor.is_ready,
        }

    return app


app = create_app()
