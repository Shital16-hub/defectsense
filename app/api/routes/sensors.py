"""
Sensor API Routes

  POST /api/sensors/ingest              — ingest one SensorReading, run anomaly detection
  GET  /api/sensors/{machine_id}/history — recent readings for a machine
  WS   /ws/sensors                       — WebSocket stream of live anomaly results
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from loguru import logger

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading

router = APIRouter()


# ── POST /api/sensors/ingest ──────────────────────────────────────────────────

@router.post("/ingest", response_model=AnomalyResult, summary="Ingest a sensor reading")
async def ingest_sensor(reading: SensorReading, request: Request) -> AnomalyResult:
    """
    Accept one SensorReading, run anomaly detection, return AnomalyResult.

    If an anomaly is detected the result is also:
    - Logged to MongoDB
    - Cached in Redis
    - Broadcast to all connected WebSocket clients
    """
    detector   = request.app.state.detector
    orchestrator = getattr(request.app.state, "orchestrator", None)
    ws_manager: ConnectionManager = request.app.state.ws_manager

    result: AnomalyResult = await detector.run(reading)

    if result.is_anomaly:
        # Broadcast raw anomaly to WebSocket clients
        asyncio.create_task(
            ws_manager.broadcast(result.model_dump(mode="json"))
        )
        # Kick off full pipeline: root cause → alert (non-blocking)
        if orchestrator is not None:
            asyncio.create_task(_run_pipeline(orchestrator, reading))

    return result


async def _run_pipeline(orchestrator, reading: SensorReading) -> None:
    """Background task: run full orchestrator pipeline for an anomaly."""
    try:
        state = await orchestrator.run(reading)
        alert = state.get("alert")
        if alert:
            logger.info(
                "Pipeline complete: alert {} | machine={} approved={}",
                alert.alert_id[:8], alert.machine_id, alert.approved,
            )
        else:
            logger.info(
                "Pipeline complete: no alert generated for machine={} (anomaly={} approved={})",
                reading.machine_id,
                state.get("is_anomaly"),
                state.get("approved"),
            )
    except Exception as exc:
        logger.error("Pipeline error for machine={}: {}", reading.machine_id, exc)


# ── GET /api/sensors/{machine_id}/history ─────────────────────────────────────

@router.get(
    "/{machine_id}/history",
    response_model=list[SensorReading],
    summary="Recent sensor readings for a machine",
)
async def get_history(machine_id: str, request: Request, n: int = 50) -> list[SensorReading]:
    """Return the last `n` sensor readings for `machine_id` (max 200)."""
    if n < 1 or n > 200:
        raise HTTPException(status_code=422, detail="n must be between 1 and 200")

    redis = request.app.state.redis
    readings = await redis.get_history(machine_id, n=n)
    return readings


# ── WebSocket /ws/sensors ─────────────────────────────────────────────────────

@router.websocket("/stream")
async def websocket_sensor_stream(websocket: WebSocket, request: Request) -> None:
    """
    WebSocket endpoint — broadcasts AnomalyResult JSON to all connected clients
    whenever an anomaly is detected via POST /api/sensors/ingest.

    Also streams a heartbeat every 5 seconds so the client knows the connection
    is alive even during quiet periods.
    """
    ws_manager: ConnectionManager = websocket.app.state.ws_manager
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client can send pings, we echo them
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                await websocket.send_text(json.dumps({"type": "pong", "data": data}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket error: {}", exc)
        ws_manager.disconnect(websocket)


# ── WebSocket connection manager ───────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket: client connected ({} total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections = [c for c in self._connections if c is not ws]
        logger.info("WebSocket: client disconnected ({} remaining)", len(self._connections))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Send payload to all connected WebSocket clients; drop dead connections."""
        message = json.dumps(payload, default=str)
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
