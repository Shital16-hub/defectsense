"""
WebSocket hub — Session 5.

Endpoints:
  WS /ws/alerts   — real-time MaintenanceAlert stream (Redis alerts:new pub/sub)
  WS /ws/sensors  — real-time sensor anomaly stream  (Redis anomalies:new pub/sub)

Redis pub/sub messages are forwarded as-is (JSON strings) to all connected clients.
Client→server messages: {"action": "ping"} → {"action": "pong"}
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

CHANNEL_ALERTS   = "alerts:new"
CHANNEL_ANOMALIES = "anomalies:new"


# ── Connection manager ─────────────────────────────────────────────────────────

class AlertWSManager:
    """
    Manages WebSocket connections and fans out Redis pub/sub messages.

    One background task per channel runs while at least one client is connected.
    """

    def __init__(self) -> None:
        self._alert_clients:   list[WebSocket] = []
        self._anomaly_clients: list[WebSocket] = []
        self._alert_task:   Optional[asyncio.Task] = None
        self._anomaly_task: Optional[asyncio.Task] = None

    # ── Connect / Disconnect ───────────────────────────────────────────────────

    async def connect_alerts(self, ws: WebSocket, redis_service) -> None:
        await ws.accept()
        self._alert_clients.append(ws)
        logger.info("WS /ws/alerts: client connected (total={})", len(self._alert_clients))
        if self._alert_task is None or self._alert_task.done():
            self._alert_task = asyncio.create_task(
                self._redis_listener(redis_service, CHANNEL_ALERTS, self._alert_clients)
            )

    def disconnect_alerts(self, ws: WebSocket) -> None:
        self._remove(ws, self._alert_clients)
        logger.info("WS /ws/alerts: client disconnected (total={})", len(self._alert_clients))

    async def connect_anomalies(self, ws: WebSocket, redis_service) -> None:
        await ws.accept()
        self._anomaly_clients.append(ws)
        logger.info("WS /ws/sensors: client connected (total={})", len(self._anomaly_clients))
        if self._anomaly_task is None or self._anomaly_task.done():
            self._anomaly_task = asyncio.create_task(
                self._redis_listener(redis_service, CHANNEL_ANOMALIES, self._anomaly_clients)
            )

    def disconnect_anomalies(self, ws: WebSocket) -> None:
        self._remove(ws, self._anomaly_clients)
        logger.info("WS /ws/sensors: client disconnected (total={})", len(self._anomaly_clients))

    # ── Broadcast ──────────────────────────────────────────────────────────────

    async def broadcast(self, message: str, clients: list[WebSocket]) -> None:
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._remove(ws, clients)

    # ── Redis listener ─────────────────────────────────────────────────────────

    async def _redis_listener(
        self,
        redis_service,
        channel: str,
        clients: list[WebSocket],
    ) -> None:
        """Subscribe to a Redis channel and fan-out messages to WebSocket clients."""
        logger.info("WS Redis listener: subscribing to '{}'", channel)
        try:
            pubsub = redis_service._client.pubsub()
            await pubsub.subscribe(channel)

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                payload = message["data"]
                if isinstance(payload, bytes):
                    payload = payload.decode()
                if clients:
                    await self.broadcast(payload, clients)
                else:
                    # No clients — unsubscribe and exit
                    break

            await pubsub.unsubscribe(channel)
            logger.info("WS Redis listener: unsubscribed from '{}'", channel)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("WS Redis listener error on '{}': {}", channel, exc)

    @staticmethod
    def _remove(ws: WebSocket, lst: list[WebSocket]) -> None:
        try:
            lst.remove(ws)
        except ValueError:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────
_manager = AlertWSManager()


def get_ws_manager() -> AlertWSManager:
    return _manager


# ── WebSocket endpoints ────────────────────────────────────────────────────────

@router.websocket("/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    Stream MaintenanceAlert JSON to connected clients.
    Sourced from Redis channel 'alerts:new'.
    """
    redis_service = getattr(websocket.app.state, "redis", None)
    if redis_service is None or not redis_service.is_connected:
        await websocket.accept()
        await websocket.send_json({"error": "Redis unavailable — real-time alerts disabled"})
        await websocket.close()
        return

    mgr = get_ws_manager()
    await mgr.connect_alerts(websocket, redis_service)
    try:
        while True:
            data = await websocket.receive_text()
            if data == '{"action":"ping"}':
                await websocket.send_text('{"action":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        mgr.disconnect_alerts(websocket)


@router.websocket("/sensors")
async def ws_sensors(websocket: WebSocket):
    """
    Stream anomaly detection events to connected clients.
    Sourced from Redis channel 'anomalies:new'.
    """
    redis_service = getattr(websocket.app.state, "redis", None)
    if redis_service is None or not redis_service.is_connected:
        await websocket.accept()
        await websocket.send_json({"error": "Redis unavailable — real-time sensors disabled"})
        await websocket.close()
        return

    mgr = get_ws_manager()
    await mgr.connect_anomalies(websocket, redis_service)
    try:
        while True:
            data = await websocket.receive_text()
            if data == '{"action":"ping"}':
                await websocket.send_text('{"action":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        mgr.disconnect_anomalies(websocket)
