"""
tests/test_api.py — 20 tests for FastAPI routes.

Uses FastAPI TestClient with mocked app.state services.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.anomaly import AnomalyResult
from app.models.alert import MaintenanceAlert, RootCauseReport


# ── App factory with mocked state ──────────────────────────────────────────────

def make_test_app(
    anomaly_result: AnomalyResult | None = None,
    mongo_alerts: list[dict] | None = None,
):
    """Build the FastAPI app and inject mock state."""
    from app.main import create_app

    app = create_app()

    # Default anomaly result: normal reading
    if anomaly_result is None:
        anomaly_result = AnomalyResult(
            machine_id="M001", anomaly_score=0.1, failure_probability=0.1,
            is_anomaly=False, sensor_deltas={}, ml_model_used="isolation_forest",
        )

    # Mock detector
    mock_detector = MagicMock()
    mock_detector.run = AsyncMock(return_value=anomaly_result)

    # Mock orchestrator
    mock_orch = MagicMock()
    mock_orch.run = AsyncMock(return_value={"is_anomaly": False, "alert": None})

    # Mock Redis
    mock_redis = MagicMock()
    mock_redis.is_connected = True
    mock_redis.get_history  = AsyncMock(return_value=[])

    # Mock MongoDB
    mock_mongo = _make_mock_mongo(mongo_alerts or [])

    # Mock connection manager
    mock_ws = MagicMock()
    mock_ws.broadcast = AsyncMock(return_value=None)

    @app.on_event("startup")  # type: ignore[call-arg]
    async def _inject():
        pass

    app.state.detector      = mock_detector
    app.state.orchestrator  = mock_orch
    app.state.redis         = mock_redis
    app.state.mongo_db      = mock_mongo
    app.state.ws_manager    = mock_ws
    app.state.ml            = MagicMock(is_ready=True)
    app.state.qdrant        = MagicMock()
    app.state.context_retriever = MagicMock()
    app.state.amem          = MagicMock(is_ready=True)
    app.state.letta         = MagicMock(is_ready=True)
    app.state.alert_generator = MagicMock()

    return app


def _make_mock_mongo(alerts: list[dict]):
    """Create a motor-like mock that returns the given alerts from find()."""
    mock_db   = MagicMock()
    mock_coll = MagicMock()

    # count_documents
    mock_coll.count_documents = AsyncMock(return_value=len(alerts))

    # find() → cursor → to_list()
    mock_cursor = MagicMock()
    mock_cursor.sort   = MagicMock(return_value=mock_cursor)
    mock_cursor.limit  = MagicMock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=alerts)
    mock_coll.find = MagicMock(return_value=mock_cursor)

    # find_one
    mock_coll.find_one  = AsyncMock(return_value=alerts[0] if alerts else None)
    mock_coll.update_one = AsyncMock(return_value=None)
    mock_coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="x"))

    # aggregate
    mock_coll.aggregate = MagicMock(return_value=MagicMock(
        to_list=AsyncMock(return_value=[])
    ))

    mock_db.__getitem__ = MagicMock(return_value=mock_coll)
    return mock_db


VALID_READING = {
    "machine_id":          "M001",
    "air_temperature":     298.1,
    "process_temperature": 308.6,
    "rotational_speed":    1500.0,
    "torque":              40.0,
    "tool_wear":           50.0,
}

SAMPLE_ALERT = {
    "alert_id":    "aaa-bbb-ccc-ddd",
    "machine_id":  "M001",
    "session_id":  "sess-001",
    "approved":    None,
    "created_at":  "2026-03-25T08:00:00",
    "root_cause_report": {
        "severity":   "HIGH",
        "confidence": 0.88,
        "root_cause": "Cooling failure",
    },
    "plain_language_explanation": "Machine M001 is overheating.",
}


# ── /api/sensors tests ─────────────────────────────────────────────────────────

class TestSensorsAPI:

    def test_ingest_valid_reading_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/api/sensors/ingest", json=VALID_READING)
        assert r.status_code == 200

    def test_ingest_returns_anomaly_result_schema(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/sensors/ingest", json=VALID_READING)
        body   = r.json()
        assert "is_anomaly"          in body
        assert "anomaly_score"       in body
        assert "failure_probability" in body

    def test_ingest_missing_field_returns_422(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        bad    = {k: v for k, v in VALID_READING.items() if k != "machine_id"}
        r      = client.post("/api/sensors/ingest", json=bad)
        assert r.status_code == 422

    def test_ingest_invalid_temperature_returns_422(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        bad    = {**VALID_READING, "air_temperature": -5.0}
        r      = client.post("/api/sensors/ingest", json=bad)
        assert r.status_code == 422

    def test_ingest_negative_tool_wear_returns_422(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        bad    = {**VALID_READING, "tool_wear": -1.0}
        r      = client.post("/api/sensors/ingest", json=bad)
        assert r.status_code == 422

    def test_sensor_history_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/sensors/M001/history")
        assert r.status_code == 200

    def test_sensor_history_invalid_n_returns_422(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/sensors/M001/history?n=999")
        assert r.status_code == 422


# ── /api/alerts tests ──────────────────────────────────────────────────────────

class TestAlertsAPI:

    def test_list_alerts_returns_200(self):
        app    = make_test_app(mongo_alerts=[SAMPLE_ALERT])
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/alerts")
        assert r.status_code == 200

    def test_list_alerts_returns_alerts_key(self):
        app    = make_test_app(mongo_alerts=[SAMPLE_ALERT])
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.get("/api/alerts").json()
        assert "alerts" in body
        assert "count"  in body

    def test_list_alerts_pagination_limit(self):
        app    = make_test_app(mongo_alerts=[SAMPLE_ALERT])
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/alerts?limit=5")
        assert r.status_code == 200

    def test_get_alert_found(self):
        app    = make_test_app(mongo_alerts=[SAMPLE_ALERT])
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/alerts/aaa-bbb-ccc-ddd")
        assert r.status_code == 200

    def test_get_alert_not_found(self):
        mongo  = _make_mock_mongo([])
        mongo["alerts"].find_one = AsyncMock(return_value=None)
        app    = make_test_app(mongo_alerts=[])
        app.state.mongo_db = mongo
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/alerts/nonexistent-id")
        assert r.status_code == 404

    def test_approve_pending_alert(self):
        app    = make_test_app(mongo_alerts=[SAMPLE_ALERT])
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/alerts/aaa-bbb-ccc-ddd/approve",
                             json={"approved_by": "eng1"})
        assert r.status_code == 200
        assert r.json()["status"] == "approved"

    def test_approve_already_decided_returns_409(self):
        decided = {**SAMPLE_ALERT, "approved": True}
        app     = make_test_app(mongo_alerts=[decided])
        client  = TestClient(app, raise_server_exceptions=False)
        r       = client.post("/api/alerts/aaa-bbb-ccc-ddd/approve",
                              json={"approved_by": "eng1"})
        assert r.status_code == 409

    def test_reject_alert(self):
        app    = make_test_app(mongo_alerts=[SAMPLE_ALERT])
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/alerts/aaa-bbb-ccc-ddd/reject",
                             json={"rejection_reason": "False positive",
                                   "rejected_by": "eng1"})
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"

    def test_alert_stats_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/alerts/stats")
        assert r.status_code == 200
        assert "stats" in r.json()


# ── /api/dashboard tests ───────────────────────────────────────────────────────

class TestDashboardAPI:

    def test_dashboard_stats_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/dashboard/stats")
        assert r.status_code == 200

    def test_dashboard_stats_contains_expected_keys(self):
        app  = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        body = client.get("/api/dashboard/stats").json()
        for key in ("total_machines_monitored", "anomalies_last_24h",
                    "alerts_pending_approval", "failure_type_distribution"):
            assert key in body, f"Missing key: {key}"

    def test_dashboard_machines_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/dashboard/machines")
        assert r.status_code == 200
        assert "machines" in r.json()


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/health")
        assert r.status_code == 200

    def test_health_contains_all_service_keys(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.get("/health").json()
        for key in ("ml_ready", "redis_connected", "mongo_connected",
                    "qdrant_connected", "orchestrator_ready"):
            assert key in body
