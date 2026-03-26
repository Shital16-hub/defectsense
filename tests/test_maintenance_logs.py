"""
tests/test_maintenance_logs.py — unit tests for /api/maintenance-logs endpoints.

Uses FastAPI TestClient with mocked app.state services, following the exact
same pattern as tests/test_api.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.models.maintenance import MaintenanceLog


# ── Fixtures ───────────────────────────────────────────────────────────────────

VALID_LOG = {
    "machine_id":             "M042",
    "date":                   "2024-03-15T08:30:00",
    "failure_type":           "HDF",
    "symptoms":               "Process temperature exceeded 315K, rotational speed dropped below 1400 RPM",
    "root_cause":             "Cooling fan blade fracture causing heat dissipation failure",
    "action_taken":           "Replaced cooling fan assembly, cleaned heat exchange fins",
    "resolution_time_hours":  4.5,
    "technician":             "J. Smith",
    "machine_type":           "M",
}


def _make_mock_qdrant(upsert_count: int = 1, collection_count: int = 5) -> MagicMock:
    mock = MagicMock()
    mock.upsert_logs       = AsyncMock(return_value=upsert_count)
    mock.collection_count  = AsyncMock(return_value=collection_count)
    return mock


def _make_mock_mongo(
    logs: list[dict] | None = None,
    insert_ok: bool = True,
    count: int = 3,
) -> MagicMock:
    logs = logs or []
    mock_db   = MagicMock()
    mock_coll = MagicMock()

    # insert_one
    if insert_ok:
        mock_coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="x"))
    else:
        mock_coll.insert_one = AsyncMock(side_effect=Exception("DB error"))

    # insert_many
    mock_coll.insert_many = AsyncMock(
        return_value=MagicMock(inserted_ids=["a", "b", "c"])
    )

    # count_documents
    mock_coll.count_documents = AsyncMock(return_value=count)

    # find() → cursor → to_list()
    mock_cursor = MagicMock()
    mock_cursor.sort   = MagicMock(return_value=mock_cursor)
    mock_cursor.skip   = MagicMock(return_value=mock_cursor)
    mock_cursor.limit  = MagicMock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=logs)
    mock_coll.find = MagicMock(return_value=mock_cursor)

    # aggregate (used by alerts/dashboard)
    mock_coll.aggregate = MagicMock(return_value=MagicMock(
        to_list=AsyncMock(return_value=[])
    ))
    mock_coll.find_one  = AsyncMock(return_value=None)
    mock_coll.update_one = AsyncMock(return_value=None)

    mock_db.__getitem__ = MagicMock(return_value=mock_coll)
    return mock_db


def make_test_app(
    mock_mongo=None,
    mock_qdrant=None,
    mongo_unavailable: bool = False,
    qdrant_unavailable: bool = False,
):
    """Build the FastAPI app and inject mock services into app.state."""
    from app.main import create_app

    app = create_app()

    app.state.ml            = MagicMock(is_ready=True)
    app.state.redis         = MagicMock(is_connected=True, get_history=AsyncMock(return_value=[]))
    app.state.detector      = MagicMock(run=AsyncMock(return_value=MagicMock(
        is_anomaly=False, anomaly_score=0.1, failure_probability=0.1,
        sensor_deltas={}, ml_model_used="isolation_forest",
    )))
    app.state.orchestrator  = MagicMock(run=AsyncMock(return_value={"is_anomaly": False, "alert": None}))
    app.state.ws_manager    = MagicMock(broadcast=AsyncMock(return_value=None))
    app.state.context_retriever = MagicMock()
    app.state.amem          = MagicMock(is_ready=True)
    app.state.letta         = MagicMock(is_ready=True)
    app.state.alert_generator = MagicMock()

    app.state.mongo_db = None if mongo_unavailable else (mock_mongo or _make_mock_mongo())
    app.state.qdrant   = None if qdrant_unavailable else (mock_qdrant or _make_mock_qdrant())

    return app


# ── POST /api/maintenance-logs/add ────────────────────────────────────────────

class TestAddMaintenanceLog:

    def test_add_valid_log_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/add", json=VALID_LOG)
        assert r.status_code == 200

    def test_add_valid_log_returns_log_id(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.post("/api/maintenance-logs/add", json=VALID_LOG).json()
        assert "log_id" in body
        assert body["log_id"]  # non-empty UUID

    def test_add_valid_log_response_schema(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.post("/api/maintenance-logs/add", json=VALID_LOG).json()
        assert "mongo_saved"      in body
        assert "qdrant_upserted"  in body
        assert "message"          in body

    def test_add_log_missing_required_field_returns_422(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        bad    = {k: v for k, v in VALID_LOG.items() if k != "symptoms"}
        r      = client.post("/api/maintenance-logs/add", json=bad)
        assert r.status_code == 422

    def test_add_log_negative_resolution_time_returns_422(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        bad    = {**VALID_LOG, "resolution_time_hours": -1.0}
        r      = client.post("/api/maintenance-logs/add", json=bad)
        assert r.status_code == 422

    def test_add_log_qdrant_unavailable_returns_503(self):
        app    = make_test_app(qdrant_unavailable=True)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/add", json=VALID_LOG)
        assert r.status_code == 503

    def test_add_log_mongodb_unavailable_returns_503(self):
        app    = make_test_app(mongo_unavailable=True)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/add", json=VALID_LOG)
        assert r.status_code == 503


# ── POST /api/maintenance-logs/bulk-add ───────────────────────────────────────

class TestBulkAddMaintenanceLogs:

    def test_bulk_add_valid_list_returns_200(self):
        logs   = [VALID_LOG, {**VALID_LOG, "failure_type": "TWF"}, {**VALID_LOG, "failure_type": "PWF"}]
        app    = make_test_app(mock_qdrant=_make_mock_qdrant(upsert_count=3))
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/bulk-add", json={"logs": logs})
        assert r.status_code == 200

    def test_bulk_add_returns_count(self):
        logs   = [VALID_LOG, {**VALID_LOG, "failure_type": "TWF"}]
        app    = make_test_app(mock_qdrant=_make_mock_qdrant(upsert_count=2))
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.post("/api/maintenance-logs/bulk-add", json={"logs": logs}).json()
        assert body["count"] == 2

    def test_bulk_add_more_than_100_returns_422(self):
        logs   = [VALID_LOG] * 101
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/bulk-add", json={"logs": logs})
        assert r.status_code == 422

    def test_bulk_add_exactly_100_returns_200(self):
        logs   = [VALID_LOG] * 100
        app    = make_test_app(mock_qdrant=_make_mock_qdrant(upsert_count=100))
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/bulk-add", json={"logs": logs})
        assert r.status_code == 200

    def test_bulk_add_qdrant_unavailable_returns_503(self):
        logs   = [VALID_LOG]
        app    = make_test_app(qdrant_unavailable=True)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/bulk-add", json={"logs": logs})
        assert r.status_code == 503

    def test_bulk_add_mongodb_unavailable_returns_503(self):
        logs   = [VALID_LOG]
        app    = make_test_app(mongo_unavailable=True)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.post("/api/maintenance-logs/bulk-add", json={"logs": logs})
        assert r.status_code == 503


# ── GET /api/maintenance-logs ─────────────────────────────────────────────────

class TestListMaintenanceLogs:

    def test_list_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/maintenance-logs")
        assert r.status_code == 200

    def test_list_returns_logs_and_count_keys(self):
        app    = make_test_app(mock_mongo=_make_mock_mongo(logs=[VALID_LOG]))
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.get("/api/maintenance-logs").json()
        assert "logs"  in body
        assert "count" in body

    def test_list_filter_by_failure_type(self):
        hdf_log = {**VALID_LOG, "failure_type": "HDF"}
        mock_mongo = _make_mock_mongo(logs=[hdf_log])
        app    = make_test_app(mock_mongo=mock_mongo)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/maintenance-logs?failure_type=HDF")
        assert r.status_code == 200
        body   = r.json()
        # The mock returns whatever logs we set up; assert the filter param was accepted
        assert body["count"] == 1

    def test_list_pagination_params_accepted(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/maintenance-logs?limit=10&skip=5")
        assert r.status_code == 200
        body   = r.json()
        assert body["limit"] == 10
        assert body["skip"]  == 5

    def test_list_mongodb_unavailable_returns_503(self):
        app    = make_test_app(mongo_unavailable=True)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/maintenance-logs")
        assert r.status_code == 503


# ── GET /api/maintenance-logs/count ───────────────────────────────────────────

class TestCountMaintenanceLogs:

    def test_count_returns_200(self):
        app    = make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/maintenance-logs/count")
        assert r.status_code == 200

    def test_count_returns_both_counts(self):
        mock_qdrant = _make_mock_qdrant(collection_count=7)
        mock_mongo  = _make_mock_mongo(count=7)
        app    = make_test_app(mock_mongo=mock_mongo, mock_qdrant=mock_qdrant)
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.get("/api/maintenance-logs/count").json()
        assert "mongodb_count" in body
        assert "qdrant_count"  in body
        assert "in_sync"       in body

    def test_count_in_sync_true_when_equal(self):
        mock_qdrant = _make_mock_qdrant(collection_count=5)
        mock_mongo  = _make_mock_mongo(count=5)
        app    = make_test_app(mock_mongo=mock_mongo, mock_qdrant=mock_qdrant)
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.get("/api/maintenance-logs/count").json()
        assert body["in_sync"] is True

    def test_count_in_sync_false_when_different(self):
        mock_qdrant = _make_mock_qdrant(collection_count=3)
        mock_mongo  = _make_mock_mongo(count=5)
        app    = make_test_app(mock_mongo=mock_mongo, mock_qdrant=mock_qdrant)
        client = TestClient(app, raise_server_exceptions=False)
        body   = client.get("/api/maintenance-logs/count").json()
        assert body["in_sync"] is False

    def test_count_mongodb_unavailable_returns_503(self):
        app    = make_test_app(mongo_unavailable=True)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/maintenance-logs/count")
        assert r.status_code == 503

    def test_count_qdrant_unavailable_returns_503(self):
        app    = make_test_app(qdrant_unavailable=True)
        client = TestClient(app, raise_server_exceptions=False)
        r      = client.get("/api/maintenance-logs/count")
        assert r.status_code == 503
