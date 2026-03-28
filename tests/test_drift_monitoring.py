"""
Unit tests for DriftMonitoringService and drift API endpoints.

All tests are fully offline — no real DB, Redis, or Evidently calls
unless explicitly testing drift computation (which uses in-process numpy).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.models.sensor import SensorReading
from datetime import datetime

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normal_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "air_temperature":     rng.uniform(295, 305, n),
        "process_temperature": rng.uniform(308, 313, n),
        "rotational_speed":    rng.uniform(1200, 2000, n),
        "torque":              rng.uniform(10, 70, n),
        "tool_wear":           rng.uniform(0, 250, n),
        "machine_failure":     [0] * n,
    })


def _shifted_df(n: int = 50, multiplier: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "air_temperature":     rng.uniform(295, 305, n) * multiplier,
        "process_temperature": rng.uniform(308, 313, n) * multiplier,
        "rotational_speed":    rng.uniform(1200, 2000, n) * multiplier,
        "torque":              rng.uniform(10, 70, n) * multiplier,
        "tool_wear":           rng.uniform(0, 250, n) * multiplier,
    })


def _make_readings(n: int = 50) -> list[SensorReading]:
    rng = np.random.default_rng(7)
    readings = []
    for i in range(n):
        readings.append(SensorReading(
            machine_id="M001",
            air_temperature=float(rng.uniform(296, 304)),
            process_temperature=float(rng.uniform(309, 312)),
            rotational_speed=float(rng.uniform(1300, 1900)),
            torque=float(rng.uniform(15, 65)),
            tool_wear=float(rng.uniform(1, 240)),
            timestamp=datetime.utcnow(),
        ))
    return readings


# ── Service initialisation ─────────────────────────────────────────────────────

class TestDriftServiceInit:
    def test_drift_service_initializes(self):
        """DriftMonitoringService() must not crash."""
        from app.services.drift_monitoring_service import DriftMonitoringService
        svc = DriftMonitoringService()
        assert svc is not None

    @pytest.mark.asyncio
    async def test_init_sets_ready_false_without_data(self):
        """Without postgres or csv, init() handles gracefully."""
        from app.services.drift_monitoring_service import DriftMonitoringService

        svc = DriftMonitoringService(mongo_db=None, postgres_url=None)
        # Patch load_reference_data to simulate total failure
        async def _fail():
            svc._reference_data = None
        svc.load_reference_data = _fail

        await svc.init()
        # _ready is set True even if no data — graceful
        # is_ready property checks both flags
        assert svc.is_ready is False  # _reference_data is None

    @pytest.mark.asyncio
    async def test_load_reference_data_from_csv(self):
        """When PostgreSQL is unavailable, service falls back to CSV."""
        from app.services.drift_monitoring_service import DriftMonitoringService

        csv_df = _normal_df(300)

        svc = DriftMonitoringService(mongo_db=None, postgres_url=None)

        # Patch pd.read_csv and Path.exists so CSV branch succeeds
        fake_path = MagicMock()
        fake_path.exists.return_value = True

        with patch("app.services.drift_monitoring_service.pd.read_csv", return_value=csv_df), \
             patch("app.services.drift_monitoring_service.Path", return_value=fake_path):
            # Trigger the CSV fallback by ensuring postgres_url is None
            await svc.load_reference_data()

        assert svc._reference_data is not None
        assert len(svc._reference_data) > 0
        assert set(FEATURES).issubset(svc._reference_data.columns)


# ── Drift report computation ───────────────────────────────────────────────────

class TestDriftReport:
    @pytest.fixture
    def svc_with_reference(self):
        from app.services.drift_monitoring_service import DriftMonitoringService
        svc = DriftMonitoringService(mongo_db=None, postgres_url=None)
        svc._reference_data = _normal_df(200)[FEATURES]
        svc._ready = True
        return svc

    @pytest.mark.asyncio
    async def test_run_drift_report_returns_structure(self, svc_with_reference):
        """Drift report returns dict with all required keys."""
        current = _normal_df(50)[FEATURES]
        result = await svc_with_reference.run_drift_report(current)

        required = {
            "is_drifted", "drift_share", "n_features_drifted",
            "total_features", "reference_size", "current_size", "run_at",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )
        assert result["total_features"] == len(FEATURES)
        assert result["reference_size"] == 200
        assert result["current_size"] == 50

    @pytest.mark.asyncio
    async def test_no_drift_on_identical_data(self, svc_with_reference):
        """Same distribution → is_drifted should be False."""
        ref = svc_with_reference._reference_data.copy()
        # Take a slice of the reference — same distribution
        current = ref.sample(n=80, random_state=1).reset_index(drop=True)
        result = await svc_with_reference.run_drift_report(current)
        assert result["is_drifted"] is False

    @pytest.mark.asyncio
    async def test_drift_detected_on_shifted_data(self, svc_with_reference):
        """Heavily shifted data → is_drifted should be True."""
        current = _shifted_df(50, multiplier=10.0)
        result = await svc_with_reference.run_drift_report(current)
        assert result["is_drifted"] is True
        assert result["n_features_drifted"] > 0

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_error(self, svc_with_reference):
        """DataFrame with < 10 rows → error dict, no crash."""
        small = _normal_df(5)[FEATURES]
        result = await svc_with_reference.run_drift_report(small)
        assert "error" in result
        assert result["is_drifted"] is False

    @pytest.mark.asyncio
    async def test_no_reference_data_returns_error(self):
        """No reference data loaded → error dict."""
        from app.services.drift_monitoring_service import DriftMonitoringService
        svc = DriftMonitoringService()
        svc._reference_data = None
        current = _normal_df(50)[FEATURES]
        result = await svc.run_drift_report(current)
        assert "error" in result
        assert result["is_drifted"] is False

    @pytest.mark.asyncio
    async def test_report_saved_to_mongodb(self, mock_mongo):
        """Drift report saves document to MongoDB drift_reports collection."""
        from app.services.drift_monitoring_service import DriftMonitoringService

        svc = DriftMonitoringService(mongo_db=mock_mongo, postgres_url=None)
        svc._reference_data = _normal_df(200)[FEATURES]
        svc._ready = True

        current = _normal_df(50)[FEATURES]
        await svc.run_drift_report(current)

        # Verify insert_one was called on the drift_reports collection
        mock_mongo.__getitem__.assert_any_call("drift_reports")
        coll = mock_mongo["drift_reports"]
        coll.insert_one.assert_called_once()


# ── Redis window ───────────────────────────────────────────────────────────────

class TestCurrentWindowData:
    @pytest.mark.asyncio
    async def test_get_current_window_from_redis(self, mock_redis):
        """Mock redis returning readings → DataFrame with FEATURES columns."""
        from app.services.drift_monitoring_service import DriftMonitoringService

        readings = _make_readings(50)
        mock_redis.get_recent_readings = AsyncMock(return_value=readings)

        svc = DriftMonitoringService()
        df = await svc.get_current_window_data(mock_redis, ["M001"], n_per_machine=50)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert set(FEATURES).issubset(df.columns)

    @pytest.mark.asyncio
    async def test_empty_redis_returns_empty_df(self, mock_redis):
        """Redis returning empty list → empty DataFrame."""
        from app.services.drift_monitoring_service import DriftMonitoringService

        mock_redis.get_recent_readings = AsyncMock(return_value=[])

        svc = DriftMonitoringService()
        df = await svc.get_current_window_data(mock_redis, ["M001"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ── Full orchestration ─────────────────────────────────────────────────────────

class TestFullDriftCheck:
    @pytest.mark.asyncio
    async def test_full_drift_check_no_machines(self, mock_redis):
        """Empty Redis → error dict, no crash."""
        from app.services.drift_monitoring_service import DriftMonitoringService

        mock_redis._client = AsyncMock()
        mock_redis._client.keys = AsyncMock(return_value=[])

        svc = DriftMonitoringService()
        result = await svc.run_full_drift_check(mock_redis)

        assert "error" in result
        assert result["is_drifted"] is False


# ── Drift API endpoints ────────────────────────────────────────────────────────

@pytest.fixture
def drift_test_app(mock_mongo, mock_redis):
    """Minimal FastAPI app with drift endpoints and mocked state."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes.evaluation import router as eval_router
    from app.services.drift_monitoring_service import DriftMonitoringService

    app = FastAPI()
    app.include_router(eval_router)

    drift_svc = DriftMonitoringService(mongo_db=mock_mongo)
    drift_svc._reference_data = _normal_df(200)[FEATURES]
    drift_svc._ready = True

    @app.on_event("startup")  # type: ignore[call-arg]
    async def _setup():
        app.state.mongo_db     = mock_mongo
        app.state.redis        = mock_redis
        app.state.drift_monitor = drift_svc
        app.state.rag_eval_service  = None
        app.state.llm_judge_service = None

    return TestClient(app)


class TestDriftAPI:
    def test_drift_api_latest_returns_200(self, drift_test_app):
        resp = drift_test_app.get("/drift/latest")
        assert resp.status_code == 200

    def test_drift_api_history_returns_200(self, drift_test_app):
        resp = drift_test_app.get("/drift/history")
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body

    def test_drift_api_run_returns_started(self, drift_test_app):
        resp = drift_test_app.get("/drift/run")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "started"
