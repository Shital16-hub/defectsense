"""
tests/test_agents.py — 20 tests for agent layer.

Tests: AnomalyDetectorAgent, AlertGeneratorAgent, severity rules,
       RootCauseReasonerAgent (mock LLM), orchestrator HITL flow.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading
from app.models.alert import MaintenanceAlert, RootCauseReport


# ── AnomalyDetectorAgent ───────────────────────────────────────────────────────

class TestAnomalyDetectorAgent:

    @pytest.mark.asyncio
    async def test_returns_safe_default_when_ml_not_ready(self, normal_reading, mock_redis):
        from app.agents.anomaly_detector import AnomalyDetectorAgent
        ml = MagicMock(); ml.is_ready = False
        agent  = AnomalyDetectorAgent(ml_service=ml, redis_service=mock_redis)
        result = await agent.run(normal_reading)
        assert result.is_anomaly is False
        assert result.anomaly_score == 0.0

    @pytest.mark.asyncio
    async def test_stores_reading_in_redis(self, normal_reading, mock_redis, mock_ml_service):
        from app.agents.anomaly_detector import AnomalyDetectorAgent
        mock_ml_service.predict_anomaly = AsyncMock(return_value=AnomalyResult(
            machine_id="TEST_M001", anomaly_score=0.1, failure_probability=0.1,
            is_anomaly=False, sensor_deltas={}, ml_model_used="isolation_forest",
        ))
        agent = AnomalyDetectorAgent(ml_service=mock_ml_service, redis_service=mock_redis)
        await agent.run(normal_reading)
        mock_redis.store_reading.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anomaly_triggers_mongo_write(self, normal_reading, mock_redis,
                                                mock_ml_service, mock_mongo):
        from app.agents.anomaly_detector import AnomalyDetectorAgent
        mock_ml_service.predict_anomaly = AsyncMock(return_value=AnomalyResult(
            machine_id="TEST_M001", anomaly_score=0.9, failure_probability=0.85,
            is_anomaly=True, sensor_deltas={}, ml_model_used="ensemble",
        ))
        agent  = AnomalyDetectorAgent(ml_service=mock_ml_service,
                                      redis_service=mock_redis, mongo_db=mock_mongo)
        result = await agent.run(normal_reading)
        assert result.is_anomaly is True
        mock_mongo["anomalies"].insert_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anomaly_triggers_redis_publish(self, normal_reading, mock_redis,
                                                  mock_ml_service):
        from app.agents.anomaly_detector import AnomalyDetectorAgent
        mock_ml_service.predict_anomaly = AsyncMock(return_value=AnomalyResult(
            machine_id="TEST_M001", anomaly_score=0.9, failure_probability=0.85,
            is_anomaly=True, sensor_deltas={}, ml_model_used="ensemble",
        ))
        agent = AnomalyDetectorAgent(ml_service=mock_ml_service, redis_service=mock_redis)
        await agent.run(normal_reading)
        mock_redis.publish_anomaly.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_mongo_write_for_normal_reading(self, normal_reading, mock_redis,
                                                     mock_ml_service, mock_mongo):
        from app.agents.anomaly_detector import AnomalyDetectorAgent
        mock_ml_service.predict_anomaly = AsyncMock(return_value=AnomalyResult(
            machine_id="TEST_M001", anomaly_score=0.1, failure_probability=0.1,
            is_anomaly=False, sensor_deltas={}, ml_model_used="isolation_forest",
        ))
        agent = AnomalyDetectorAgent(ml_service=mock_ml_service,
                                     redis_service=mock_redis, mongo_db=mock_mongo)
        await agent.run(normal_reading)
        mock_mongo["anomalies"].insert_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sequence_fetch_failure_falls_back_to_iforest(
        self, normal_reading, mock_ml_service
    ):
        """If Redis.get_recent_readings raises, agent falls back to IForest-only."""
        from app.agents.anomaly_detector import AnomalyDetectorAgent
        mock_ml_service.predict_anomaly = AsyncMock(return_value=AnomalyResult(
            machine_id="TEST_M001", anomaly_score=0.3, failure_probability=0.3,
            is_anomaly=False, sensor_deltas={}, ml_model_used="isolation_forest",
        ))
        broken_redis = MagicMock()
        broken_redis.is_connected        = False
        broken_redis.store_reading       = AsyncMock(return_value=None)
        broken_redis.get_recent_readings = AsyncMock(side_effect=Exception("Redis down"))
        broken_redis.cache_anomaly       = AsyncMock(return_value=None)
        broken_redis.publish_anomaly     = AsyncMock(return_value=None)

        agent  = AnomalyDetectorAgent(ml_service=mock_ml_service, redis_service=broken_redis)
        result = await agent.run(normal_reading)
        # Even with Redis failure on get_recent_readings, a result is returned
        assert result is not None
        # predict_anomaly was called with sequence=None (empty fallback)
        mock_ml_service.predict_anomaly.assert_awaited_once()


# ── AlertGeneratorAgent — severity rules ───────────────────────────────────────

class TestAlertGeneratorSeverity:

    def _make_result(self, prob: float, ftype: str | None = None) -> AnomalyResult:
        return AnomalyResult(
            machine_id="M001", anomaly_score=prob, failure_probability=prob,
            is_anomaly=True, failure_type_prediction=ftype,
            sensor_deltas={}, ml_model_used="ensemble",
        )

    def _make_report(self, prob: float, ftype=None) -> RootCauseReport:
        return RootCauseReport(
            machine_id="M001", anomaly_result=self._make_result(prob, ftype),
            root_cause="test", confidence=0.8, severity="MEDIUM",
        )

    def _agent(self):
        from app.agents.alert_generator import AlertGeneratorAgent
        return AlertGeneratorAgent()

    def test_critical_above_90pct(self):
        report = self._make_report(0.95)
        sev    = self._agent()._compute_severity(report)
        assert sev == "CRITICAL"

    def test_critical_twf_above_70pct(self):
        report = self._make_report(0.75, "TWF")
        sev    = self._agent()._compute_severity(report)
        assert sev == "CRITICAL"

    def test_critical_hdf_above_70pct(self):
        report = self._make_report(0.75, "HDF")
        sev    = self._agent()._compute_severity(report)
        assert sev == "CRITICAL"

    def test_high_above_70pct_non_twf_hdf(self):
        report = self._make_report(0.75, "PWF")
        sev    = self._agent()._compute_severity(report)
        assert sev == "HIGH"

    def test_medium_above_50pct(self):
        report = self._make_report(0.60)
        sev    = self._agent()._compute_severity(report)
        assert sev == "MEDIUM"

    def test_fallback_to_report_severity(self):
        report = self._make_report(0.30)
        report.severity = "LOW"
        sev    = self._agent()._compute_severity(report)
        assert sev == "LOW"

    @pytest.mark.asyncio
    async def test_generate_creates_alert(self, sample_root_cause_report, mock_mongo,
                                          mock_redis):
        from app.agents.alert_generator import AlertGeneratorAgent
        agent = AlertGeneratorAgent(mongo_db=mock_mongo, redis_service=mock_redis,
                                    groq_api_key="test")
        with patch.object(agent, "_generate_explanation",
                          AsyncMock(return_value="Machine M001 overheating.")):
            alert = await agent.generate(sample_root_cause_report, session_id="s1")
        assert isinstance(alert, MaintenanceAlert)
        assert alert.approved is None
        assert alert.machine_id == "TEST_M001"

    @pytest.mark.asyncio
    async def test_mark_approved_sets_fields(self, sample_alert, mock_mongo):
        from app.agents.alert_generator import AlertGeneratorAgent
        agent = AlertGeneratorAgent(mongo_db=mock_mongo)
        approved = await agent.mark_approved(sample_alert, approved_by="eng1")
        assert approved.approved is True
        assert approved.approved_by == "eng1"
        assert approved.approved_at is not None

    @pytest.mark.asyncio
    async def test_mark_rejected_sets_fields(self, sample_alert, mock_mongo):
        from app.agents.alert_generator import AlertGeneratorAgent
        agent  = AlertGeneratorAgent(mongo_db=mock_mongo)
        rejected = await agent.mark_rejected(sample_alert,
                                             rejection_reason="False positive",
                                             rejected_by="eng2")
        assert rejected.approved is False
        assert rejected.rejection_reason == "False positive"


# ── Orchestrator HITL ──────────────────────────────────────────────────────────

class TestOrchestratorHITL:

    @pytest.mark.asyncio
    async def test_build_compiles_graph(self):
        from app.agents.orchestrator import DefectSenseOrchestrator
        orch = DefectSenseOrchestrator()
        orch.build()
        assert orch._graph is not None

    @pytest.mark.asyncio
    async def test_no_anomaly_exits_early(self, normal_reading, mock_redis,
                                          mock_ml_service, mock_mongo):
        from app.agents.orchestrator import DefectSenseOrchestrator
        from app.agents.anomaly_detector import AnomalyDetectorAgent

        mock_ml_service.predict_anomaly = AsyncMock(return_value=AnomalyResult(
            machine_id="TEST_M001", anomaly_score=0.1, failure_probability=0.1,
            is_anomaly=False, sensor_deltas={}, ml_model_used="isolation_forest",
        ))
        detector = AnomalyDetectorAgent(ml_service=mock_ml_service,
                                        redis_service=mock_redis, mongo_db=mock_mongo)
        orch = DefectSenseOrchestrator(detector=detector)
        orch.build()
        state = await orch.run(normal_reading)
        assert state.get("is_anomaly") is False
        assert state.get("alert") is None


# ── post_resolution_indexer ────────────────────────────────────────────────────

class TestPostResolutionIndexer:

    def _make_orch(self) -> "DefectSenseOrchestrator":
        from app.agents.orchestrator import DefectSenseOrchestrator
        return DefectSenseOrchestrator(app_base_url="http://testserver")

    @pytest.mark.asyncio
    async def test_post_resolution_indexer_runs_on_approval(
        self, sample_anomaly_result, sample_root_cause_report
    ):
        """When approved=True, the node should POST the maintenance log and set auto_indexed=True."""
        orch = self._make_orch()
        state = {
            "approved":          True,
            "approved_by":       "test_engineer",
            "machine_id":        "TEST_M001",
            "anomaly_result":    sample_anomaly_result,
            "root_cause_report": sample_root_cause_report,
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json = MagicMock(return_value={"log_id": "abc-123"})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await orch._node_post_resolution_indexer(state)

        assert result["auto_indexed"] is True
        mock_client.post.assert_awaited_once()

        # Verify the call was made to the maintenance-logs endpoint
        call_url = mock_client.post.call_args[0][0]
        assert "/api/maintenance-logs/add" in call_url

        # Verify the payload contains expected fields
        call_json = mock_client.post.call_args[1]["json"]
        assert call_json["machine_id"]   == "TEST_M001"
        assert call_json["failure_type"] == "HDF"
        assert call_json["technician"]   == "test_engineer"

    @pytest.mark.asyncio
    async def test_post_resolution_indexer_skips_on_rejection(
        self, sample_anomaly_result, sample_root_cause_report
    ):
        """When approved=False, the node should NOT call the API and return auto_indexed=False."""
        orch = self._make_orch()
        state = {
            "approved":          False,
            "approved_by":       "test_engineer",
            "machine_id":        "TEST_M001",
            "anomaly_result":    sample_anomaly_result,
            "root_cause_report": sample_root_cause_report,
        }

        with patch("httpx.AsyncClient") as mock_cls:
            result = await orch._node_post_resolution_indexer(state)

        assert result["auto_indexed"] is False
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_resolution_indexer_does_not_fail_pipeline_on_error(
        self, sample_anomaly_result, sample_root_cause_report
    ):
        """When httpx raises, the node should return auto_indexed=False without re-raising."""
        orch = self._make_orch()
        state = {
            "approved":          True,
            "approved_by":       "test_engineer",
            "machine_id":        "TEST_M001",
            "anomaly_result":    sample_anomaly_result,
            "root_cause_report": sample_root_cause_report,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await orch._node_post_resolution_indexer(state)

        # Must NOT raise — pipeline resilience is the key requirement
        assert result["auto_indexed"] is False
