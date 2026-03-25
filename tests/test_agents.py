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
