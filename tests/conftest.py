"""Shared pytest fixtures for DefectSense test suite."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading
from app.models.alert import MaintenanceAlert, RootCauseReport


# ── Event loop ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Sample data fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def normal_reading() -> SensorReading:
    return SensorReading(
        machine_id="TEST_M001",
        volt=176.22,
        rotate=418.50,
        pressure=113.08,
        vibration=45.09,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def failure_reading() -> SensorReading:
    """BWF-like: high vibration indicating bearing wear."""
    return SensorReading(
        machine_id="TEST_M001",
        volt=174.0,
        rotate=443.0,
        pressure=103.0,
        vibration=62.0,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def twf_reading() -> SensorReading:
    """EVF-like: high voltage indicating electrical overload."""
    return SensorReading(
        machine_id="TEST_M002",
        volt=195.0,
        rotate=450.0,
        pressure=101.0,
        vibration=41.0,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def normal_reading_sequence(normal_reading) -> list[SensorReading]:
    """198 slightly-varied normal readings for LSTM sequence (Azure model requires 198 steps)."""
    return [
        SensorReading(
            machine_id="TEST_M001",
            volt=176.0 + i * 0.02,
            rotate=418.0 + i * 0.1,
            pressure=113.0 + i * 0.02,
            vibration=45.0 + i * 0.01,
            timestamp=datetime.utcnow(),
        )
        for i in range(198)
    ]


@pytest.fixture
def sample_anomaly_result() -> AnomalyResult:
    return AnomalyResult(
        machine_id="TEST_M001",
        anomaly_score=0.82,
        failure_probability=0.75,
        is_anomaly=True,
        failure_type_prediction="RSF",
        sensor_deltas={
            "volt": 0.1,
            "rotate": -2.2,
            "pressure": 0.3,
            "vibration": 0.8,
        },
        ml_model_used="lstm_autoencoder",
        reconstruction_error=0.045,
        isolation_score=None,
    )


@pytest.fixture
def sample_root_cause_report(sample_anomaly_result) -> RootCauseReport:
    return RootCauseReport(
        machine_id="TEST_M001",
        anomaly_result=sample_anomaly_result,
        root_cause="Heat dissipation failure — cooling system degraded",
        confidence=0.88,
        evidence=[
            "Process temperature 3.1 std deviations above normal",
            "Rotational speed below threshold",
        ],
        recommended_actions=["Inspect cooling fan", "Schedule maintenance"],
        severity="HIGH",
        reasoning_steps=["THINK: rotation below threshold", "CONCLUDE: RSF pattern"],
    )


@pytest.fixture
def sample_alert(sample_root_cause_report) -> MaintenanceAlert:
    return MaintenanceAlert(
        session_id="test-session-001",
        machine_id="TEST_M001",
        root_cause_report=sample_root_cause_report,
        plain_language_explanation="Machine TEST_M001 is overheating. Contact maintenance.",
        approved=None,
    )


# ── Mock service fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.is_connected = True
    redis.store_reading    = AsyncMock(return_value=None)
    redis.get_recent_readings = AsyncMock(return_value=[])
    redis.cache_anomaly    = AsyncMock(return_value=None)
    redis.publish_anomaly  = AsyncMock(return_value=None)
    redis.get_history      = AsyncMock(return_value=[])
    return redis


@pytest.fixture
def mock_mongo():
    db = MagicMock()
    collection = MagicMock()
    collection.insert_one  = AsyncMock(return_value=MagicMock(inserted_id="abc"))
    collection.find_one    = AsyncMock(return_value=None)
    collection.replace_one = AsyncMock(return_value=None)
    collection.update_one  = AsyncMock(return_value=None)
    db.__getitem__ = MagicMock(return_value=collection)
    return db


@pytest.fixture
def mock_ml_service():
    svc = MagicMock()
    svc.is_ready = True
    return svc
