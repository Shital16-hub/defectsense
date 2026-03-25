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
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1500.0,
        torque=40.0,
        tool_wear=50.0,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def failure_reading() -> SensorReading:
    """HDF-like: low temp delta + low RPM + high torque."""
    return SensorReading(
        machine_id="TEST_M001",
        air_temperature=302.0,
        process_temperature=309.0,
        rotational_speed=1182.0,
        torque=68.5,
        tool_wear=195.0,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def twf_reading() -> SensorReading:
    """Tool wear failure: very high tool_wear."""
    return SensorReading(
        machine_id="TEST_M002",
        air_temperature=298.0,
        process_temperature=308.5,
        rotational_speed=1400.0,
        torque=50.0,
        tool_wear=245.0,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def normal_reading_sequence(normal_reading) -> list[SensorReading]:
    """30 slightly-varied normal readings for LSTM sequence."""
    return [
        SensorReading(
            machine_id="TEST_M001",
            air_temperature=298.0 + i * 0.02,
            process_temperature=308.5 + i * 0.02,
            rotational_speed=1498.0 + i * 0.1,
            torque=40.0 + i * 0.05,
            tool_wear=float(i),
            timestamp=datetime.utcnow(),
        )
        for i in range(30)
    ]


@pytest.fixture
def sample_anomaly_result() -> AnomalyResult:
    return AnomalyResult(
        machine_id="TEST_M001",
        anomaly_score=0.82,
        failure_probability=0.75,
        is_anomaly=True,
        failure_type_prediction="HDF",
        sensor_deltas={
            "air_temperature": 0.3,
            "process_temperature": 3.1,
            "rotational_speed": -2.2,
            "torque": 2.8,
            "tool_wear": 1.1,
        },
        ml_model_used="ensemble",
        reconstruction_error=0.045,
        isolation_score=-0.12,
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
        reasoning_steps=["THINK: high temp delta detected", "CONCLUDE: HDF pattern"],
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
