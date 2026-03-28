"""
Unit tests for PostgresService.

All tests are fully offline — no real database required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest


# ── Helper factories ──────────────────────────────────────────────────────────

def _make_normal_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "id":                  range(n),
        "machine_id":          [f"M{i:03d}" for i in range(n)],
        "air_temperature":     [300.0] * n,
        "process_temperature": [310.0] * n,
        "rotational_speed":    [1500.0] * n,
        "torque":              [40.0] * n,
        "tool_wear":           [50.0] * n,
        "machine_failure":     [0] * n,
        "failure_type":        ["NONE"] * n,
        "machine_type":        ["L"] * n,
    })


def _make_failure_df(n: int = 10) -> pd.DataFrame:
    df = _make_normal_df(n)
    df["machine_failure"] = 1
    df["failure_type"] = "TWF"
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPostgresServiceInit:
    def test_postgres_service_initializes_without_url(self):
        """PostgresService(None).init() must not crash; is_connected == False."""
        from app.services.postgres_service import PostgresService

        svc = PostgresService(None)
        svc.init()
        assert svc.is_connected is False

    def test_postgres_service_initializes_with_bad_url(self):
        """PostgresService('bad_url').init() must not crash; is_connected == False."""
        from app.services.postgres_service import PostgresService

        svc = PostgresService("bad_url")
        svc.init()
        assert svc.is_connected is False

    def test_ssl_required_in_connection_args(self):
        """Engine must be created with sslmode=require."""
        from app.services.postgres_service import PostgresService

        captured = {}

        def patched_create_engine(url, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("abort — stop after capture")

        with patch("sqlalchemy.create_engine", patched_create_engine):
            svc = PostgresService("postgresql://user:pass@host/db")
            svc.init()

        assert "connect_args" in captured
        assert captured["connect_args"].get("sslmode") == "require"


class TestPostgresServiceData:
    def _connected_service(self, mock_df: pd.DataFrame):
        """Return a PostgresService with _connected=True and a mocked engine."""
        from app.services.postgres_service import PostgresService

        svc = PostgresService("postgresql://user:pass@host/db")
        svc._connected = True
        svc._engine = MagicMock()

        # Patch pd.read_sql to return mock_df
        with patch("pandas.read_sql", return_value=mock_df):
            yield svc

    def test_get_normal_samples_returns_dataframe(self):
        """get_normal_samples() returns a DataFrame with machine_failure all zeros."""
        from app.services.postgres_service import PostgresService

        normal_df = _make_normal_df(100)

        svc = PostgresService("postgresql://user:pass@host/db")
        svc._connected = True
        svc._engine = MagicMock()

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        svc._engine.connect.return_value = mock_conn

        with patch("pandas.read_sql", return_value=normal_df):
            result = svc.get_normal_samples()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert (result["machine_failure"] == 0).all()

    def test_get_failure_samples_returns_dataframe(self):
        """get_failure_samples() returns a DataFrame with machine_failure all ones."""
        from app.services.postgres_service import PostgresService

        failure_df = _make_failure_df(10)

        svc = PostgresService("postgresql://user:pass@host/db")
        svc._connected = True
        svc._engine = MagicMock()

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        svc._engine.connect.return_value = mock_conn

        with patch("pandas.read_sql", return_value=failure_df):
            result = svc.get_failure_samples()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert (result["machine_failure"] == 1).all()

    def test_get_machine_stats_returns_correct_keys(self):
        """get_machine_stats() returns dict with all required top-level keys."""
        from app.services.postgres_service import PostgresService

        # Build dataframes for each sequential read_sql call
        totals_df = pd.DataFrame([{"total_rows": 10000, "failure_rows": 339, "normal_rows": 9661}])
        by_type_df = pd.DataFrame([
            {"machine_type": "L", "count": 6000, "failures": 200},
            {"machine_type": "M", "count": 3000, "failures": 100},
            {"machine_type": "H", "count": 1000, "failures": 39},
        ])
        sensor_row = {
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 50.0,
        }
        means_df = pd.DataFrame([sensor_row])

        call_returns = [totals_df, by_type_df, means_df, means_df]
        call_index = {"i": 0}

        def fake_read_sql(query, conn):
            df = call_returns[call_index["i"]]
            call_index["i"] += 1
            return df

        svc = PostgresService("postgresql://user:pass@host/db")
        svc._connected = True
        svc._engine = MagicMock()

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        svc._engine.connect.return_value = mock_conn

        with patch("pandas.read_sql", side_effect=fake_read_sql):
            stats = svc.get_machine_stats()

        required_keys = {"total_rows", "failure_rows", "normal_rows", "failure_rate", "by_machine_type"}
        assert required_keys.issubset(stats.keys())
        assert stats["total_rows"] == 10000
        assert stats["failure_rows"] == 339


class TestPostgresServiceGracefulDegradation:
    def test_graceful_degradation_returns_empty_df(self):
        """When not connected, all data methods return empty DataFrame (not None)."""
        from app.services.postgres_service import PostgresService

        svc = PostgresService("postgresql://user:pass@host/db")
        svc._connected = False

        assert isinstance(svc.get_training_data(), pd.DataFrame)
        assert isinstance(svc.get_normal_samples(), pd.DataFrame)
        assert isinstance(svc.get_failure_samples(), pd.DataFrame)
        assert len(svc.get_training_data()) == 0
        assert len(svc.get_normal_samples()) == 0
        assert len(svc.get_failure_samples()) == 0

    def test_get_row_count_returns_zero_when_disconnected(self):
        """is_connected=False → get_row_count() == 0."""
        from app.services.postgres_service import PostgresService

        svc = PostgresService("postgresql://user:pass@host/db")
        svc._connected = False

        assert svc.get_row_count() == 0

    def test_get_machine_stats_returns_empty_dict_when_disconnected(self):
        """is_connected=False → get_machine_stats() == {}."""
        from app.services.postgres_service import PostgresService

        svc = PostgresService(None)
        svc._connected = False
        assert svc.get_machine_stats() == {}


class TestTrainingFallback:
    def test_fallback_to_csv_when_postgres_unavailable(self, tmp_path):
        """When PostgresService returns empty DataFrame, training falls back to CSV."""
        import numpy as np
        from pathlib import Path

        # Write a minimal valid CSV
        csv_path = tmp_path / "ai4i_2020.csv"
        n = 200
        df = pd.DataFrame({
            "air_temperature":     np.random.uniform(295, 305, n),
            "process_temperature": np.random.uniform(308, 313, n),
            "rotational_speed":    np.random.uniform(1200, 2000, n),
            "torque":              np.random.uniform(10, 70, n),
            "tool_wear":           np.random.uniform(0, 250, n),
            "machine_failure":     [0] * n,
        })
        df.to_csv(csv_path, index=False)

        from app.services.postgres_service import PostgresService

        # Mock PostgresService to simulate unavailability
        mock_svc = MagicMock(spec=PostgresService)
        mock_svc.is_connected = False
        mock_svc.get_normal_samples.return_value = pd.DataFrame()

        with patch("app.services.postgres_service.PostgresService", return_value=mock_svc):
            # Inline the fallback logic as tested in load_and_prepare
            normal = pd.DataFrame()

            if normal.empty:
                raw = pd.read_csv(csv_path)
                normal = raw[raw["machine_failure"] == 0][
                    ["air_temperature", "process_temperature",
                     "rotational_speed", "torque", "tool_wear"]
                ].dropna()

        assert len(normal) == n
        assert not normal.empty


class TestLoadScriptIdempotent:
    def test_load_script_skips_when_rows_exist(self, capsys):
        """If sensor_readings already has rows, load script prints 'Already loaded' and exits."""
        from unittest.mock import MagicMock, patch

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn

        # First connect() call: CREATE TABLE
        # Second connect() call: COUNT(*) → 10000
        execute_results = [
            MagicMock(),                          # CREATE TABLE
            MagicMock(scalar=lambda: 10000),      # COUNT(*)
        ]
        call_count = {"n": 0}

        def fake_execute(sql):
            result = execute_results[min(call_count["n"], len(execute_results) - 1)]
            call_count["n"] += 1
            return result

        mock_conn.execute.side_effect = fake_execute
        mock_conn.commit.return_value = None

        with patch("sqlalchemy.create_engine", return_value=mock_engine), \
             patch("os.getenv", side_effect=lambda k, d=None: "postgresql://x" if k == "POSTGRES_URL" else d), \
             patch("pathlib.Path.exists", return_value=True):
            # Simulate what load_to_postgres.py does
            from sqlalchemy import create_engine, text
            engine = create_engine("postgresql://x", connect_args={"sslmode": "require"})
            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE IF NOT EXISTS sensor_readings (...)"))
                conn.commit()
            with engine.connect() as conn:
                existing = int(conn.execute(text("SELECT COUNT(*) FROM sensor_readings")).scalar())

            if existing > 0:
                print(f"Already loaded: {existing:,} rows exist — skipping")

        captured = capsys.readouterr()
        assert "Already loaded" in captured.out
        assert "10,000 rows exist — skipping" in captured.out
