"""
PostgreSQL service for DefectSense — Supabase session pooler.

Provides read access to azure_sensor_readings table for ML training
and operational stats. Always gracefully degrades — never crashes
the application if the database is unavailable.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from loguru import logger


class PostgresService:
    """SQLAlchemy-based PostgreSQL client for azure_sensor_readings table."""

    def __init__(self, postgres_url: Optional[str]) -> None:
        self._url = postgres_url
        self._engine = None
        self._connected = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Create engine and test connection. Never raises."""
        if not self._url:
            logger.warning("PostgreSQL: POSTGRES_URL not set — skipping (set it in .env)")
            self._connected = False
            return

        try:
            from sqlalchemy import create_engine, text

            self._engine = create_engine(
                self._url,
                connect_args={"sslmode": "require"},
                pool_pre_ping=True,
            )

            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Log row count if table exists
            try:
                count = self.get_row_count()
                logger.info("PostgreSQL: connected ({:,} rows in azure_sensor_readings)", count)
            except Exception:
                logger.info("PostgreSQL: connected (azure_sensor_readings table not yet created)")

            self._connected = True

        except Exception as exc:
            logger.warning("PostgreSQL unavailable — training will fall back to CSV: {}", exc)
            self._engine = None
            self._connected = False

    def close(self) -> None:
        """Dispose engine cleanly."""
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception as exc:
                logger.warning("PostgreSQL: error closing engine: {}", exc)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Data Access ────────────────────────────────────────────────────────────

    def get_training_data(self, failure_only: bool = False) -> pd.DataFrame:
        """Pull all rows (or failures only) from azure_sensor_readings."""
        if not self._connected:
            return pd.DataFrame()
        try:
            from sqlalchemy import text

            query = "SELECT * FROM azure_sensor_readings"
            if failure_only:
                query += " WHERE machine_failure = 1"

            with self._engine.connect() as conn:
                return pd.read_sql(text(query), conn)
        except Exception as exc:
            logger.warning("PostgreSQL get_training_data failed: {}", exc)
            return pd.DataFrame()

    def get_normal_samples(self) -> pd.DataFrame:
        """Return pre-calculated clean normal rows (is_clean_normal = TRUE)."""
        if not self._connected:
            return pd.DataFrame()
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                return pd.read_sql(
                    text("SELECT * FROM azure_sensor_readings WHERE is_clean_normal = TRUE"),
                    conn,
                )
        except Exception as exc:
            logger.warning("PostgreSQL get_normal_samples failed: {}", exc)
            return pd.DataFrame()

    def get_failure_samples(self) -> pd.DataFrame:
        """Return rows where machine_failure = 1."""
        if not self._connected:
            return pd.DataFrame()
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                return pd.read_sql(
                    text("SELECT * FROM azure_sensor_readings WHERE machine_failure = 1"),
                    conn,
                )
        except Exception as exc:
            logger.warning("PostgreSQL get_failure_samples failed: {}", exc)
            return pd.DataFrame()

    def get_machine_stats(self) -> dict:
        """Return summary statistics over azure_sensor_readings."""
        if not self._connected:
            return {}
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                # Totals
                totals = pd.read_sql(
                    text("""
                        SELECT
                            COUNT(*)                          AS total_rows,
                            SUM(machine_failure)              AS failure_rows,
                            COUNT(*) - SUM(machine_failure)   AS normal_rows
                        FROM azure_sensor_readings
                    """),
                    conn,
                ).iloc[0]

                total    = int(totals["total_rows"])
                failures = int(totals["failure_rows"])
                normal   = int(totals["normal_rows"])

                # By machine model
                by_type_df = pd.read_sql(
                    text("""
                        SELECT machine_model,
                               COUNT(*) AS count,
                               SUM(machine_failure) AS failures
                        FROM azure_sensor_readings
                        GROUP BY machine_model
                    """),
                    conn,
                )
                by_machine_type = {
                    row["machine_model"]: {
                        "count":    int(row["count"]),
                        "failures": int(row["failures"]),
                    }
                    for _, row in by_type_df.iterrows()
                }

                # Sensor means — normal
                means_normal = pd.read_sql(
                    text("""
                        SELECT
                            AVG(volt)      AS volt,
                            AVG(rotate)    AS rotate,
                            AVG(pressure)  AS pressure,
                            AVG(vibration) AS vibration
                        FROM azure_sensor_readings
                        WHERE machine_failure = 0
                    """),
                    conn,
                ).iloc[0].to_dict()

                # Sensor means — failure
                means_failure = pd.read_sql(
                    text("""
                        SELECT
                            AVG(volt)      AS volt,
                            AVG(rotate)    AS rotate,
                            AVG(pressure)  AS pressure,
                            AVG(vibration) AS vibration
                        FROM azure_sensor_readings
                        WHERE machine_failure = 1
                    """),
                    conn,
                ).iloc[0].to_dict()

            return {
                "total_rows":           total,
                "failure_rows":         failures,
                "normal_rows":          normal,
                "failure_rate":         failures / total if total > 0 else 0.0,
                "by_machine_type":      by_machine_type,
                "sensor_means_normal":  {k: float(v) for k, v in means_normal.items()},
                "sensor_means_failure": {k: float(v) for k, v in means_failure.items()},
            }
        except Exception as exc:
            logger.warning("PostgreSQL get_machine_stats failed: {}", exc)
            return {}

    def get_row_count(self) -> int:
        """Return total row count of azure_sensor_readings, or 0 if unavailable."""
        if not self._connected or self._engine is None:
            return 0
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM azure_sensor_readings"))
                return int(result.scalar())
        except Exception as exc:
            logger.warning("PostgreSQL get_row_count failed: {}", exc)
            return 0
