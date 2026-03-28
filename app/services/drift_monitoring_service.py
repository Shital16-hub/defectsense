"""
Drift Monitoring Service — detects sensor data distribution shift.

Uses Evidently AI to compare a reference distribution (normal PostgreSQL
samples or CSV fallback) against a current window of live Redis readings.

Drift report structure:
  is_drifted           — True if >50% of features have drifted
  drift_share          — fraction of features that drifted
  n_features_drifted   — integer count
  total_features       — always 5
  feature_details      — per-feature p-value and drift flag
  reference_size       — rows in reference dataset
  current_size         — rows in current window
  run_at               — ISO timestamp

Stored in MongoDB: drift_reports collection.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

# Evidently uses 0.5 as the default dataset-level drift threshold
# (drift declared if >50% of features have per-feature p-value < 0.05)
_DATASET_DRIFT_THRESHOLD = 0.5
_FEATURE_DRIFT_PVALUE    = 0.05


class DriftMonitoringService:
    """Compares live Redis readings against PostgreSQL / CSV reference data."""

    def __init__(self, mongo_db=None, postgres_url: Optional[str] = None) -> None:
        self._mongo         = mongo_db
        self._postgres_url  = postgres_url
        self._reference_data: Optional[pd.DataFrame] = None
        self._ready         = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Load reference data. Never raises."""
        try:
            await self.load_reference_data()
            self._ready = True
            n = len(self._reference_data) if self._reference_data is not None else 0
            logger.info("DriftMonitoringService: ready ({:,} reference samples)", n)
        except Exception as exc:
            logger.warning("DriftMonitoringService init failed (non-fatal): {}", exc)

    @property
    def is_ready(self) -> bool:
        return self._ready and self._reference_data is not None

    # ── Reference data ─────────────────────────────────────────────────────────

    async def load_reference_data(self) -> None:
        """Try PostgreSQL → fall back to CSV."""
        # ── PostgreSQL ─────────────────────────────────────────────────────────
        if self._postgres_url:
            try:
                from app.services.postgres_service import PostgresService
                pg = PostgresService(self._postgres_url)
                pg.init()
                if pg.is_connected:
                    df = pg.get_normal_samples()
                    pg.close()
                    if len(df) >= 100 and all(f in df.columns for f in FEATURES):
                        self._reference_data = df[FEATURES].dropna().reset_index(drop=True)
                        logger.info(
                            "DriftMonitor: loaded {:,} reference samples from PostgreSQL",
                            len(self._reference_data),
                        )
                        return
            except Exception as exc:
                logger.warning("DriftMonitor: PostgreSQL reference load failed — {}", exc)

        # ── CSV fallback ───────────────────────────────────────────────────────
        try:
            csv_path = Path(__file__).parent.parent.parent / "data" / "ai4i_2020.csv"
            if not csv_path.exists():
                logger.warning("DriftMonitor: CSV not found at {}", csv_path)
                self._reference_data = None
                return

            df = pd.read_csv(csv_path)
            col_map = {
                "Air temperature [K]":       "air_temperature",
                "Process temperature [K]":   "process_temperature",
                "Rotational speed [rpm]":    "rotational_speed",
                "Torque [Nm]":               "torque",
                "Tool wear [min]":           "tool_wear",
                "Machine failure":           "machine_failure",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            normal = df[df["machine_failure"] == 0][FEATURES].dropna().reset_index(drop=True)
            self._reference_data = normal
            logger.info(
                "DriftMonitor: loaded {:,} reference samples from CSV fallback",
                len(normal),
            )
        except Exception as exc:
            logger.warning("DriftMonitor: CSV fallback failed — {}", exc)
            self._reference_data = None

    # ── Drift report ───────────────────────────────────────────────────────────

    async def run_drift_report(self, current_data: pd.DataFrame) -> dict:
        """Run Evidently drift report and return structured result dict."""
        if self._reference_data is None:
            return {"error": "No reference data", "is_drifted": False}

        if len(current_data) < 10:
            return {
                "error": "Insufficient current data",
                "is_drifted": False,
                "current_size": len(current_data),
            }

        try:
            from evidently.presets import DataDriftPreset
            from evidently.core.report import Report

            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(
                reference_data=self._reference_data[FEATURES],
                current_data=current_data[FEATURES],
            )
            result_dict = snapshot.dict()

            metrics = result_dict.get("metrics", [])

            # metrics[0]: DriftedColumnsCount → {'count': float, 'share': float}
            # metrics[1..N]: ValueDrift per column → p-value float
            drift_count_metric = metrics[0] if metrics else {}
            drift_val = drift_count_metric.get("value", {})
            if not isinstance(drift_val, dict):
                drift_val = {}

            n_drifted   = int(drift_val.get("count", 0))
            drift_share = float(drift_val.get("share", 0.0))
            is_drifted  = drift_share >= _DATASET_DRIFT_THRESHOLD

            # Per-feature details from ValueDrift metrics (indices 1..N)
            feature_details: dict = {}
            for m in metrics[1:]:
                mn = m.get("metric_name", "")
                # Extract column name: 'ValueDrift(column=air_temperature,...)'
                col_match = re.search(r"column=([^,)]+)", mn)
                if col_match:
                    col = col_match.group(1).strip()
                    if col in FEATURES:
                        pval = float(m.get("value", 1.0))
                        feature_details[col] = {
                            "p_value": pval,
                            "drifted": pval < _FEATURE_DRIFT_PVALUE,
                        }

            report_doc = {
                "run_at":             datetime.now(tz=timezone.utc).isoformat(),
                "is_drifted":         is_drifted,
                "drift_share":        drift_share,
                "n_features_drifted": n_drifted,
                "total_features":     len(FEATURES),
                "reference_size":     len(self._reference_data),
                "current_size":       len(current_data),
                "feature_details":    feature_details,
            }

            if is_drifted:
                logger.warning(
                    "DRIFT DETECTED: {:.0%} features drifted — retraining recommended",
                    drift_share,
                )
            else:
                logger.info(
                    "Drift check: no drift detected (drift_share={:.0%})", drift_share
                )

            # ── Save to MongoDB ────────────────────────────────────────────────
            if self._mongo is not None:
                try:
                    await self._mongo["drift_reports"].insert_one({**report_doc})
                except Exception as exc:
                    logger.warning("DriftMonitor: MongoDB save failed — {}", exc)

            return report_doc

        except Exception as exc:
            logger.warning("DriftMonitor: report run failed — {}", exc)
            return {"error": str(exc), "is_drifted": False}

    # ── Redis window ───────────────────────────────────────────────────────────

    async def get_current_window_data(
        self,
        redis_service,
        machine_ids: list[str],
        n_per_machine: int = 100,
    ) -> pd.DataFrame:
        """Fetch recent readings from Redis and return as a feature DataFrame."""
        rows: list[dict] = []
        for mid in machine_ids:
            try:
                readings = await redis_service.get_recent_readings(mid, n=n_per_machine)
                for r in readings:
                    row = {f: getattr(r, f, None) for f in FEATURES}
                    if all(v is not None for v in row.values()):
                        rows.append(row)
            except Exception as exc:
                logger.warning("DriftMonitor: Redis read for {} failed — {}", mid, exc)

        if not rows:
            return pd.DataFrame(columns=FEATURES)

        return pd.DataFrame(rows)[FEATURES].dropna().reset_index(drop=True)

    # ── Full orchestration ─────────────────────────────────────────────────────

    async def run_full_drift_check(self, redis_service) -> dict:
        """
        1. Discover machine IDs from Redis keys.
        2. Fetch current window data.
        3. Run drift report.
        """
        try:
            # Discover machine IDs
            machine_ids: list[str] = []
            try:
                keys = await redis_service._client.keys("sensor:*:readings")
                for k in keys:
                    raw = k.decode() if isinstance(k, bytes) else k
                    parts = raw.split(":")
                    if len(parts) >= 2:
                        machine_ids.append(parts[1])
            except Exception as exc:
                logger.warning("DriftMonitor: Redis key discovery failed — {}", exc)

            if not machine_ids:
                return {"error": "No machines in Redis", "is_drifted": False}

            # Get current window
            df = await self.get_current_window_data(redis_service, machine_ids)

            if len(df) < 10:
                return {
                    "error": "Insufficient data",
                    "is_drifted": False,
                    "n_readings": len(df),
                }

            return await self.run_drift_report(df)

        except Exception as exc:
            logger.warning("DriftMonitor: full drift check failed — {}", exc)
            return {"error": str(exc), "is_drifted": False}
