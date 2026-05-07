"""
Stream Simulator — reads Azure PdM telemetry CSV row-by-row and POSTs each row to
POST /api/sensors/ingest, simulating a live sensor feed.

Features:
  - Loads PdM_telemetry.csv and PdM_failures.csv, joins on machineID + datetime
  - Sends rows in chronological order (sorted by datetime then machineID)
  - Logs a WARNING each time a failure row is sent
  - Loops over the dataset once (use --loop to repeat)
  - Configurable via CLI args

Run (server must be up first):
    python data/stream_simulator.py
    python data/stream_simulator.py --url http://localhost:8080 --interval 0.2
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

ROOT            = Path(__file__).parent.parent
TELEMETRY_PATH  = ROOT / "data" / "azure_pdm" / "PdM_telemetry.csv"
FAILURES_PATH   = ROOT / "data" / "azure_pdm" / "PdM_failures.csv"

FAILURE_COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]


def load_rows() -> list[dict]:
    """Load telemetry + failures, join, sort, return list of row dicts."""
    for path in (TELEMETRY_PATH, FAILURES_PATH):
        if not path.exists():
            logger.error("Required file not found: {}. Run data/download_data.py first.", path)
            sys.exit(1)

    telemetry = pd.read_csv(TELEMETRY_PATH, parse_dates=["datetime"])
    failures  = pd.read_csv(FAILURES_PATH,  parse_dates=["datetime"])

    # Join failures onto telemetry — failure column is NaN for normal rows
    merged = telemetry.merge(
        failures.rename(columns={"failure": "_failure_type"}),
        on=["datetime", "machineID"],
        how="left",
    )

    # Sort chronologically, then by machine for deterministic order within a timestamp
    merged = merged.sort_values(["datetime", "machineID"]).reset_index(drop=True)

    total         = len(merged)
    failure_rows  = merged[merged["_failure_type"].notna()]
    failure_rate  = 100.0 * len(failure_rows) / total if total else 0.0

    # ── Startup summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  DefectSense Stream Simulator — Azure PdM Dataset")
    print("=" * 60)
    print(f"  Total rows loaded : {total:,}")
    print(f"  Failure rows      : {len(failure_rows):,}  ({failure_rate:.2f}%)")
    print()
    print("  Failure breakdown by component:")
    for comp in FAILURE_COMPONENTS:
        count = int((failure_rows["_failure_type"] == comp).sum())
        if count:
            print(f"    {comp:<6}: {count:,}")
    print("=" * 60)
    print()

    return merged.to_dict("records")


def machine_id_str(machine_id_int: int) -> str:
    """Convert integer machineID → 'M001' format."""
    return f"M{int(machine_id_int):03d}"


def row_to_payload(row: dict) -> dict:
    """Convert a merged row dict to the SensorReading JSON payload."""
    return {
        "machine_id": machine_id_str(row["machineID"]),
        "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
        "volt":       float(row["volt"]),
        "rotate":     float(row["rotate"]),
        "pressure":   float(row["pressure"]),
        "vibration":  float(row["vibration"]),
        "source":     "azure_pdm",
    }


async def stream(
    rows: list[dict],
    base_url: str,
    interval: float,
    loop: bool,
    max_rows: int,
) -> None:
    endpoint = f"{base_url.rstrip('/')}/api/sensors/ingest"
    logger.info("Streaming to {} at {:.1f} rows/sec", endpoint, 1.0 / interval)
    logger.info("Press Ctrl+C to stop.\n")

    sent      = 0
    anomalies = 0
    errors    = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        iterations = 0
        while True:
            iterations += 1
            for row in rows:
                if max_rows and sent >= max_rows:
                    logger.info("Reached max_rows={}, stopping.", max_rows)
                    return

                payload      = row_to_payload(row)
                failure_type = row.get("_failure_type")

                if pd.notna(failure_type) if failure_type is not None else False:
                    logger.warning(
                        ">>> FAILURE ROW | {} | component={} | volt={:.2f} | rotate={:.2f}"
                        " | pressure={:.2f} | vibration={:.2f}",
                        payload["machine_id"],
                        failure_type,
                        payload["volt"],
                        payload["rotate"],
                        payload["pressure"],
                        payload["vibration"],
                    )

                try:
                    resp = await client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    result = resp.json()
                    sent += 1

                    if result.get("is_anomaly"):
                        anomalies += 1
                        logger.warning(
                            "  ✓ ANOMALY DETECTED | score={:.3f} | prob={:.3f} | type={}",
                            result.get("anomaly_score", 0),
                            result.get("failure_probability", 0),
                            result.get("failure_type_prediction", "?"),
                        )

                    if sent % 100 == 0:
                        logger.info(
                            "  rows sent={:,} | anomalies detected={:,} | errors={:,}",
                            sent, anomalies, errors,
                        )

                except httpx.HTTPStatusError as exc:
                    errors += 1
                    logger.error("HTTP {}: {}", exc.response.status_code, exc.response.text[:200])
                except Exception as exc:
                    errors += 1
                    logger.error("Request failed: {}", exc)
                    await asyncio.sleep(2.0)  # back off on connection error

                await asyncio.sleep(interval)

            if not loop:
                break
            logger.info("Dataset loop {} complete — restarting", iterations)

    logger.info("\n=== Stream complete ===")
    logger.info("  Rows sent  : {:,}", sent)
    logger.info("  Anomalies  : {:,}", anomalies)
    logger.info("  Errors     : {:,}", errors)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DefectSense stream simulator — Azure PdM")
    p.add_argument("--url",      default="http://localhost:8080", help="Base URL of the API server")
    p.add_argument("--interval", type=float, default=0.5,  help="Seconds between rows (default 0.5)")
    p.add_argument("--loop",     action="store_true",       help="Loop over dataset indefinitely")
    p.add_argument("--max-rows", type=int,   default=0,    help="Stop after N rows (0 = no limit)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = load_rows()

    try:
        asyncio.run(
            stream(
                rows,
                base_url=args.url,
                interval=args.interval,
                loop=args.loop,
                max_rows=args.max_rows,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
