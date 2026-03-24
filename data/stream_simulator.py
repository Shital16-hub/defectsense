"""
Stream Simulator — reads AI4I 2020 CSV row-by-row and POSTs each row to
POST /api/sensors/ingest every 0.5 seconds, simulating a live sensor feed.

Features:
  - Scans the dataset upfront and logs how many known failure rows it found
  - Logs a warning each time a failure row is sent (so you can watch anomalies
    appear in the server terminal)
  - Loops over the dataset once (set LOOP=True to repeat)
  - Configurable via CLI args or env vars

Run (server must be up first):
    python data/stream_simulator.py
    python data/stream_simulator.py --url http://localhost:8080 --interval 0.2
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from loguru import logger

ROOT     = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "ai4i_2020.csv"

COL_MAP = {
    "Air temperature [K]":      "air_temperature",
    "Process temperature [K]":  "process_temperature",
    "Rotational speed [rpm]":   "rotational_speed",
    "Torque [Nm]":              "torque",
    "Tool wear [min]":          "tool_wear",
    "Machine failure":          "machine_failure",
    "TWF":                      "twf",
    "HDF":                      "hdf",
    "PWF":                      "pwf",
    "OSF":                      "osf",
    "RNF":                      "rnf",
}

FAILURE_COLS = {"twf", "hdf", "pwf", "osf", "rnf"}


def load_rows() -> list[dict]:
    """Load and normalise all CSV rows. Returns list of dicts."""
    if not CSV_PATH.exists():
        logger.error("Dataset not found at {}. Run data/download_data.py first.", CSV_PATH)
        sys.exit(1)

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            normalised = {COL_MAP.get(k, k.lower()): v for k, v in row.items()}
            normalised["_row_idx"] = i
            rows.append(normalised)

    failure_rows = [r for r in rows if int(r.get("machine_failure", 0)) == 1]
    logger.info(
        "Loaded {:,} rows — {:,} failure rows ({:.1f}%)",
        len(rows),
        len(failure_rows),
        100 * len(failure_rows) / len(rows),
    )

    # Log which failure types are present
    for col in FAILURE_COLS:
        count = sum(1 for r in rows if int(r.get(col, 0)) == 1)
        if count:
            logger.info("  {:>4}  {} failures", count, col.upper())

    return rows


def row_to_payload(row: dict, machine_id_prefix: str = "M") -> dict:
    """Convert a CSV row dict to the SensorReading JSON payload."""
    udi = row.get("udi", row.get("_row_idx", 0))
    machine_id = f"{machine_id_prefix}{int(udi):04d}"
    return {
        "machine_id":          machine_id,
        "timestamp":           datetime.now(tz=timezone.utc).isoformat(),
        "air_temperature":     float(row["air_temperature"]),
        "process_temperature": float(row["process_temperature"]),
        "rotational_speed":    float(row["rotational_speed"]),
        "torque":              float(row["torque"]),
        "tool_wear":           float(row["tool_wear"]),
        "source":              "ai4i",
    }


def is_failure_row(row: dict) -> str:
    """Return the failure type string if this row is a failure, else empty string."""
    if int(row.get("machine_failure", 0)) != 1:
        return ""
    for col in FAILURE_COLS:
        if int(row.get(col, 0)) == 1:
            return col.upper()
    return "UNKNOWN"


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

                payload     = row_to_payload(row)
                failure_tag = is_failure_row(row)

                if failure_tag:
                    logger.warning(
                        ">>> SENDING FAILURE ROW | {} | type={} | torque={} Nm | wear={} min",
                        payload["machine_id"],
                        failure_tag,
                        payload["torque"],
                        payload["tool_wear"],
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
                    elif sent % 50 == 0:
                        logger.info(
                            "  sent={:,} | anomalies={:,} | errors={:,}",
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
    p = argparse.ArgumentParser(description="DefectSense stream simulator")
    p.add_argument("--url",      default="http://localhost:8080", help="Base URL of the API server")
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between rows (default 0.5)")
    p.add_argument("--loop",     action="store_true", help="Loop over dataset indefinitely")
    p.add_argument("--max-rows", type=int,  default=0,   help="Stop after N rows (0 = no limit)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    rows   = load_rows()

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
