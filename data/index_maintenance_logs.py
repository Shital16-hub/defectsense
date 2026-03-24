"""
Index maintenance logs into Qdrant 'maintenance_logs' collection.

Reads data/maintenance_logs.csv, converts each row to MaintenanceLog,
embeds with sentence-transformers/all-MiniLM-L6-v2, and upserts to Qdrant.

Run:
    python data/index_maintenance_logs.py
    python data/index_maintenance_logs.py --qdrant-url https://your-cluster.qdrant.io --api-key KEY
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "maintenance_logs.csv"


def load_logs() -> list:
    """Read CSV and return list of MaintenanceLog objects."""
    from app.models.maintenance import MaintenanceLog

    if not DATA_PATH.exists():
        logger.error("maintenance_logs.csv not found — run data/generate_logs.py first")
        sys.exit(1)

    logs = []
    skipped = 0
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                log = MaintenanceLog(
                    log_id=row["log_id"],
                    machine_id=row["machine_id"],
                    machine_type=row.get("machine_type") or None,
                    date=row["date"],
                    failure_type=row["failure_type"],
                    symptoms=row["symptoms"],
                    root_cause=row["root_cause"],
                    action_taken=row["action_taken"],
                    resolution_time_hours=float(row["resolution_time_hours"]),
                    technician=row["technician"],
                )
                logs.append(log)
            except Exception as exc:
                skipped += 1
                logger.warning("Row {}: skipped — {}", i, exc)

    logger.info("Loaded {:,} logs ({} skipped)", len(logs), skipped)
    return logs


async def main(qdrant_url: str, api_key: str | None) -> None:
    from app.services.qdrant_service import QdrantService

    logger.info("=" * 55)
    logger.info("  DefectSense — Indexing Maintenance Logs")
    logger.info("=" * 55)
    logger.info("  Qdrant URL : {}", qdrant_url)
    logger.info("  CSV path   : {}", DATA_PATH)
    logger.info("")

    # Load logs from CSV
    logs = load_logs()
    if not logs:
        logger.error("No logs to index — exiting")
        return

    # Init Qdrant service (connects + loads embedding model)
    qdrant = QdrantService(url=qdrant_url, api_key=api_key)
    await qdrant.init()

    # Upsert all logs
    logger.info("Embedding and upserting {:,} logs...", len(logs))
    count = await qdrant.upsert_logs(logs)

    # Verify
    total = await qdrant.collection_count("maintenance_logs")

    logger.info("")
    logger.info("=" * 55)
    logger.info("  Indexed {:,} maintenance logs", count)
    logger.info("  Collection total: {:,} vectors", total)
    logger.info("=" * 55)

    # Quick sanity search
    logger.info("\nSanity search: 'cooling system overheating HDF'")
    results = await qdrant.search_similar_incidents(
        "cooling system overheating high temperature", failure_type="HDF", limit=2
    )
    for i, r in enumerate(results, 1):
        logger.info("  {}. [{}] {} — {}", i, r.failure_type, r.machine_id, r.symptoms[:80])

    logger.info("\nDone. Run the API server to use RAG retrieval.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index maintenance logs into Qdrant")
    p.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (default: QDRANT_URL env or http://localhost:6333)",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("QDRANT_API_KEY") or None,
        help="Qdrant API key (optional, for Qdrant Cloud)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(qdrant_url=args.qdrant_url, api_key=args.api_key))
