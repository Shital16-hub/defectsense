"""
Sensor Ingestion Service — streams AI4I 2020 dataset rows via Redis pub/sub,
simulating a real-time manufacturing sensor feed.

Architecture:
    CSVStreamer (producer) ──► Redis channel `defectsense:sensor_readings`
                                   ▲
    SensorIngestionService (consumer) reads from that channel and hands each
    SensorReading to the AnomalyDetectorAgent.

Usage (standalone test):
    python -m app.services.sensor_ingestion

Usage (programmatic):
    from app.services.sensor_ingestion import SensorIngestionService
    svc = SensorIngestionService(redis_client)
    await svc.start_streaming()   # starts producer + consumer loops
"""
from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import redis.asyncio as aioredis
from loguru import logger

from app.models.sensor import SensorReading

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "data" / "ai4i_2020.csv"

# ── Redis channel names ────────────────────────────────────────────────────────
CHANNEL_SENSOR_READINGS = "defectsense:sensor_readings"
CHANNEL_ANOMALY_RESULTS = "defectsense:anomaly_results"

# ── Column name mapping (original AI4I CSV headers → our Pydantic field names) ─
COL_MAP = {
    "Air temperature [K]":      "air_temperature",
    "Process temperature [K]":  "process_temperature",
    "Rotational speed [rpm]":   "rotational_speed",
    "Torque [Nm]":              "torque",
    "Tool wear [min]":          "tool_wear",
    "Machine failure":          "machine_failure",
}


# ── CSV Streamer (Producer) ────────────────────────────────────────────────────

class CSVStreamer:
    """
    Reads AI4I 2020 CSV row-by-row and publishes each row as a JSON-serialised
    SensorReading to Redis.  Simulates sensor telemetry at `rows_per_second` Hz.
    Loops over the dataset if `loop_forever=True`.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        csv_path: Path = CSV_PATH,
        rows_per_second: float = 2.0,
        loop_forever: bool = True,
        machine_id_prefix: str = "M",
    ) -> None:
        self._redis            = redis_client
        self._csv_path         = csv_path
        self._interval         = 1.0 / rows_per_second
        self._loop_forever     = loop_forever
        self._machine_id_prefix = machine_id_prefix
        self._running          = False
        self._rows_sent        = 0

    async def start(self) -> None:
        """Begin streaming. Runs until cancelled or loop_forever=False and EOF."""
        if not self._csv_path.exists():
            logger.error("CSV not found: {}. Run data/download_data.py first.", self._csv_path)
            return

        self._running = True
        logger.info(
            "CSVStreamer: starting — {} rows/sec, loop={}",
            1.0 / self._interval,
            self._loop_forever,
        )

        while self._running:
            async for reading in self._iter_csv():
                if not self._running:
                    break
                payload = reading.model_dump_json()
                await self._redis.publish(CHANNEL_SENSOR_READINGS, payload)
                self._rows_sent += 1
                if self._rows_sent % 100 == 0:
                    logger.info("CSVStreamer: {} rows published", self._rows_sent)
                await asyncio.sleep(self._interval)

            if not self._loop_forever:
                break
            logger.info("CSVStreamer: dataset loop complete — restarting from row 0")

        logger.info("CSVStreamer: stopped after {} rows", self._rows_sent)

    def stop(self) -> None:
        self._running = False

    async def _iter_csv(self) -> AsyncIterator[SensorReading]:
        """Yield SensorReading objects from the CSV one row at a time."""
        with open(self._csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                # Normalise column names
                normalised = {COL_MAP.get(k, k): v for k, v in row.items()}

                # Build machine_id from UDI or row index
                udi = normalised.get("UDI", str(row_idx + 1))
                machine_id = f"{self._machine_id_prefix}{int(udi):04d}"

                try:
                    reading = SensorReading(
                        machine_id=machine_id,
                        timestamp=datetime.now(tz=timezone.utc),
                        air_temperature=float(normalised["air_temperature"]),
                        process_temperature=float(normalised["process_temperature"]),
                        rotational_speed=float(normalised["rotational_speed"]),
                        torque=float(normalised["torque"]),
                        tool_wear=float(normalised["tool_wear"]),
                        source="ai4i",
                    )
                    yield reading
                except (KeyError, ValueError) as exc:
                    logger.warning("CSVStreamer: skipping row {} — {}", row_idx, exc)
                    continue


# ── Sensor Ingestion Service (Consumer) ───────────────────────────────────────

class SensorIngestionService:
    """
    Subscribes to `defectsense:sensor_readings` Redis channel and dispatches
    each SensorReading to the AnomalyDetectorAgent.

    Designed to be started once at application startup alongside the FastAPI server.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        anomaly_detector=None,
        channel: str = CHANNEL_SENSOR_READINGS,
    ) -> None:
        self._redis     = redis_client
        self._detector  = anomaly_detector
        self._channel   = channel
        self._running   = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start consuming sensor readings in the background."""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("SensorIngestionService: listening on channel '{}'", self._channel)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SensorIngestionService: stopped")

    async def _consume_loop(self) -> None:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel)
        logger.info("SensorIngestionService: subscribed to {}", self._channel)

        try:
            async for message in pubsub.listen():
                if not self._running:
                    break
                if message["type"] != "message":
                    continue
                await self._handle_message(message["data"])
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("SensorIngestionService: consumer error — {}", exc)
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.close()

    async def _handle_message(self, data: bytes | str) -> None:
        try:
            payload = data if isinstance(data, str) else data.decode("utf-8")
            reading = SensorReading(**json.loads(payload))
        except Exception as exc:
            logger.warning("SensorIngestionService: failed to parse message — {}", exc)
            return

        if self._detector is not None:
            try:
                await self._detector.detect(reading)
            except Exception as exc:
                logger.error(
                    "SensorIngestionService: detector error for {} — {}",
                    reading.machine_id,
                    exc,
                )
        else:
            logger.debug(
                "SensorIngestionService: received reading for {} (no detector attached)",
                reading.machine_id,
            )


# ── Redis connection factory ───────────────────────────────────────────────────

def create_redis_client(url: str = "redis://localhost:6379/0") -> aioredis.Redis:
    return aioredis.from_url(url, decode_responses=False)


# ── Standalone test entrypoint ─────────────────────────────────────────────────

async def _standalone_demo() -> None:
    """
    Quick smoke test: stream 20 rows from CSV → Redis → consume + print.
    Run with:  python -m app.services.sensor_ingestion
    """
    redis = create_redis_client()

    received: list[SensorReading] = []

    class _PrintDetector:
        async def detect(self, r: SensorReading):
            received.append(r)
            print(
                f"  [{len(received):03d}] {r.machine_id} | "
                f"T_air={r.air_temperature:.1f}K | "
                f"torque={r.torque:.1f}Nm | "
                f"wear={r.tool_wear:.0f}min"
            )
            if len(received) >= 20:
                raise asyncio.CancelledError

    consumer = SensorIngestionService(redis, _PrintDetector())
    await consumer.start()

    producer = CSVStreamer(redis, rows_per_second=10.0, loop_forever=False)

    print("=== DefectSense — Sensor Ingestion Demo (20 rows) ===")
    try:
        await asyncio.gather(
            producer.start(),
            asyncio.sleep(10),
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await consumer.stop()
        await redis.aclose()
    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(_standalone_demo())
