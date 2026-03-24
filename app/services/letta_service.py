"""
Letta Service — Simplified stateful agent memory (inspired by letta/MemGPT).

Implements two memory tiers per machine:

  CORE MEMORY (always in LLM context)
  ├── machine_profile : machine ID, type, known operating ranges, failure history summary
  └── recent_patterns : last 5 anomaly observations (FIFO ring)

  ARCHIVAL MEMORY (searchable, not always in context)
  └── past root-cause reports, retrieved on demand via keyword search

Storage:
  - Core memory    → MongoDB "letta_core_memory" (one doc per machine)
  - Archival memory→ MongoDB "letta_archival" (append-only, indexed by machine_id)

If Letta (pip install letta) is later installed, this service can be swapped out
while keeping the same interface.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger


CORE_COLLECTION    = "letta_core_memory"
ARCHIVAL_COLLECTION = "letta_archival"

MAX_RECENT_PATTERNS = 5


class LettaService:
    """
    Stateful per-machine memory service.
    Inject a motor database at construction; call init() at startup.
    """

    def __init__(self, db=None) -> None:
        self._db    = db
        self._ready = False

    async def init(self) -> None:
        if self._db is not None:
            await self._ensure_indexes()
        self._ready = True
        logger.info("LettaService: ready")

    async def _ensure_indexes(self) -> None:
        try:
            await self._db[CORE_COLLECTION].create_index("machine_id", unique=True)
            await self._db[ARCHIVAL_COLLECTION].create_index(
                [("machine_id", 1), ("created_at", -1)]
            )
        except Exception:
            pass

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Core Memory ────────────────────────────────────────────────────────────

    async def get_core_memory(self, machine_id: str) -> str:
        """
        Return formatted core memory string to inject into the LLM context.
        Creates a default profile if the machine has no history yet.
        """
        doc = await self._load_core(machine_id)
        if doc is None:
            return self._default_profile(machine_id)

        profile  = doc.get("machine_profile", f"Machine {machine_id} — no profile yet.")
        patterns = doc.get("recent_patterns", [])

        lines = [
            f"=== Machine {machine_id} — Core Memory ===",
            "",
            "MACHINE PROFILE:",
            profile,
            "",
            "RECENT ANOMALY PATTERNS (last observations):",
        ]
        if patterns:
            for i, p in enumerate(reversed(patterns), 1):
                lines.append(f"  {i}. {p}")
        else:
            lines.append("  (none yet — first observation for this machine)")

        return "\n".join(lines)

    async def update_machine_profile(self, machine_id: str, profile_update: str) -> None:
        """
        Replace or set the machine_profile block.
        Called after a root-cause report is completed with a new summary.
        """
        await self._upsert_core(machine_id, {"machine_profile": profile_update})
        logger.debug("LettaService: updated profile for {}", machine_id)

    async def add_recent_pattern(self, machine_id: str, pattern: str) -> None:
        """
        Push a new observation to the recent_patterns ring (FIFO, max 5).
        Called after every anomaly detection + reasoning cycle.
        """
        doc = await self._load_core(machine_id)
        patterns = doc.get("recent_patterns", []) if doc else []
        patterns.append(pattern)
        if len(patterns) > MAX_RECENT_PATTERNS:
            patterns = patterns[-MAX_RECENT_PATTERNS:]
        await self._upsert_core(machine_id, {"recent_patterns": patterns})

    # ── Archival Memory ────────────────────────────────────────────────────────

    async def add_to_archival(self, machine_id: str, summary: str, metadata: dict | None = None) -> None:
        """
        Append a past root-cause report summary to archival memory.
        Called after every completed reasoning session.
        """
        if self._db is None:
            return
        try:
            doc = {
                "machine_id": machine_id,
                "summary":    summary,
                "metadata":   metadata or {},
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            await self._db[ARCHIVAL_COLLECTION].insert_one(doc)
        except Exception as exc:
            logger.warning("LettaService.add_to_archival failed: {}", exc)

    async def search_archival(
        self, machine_id: str, query: str, limit: int = 3
    ) -> list[str]:
        """
        Simple keyword/regex search over archival memory for a machine.
        Returns list of summary strings.
        (For production: replace with Qdrant semantic search.)
        """
        if self._db is None:
            return []
        try:
            import re
            # Build a loose OR regex from query words
            words = [w for w in re.split(r"\W+", query) if len(w) > 3]
            if not words:
                pattern = ".*"
            else:
                pattern = "|".join(re.escape(w) for w in words[:6])

            cursor = self._db[ARCHIVAL_COLLECTION].find(
                {
                    "machine_id": machine_id,
                    "summary": {"$regex": pattern, "$options": "i"},
                },
                {"_id": 0, "summary": 1},
            ).sort("created_at", -1).limit(limit)

            docs = await cursor.to_list(length=limit)
            return [d["summary"] for d in docs]
        except Exception as exc:
            logger.warning("LettaService.search_archival failed: {}", exc)
            return []

    async def get_recent_archival(self, machine_id: str, limit: int = 3) -> list[str]:
        """Return the most recent archival entries for a machine."""
        if self._db is None:
            return []
        try:
            cursor = self._db[ARCHIVAL_COLLECTION].find(
                {"machine_id": machine_id}, {"_id": 0, "summary": 1}
            ).sort("created_at", -1).limit(limit)
            docs = await cursor.to_list(length=limit)
            return [d["summary"] for d in docs]
        except Exception as exc:
            logger.warning("LettaService.get_recent_archival failed: {}", exc)
            return []

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _load_core(self, machine_id: str) -> Optional[dict]:
        if self._db is None:
            return None
        try:
            return await self._db[CORE_COLLECTION].find_one(
                {"machine_id": machine_id}, {"_id": 0}
            )
        except Exception:
            return None

    async def _upsert_core(self, machine_id: str, fields: dict) -> None:
        if self._db is None:
            return
        try:
            fields["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
            await self._db[CORE_COLLECTION].update_one(
                {"machine_id": machine_id},
                {"$set": fields},
                upsert=True,
            )
        except Exception as exc:
            logger.warning("LettaService._upsert_core failed: {}", exc)

    @staticmethod
    def _default_profile(machine_id: str) -> str:
        return (
            f"=== Machine {machine_id} — Core Memory ===\n\n"
            f"MACHINE PROFILE:\n"
            f"Machine {machine_id} — no historical profile yet. "
            f"This is the first recorded anomaly for this machine.\n\n"
            f"RECENT ANOMALY PATTERNS (last observations):\n"
            f"  (none yet — first observation for this machine)"
        )
