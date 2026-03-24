"""
A-MEM Service — Simplified Agentic Memory (Zettelkasten-inspired).

Implements the core ideas from A-MEM (agiresearch/A-MEM):
  - Memory notes with content, keywords, and inter-note links
  - Semantic search using cosine similarity over stored embeddings
  - Auto-linking: when a note is added, similar existing notes are auto-linked
  - Linked recall: searching returns hit notes + their linked neighbours
  - Notes evolve: update_memory() appends new observations and re-embeds

Storage: MongoDB collection "agent_memory"
Embedding: sentence-transformers/all-MiniLM-L6-v2 (same model as QdrantService)

The agent reads memory BEFORE reasoning → is primed with past patterns.
The agent updates memory AFTER reasoning → system gets smarter over time.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger


COLLECTION = "agent_memory"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AUTO_LINK_THRESHOLD = 0.75   # cosine similarity above which notes are auto-linked
AUTO_LINK_TOP_K     = 3      # max auto-links per new note


class MemoryNote:
    """In-memory representation of a single A-MEM note."""

    __slots__ = (
        "note_id", "content", "keywords", "links",
        "embedding", "created_at", "updated_at", "access_count",
    )

    def __init__(
        self,
        note_id: str,
        content: str,
        keywords: list[str],
        links: list[dict],
        embedding: list[float],
        created_at: str,
        updated_at: str,
        access_count: int = 0,
    ) -> None:
        self.note_id      = note_id
        self.content      = content
        self.keywords     = keywords
        self.links        = links          # [{note_id, relationship}]
        self.embedding    = embedding
        self.created_at   = created_at
        self.updated_at   = updated_at
        self.access_count = access_count

    def to_doc(self) -> dict:
        return {
            "note_id":      self.note_id,
            "content":      self.content,
            "keywords":     self.keywords,
            "links":        self.links,
            "embedding":    self.embedding,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_doc(cls, doc: dict) -> "MemoryNote":
        return cls(
            note_id=doc["note_id"],
            content=doc["content"],
            keywords=doc.get("keywords", []),
            links=doc.get("links", []),
            embedding=doc.get("embedding", []),
            created_at=doc.get("created_at", ""),
            updated_at=doc.get("updated_at", ""),
            access_count=doc.get("access_count", 0),
        )


class AMEMService:
    """
    Agentic memory service. Inject a motor database at construction.
    Call init() once to load the encoder.
    """

    def __init__(self, db=None) -> None:
        self._db      = db       # motor AsyncIOMotorDatabase
        self._encoder = None
        self._ready   = False

    async def init(self) -> None:
        """Load embedding model (CPU-bound, run in thread pool)."""
        import asyncio, concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            await loop.run_in_executor(ex, self._load_encoder)
        if self._db is not None:
            await self._ensure_index()
        self._ready = True
        logger.info("AMEMService: ready (encoder={})", EMBED_MODEL)

    def _load_encoder(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(EMBED_MODEL)

    async def _ensure_index(self) -> None:
        try:
            await self._db[COLLECTION].create_index("note_id", unique=True)
            await self._db[COLLECTION].create_index("keywords")
        except Exception:
            pass

    @property
    def is_ready(self) -> bool:
        return self._ready and self._encoder is not None

    # ── Core API ───────────────────────────────────────────────────────────────

    async def add_memory(self, content: str, keywords: list[str]) -> str:
        """
        Create a new memory note, auto-embed it, and auto-link to similar notes.

        Returns:
            note_id of the newly created note.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        embedding = self._embed(content)
        note_id   = str(uuid.uuid4())

        # Auto-link to existing similar notes
        links = await self._find_auto_links(embedding, note_id)

        note = MemoryNote(
            note_id=note_id,
            content=content,
            keywords=keywords,
            links=links,
            embedding=embedding,
            created_at=now,
            updated_at=now,
            access_count=0,
        )

        await self._save(note)

        # Add back-links from linked notes to this new note
        for link in links:
            await self._add_backlink(
                link["note_id"], note_id, link["relationship"]
            )

        logger.debug(
            "AMEMService: added note {} with {} auto-links", note_id[:8], len(links)
        )
        return note_id

    async def search_memory(
        self, query: str, limit: int = 3
    ) -> list[tuple[MemoryNote, float]]:
        """
        Semantic search over all memory notes.

        Returns:
            List of (MemoryNote, similarity_score) sorted by similarity desc.
            Also includes linked neighbours of top hits (Zettelkasten recall).
        """
        if not self.is_ready:
            return []

        query_vec = np.array(self._embed(query), dtype=np.float32)
        all_notes = await self._load_all()

        if not all_notes:
            return []

        # Cosine similarity over all notes
        matrix = np.array(
            [n.embedding for n in all_notes], dtype=np.float32
        )
        # Normalise rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        matrix_norm = matrix / norms
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        sims = matrix_norm @ q_norm

        top_idx  = np.argsort(sims)[::-1][:limit]
        top_hits = [(all_notes[i], float(sims[i])) for i in top_idx if sims[i] > 0.1]

        # Increment access count for retrieved notes
        for note, _ in top_hits:
            note.access_count += 1
            await self._update_access_count(note.note_id, note.access_count)

        # Zettelkasten: surface linked neighbours of top hits
        note_ids_seen = {n.note_id for n, _ in top_hits}
        extra: list[tuple[MemoryNote, float]] = []
        note_map = {n.note_id: (n, s) for n, s in zip(all_notes, sims)}

        for note, score in top_hits:
            for link in note.links[:2]:  # max 2 neighbours per hit
                lid = link["note_id"]
                if lid not in note_ids_seen and lid in note_map:
                    linked_note, linked_sim = note_map[lid]
                    extra.append((linked_note, linked_sim * 0.85))  # slight discount
                    note_ids_seen.add(lid)

        results = top_hits + extra
        results.sort(key=lambda x: -x[1])
        return results[:limit + 2]  # return a few extra for context

    async def update_memory(self, note_id: str, new_observation: str) -> bool:
        """
        Append a new observation to an existing note and re-embed.
        This is how the agent updates its knowledge after each incident.

        Returns:
            True if note was found and updated, False otherwise.
        """
        note = await self._load_one(note_id)
        if note is None:
            logger.warning("AMEMService: note {} not found for update", note_id[:8])
            return False

        note.content   = f"{note.content}\n\nUpdate ({datetime.now(tz=timezone.utc).date()}): {new_observation}"
        note.embedding = self._embed(note.content)
        note.updated_at = datetime.now(tz=timezone.utc).isoformat()
        await self._save(note, upsert=True)
        logger.debug("AMEMService: updated note {}", note_id[:8])
        return True

    async def link_memories(
        self, note_id_a: str, note_id_b: str, relationship: str
    ) -> None:
        """Explicitly link two memory notes with a named relationship."""
        await self._add_backlink(note_id_a, note_id_b, relationship)
        await self._add_backlink(note_id_b, note_id_a, relationship)

    async def memory_count(self) -> int:
        if self._db is None:
            return 0
        try:
            return await self._db[COLLECTION].count_documents({})
        except Exception:
            return 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        vec = self._encoder.encode([text], show_progress_bar=False)[0]
        return vec.tolist()

    async def _save(self, note: MemoryNote, upsert: bool = False) -> None:
        if self._db is None:
            return
        try:
            if upsert:
                await self._db[COLLECTION].replace_one(
                    {"note_id": note.note_id}, note.to_doc(), upsert=True
                )
            else:
                await self._db[COLLECTION].insert_one(note.to_doc())
        except Exception as exc:
            logger.warning("AMEMService._save failed: {}", exc)

    async def _load_all(self) -> list[MemoryNote]:
        if self._db is None:
            return []
        try:
            cursor = self._db[COLLECTION].find({}, {"_id": 0})
            docs   = await cursor.to_list(length=10_000)
            return [MemoryNote.from_doc(d) for d in docs if d.get("embedding")]
        except Exception as exc:
            logger.warning("AMEMService._load_all failed: {}", exc)
            return []

    async def _load_one(self, note_id: str) -> Optional[MemoryNote]:
        if self._db is None:
            return None
        try:
            doc = await self._db[COLLECTION].find_one({"note_id": note_id}, {"_id": 0})
            return MemoryNote.from_doc(doc) if doc else None
        except Exception:
            return None

    async def _update_access_count(self, note_id: str, count: int) -> None:
        if self._db is None:
            return
        try:
            await self._db[COLLECTION].update_one(
                {"note_id": note_id}, {"$set": {"access_count": count}}
            )
        except Exception:
            pass

    async def _add_backlink(
        self, note_id: str, target_id: str, relationship: str
    ) -> None:
        if self._db is None:
            return
        try:
            await self._db[COLLECTION].update_one(
                {"note_id": note_id},
                {"$addToSet": {"links": {"note_id": target_id, "relationship": relationship}}},
            )
        except Exception:
            pass

    async def _find_auto_links(
        self, embedding: list[float], exclude_id: str
    ) -> list[dict]:
        """Find existing notes similar enough to auto-link."""
        all_notes = await self._load_all()
        if not all_notes:
            return []

        q_vec    = np.array(embedding, dtype=np.float32)
        q_norm   = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        matrix   = np.array([n.embedding for n in all_notes], dtype=np.float32)
        norms    = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        sims     = (matrix / norms) @ q_norm

        links = []
        for i in np.argsort(sims)[::-1]:
            if len(links) >= AUTO_LINK_TOP_K:
                break
            if sims[i] < AUTO_LINK_THRESHOLD:
                break
            note = all_notes[i]
            if note.note_id != exclude_id:
                links.append({
                    "note_id":      note.note_id,
                    "relationship": "similar_pattern",
                })
        return links
