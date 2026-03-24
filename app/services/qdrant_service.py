"""
Qdrant Vector Store Service — manages two collections:
  - "maintenance_logs"  : past failure incidents (symptoms, root cause, action taken)
  - "machine_manuals"   : machine manual sections (for future RAG expansion)

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, free, 384-dim)

Usage:
    svc = QdrantService()
    await svc.init()
    await svc.upsert_logs(logs)
    results = await svc.search_similar_incidents("high temperature HDF", failure_type="HDF")
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
from typing import Optional

import numpy as np
from loguru import logger

from app.models.maintenance import MaintenanceLog

COLLECTION_LOGS    = "maintenance_logs"
COLLECTION_MANUALS = "machine_manuals"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE        = 384


def _make_point_id(text: str) -> int:
    """Deterministic int ID from text hash (Qdrant requires int or UUID)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:15], 16)


def _log_to_embed_text(log: MaintenanceLog) -> str:
    """Embedding text captures failure signature for semantic search."""
    return (
        f"{log.failure_type}: {log.symptoms}. "
        f"Root cause: {log.root_cause}. "
        f"Action: {log.action_taken}."
    )


class QdrantService:
    """
    Async Qdrant client wrapping vector upsert and search.
    Embedding runs in a thread-pool executor (CPU-bound).
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None) -> None:
        self._url     = url
        self._api_key = api_key
        self._client  = None
        self._encoder = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Connect to Qdrant and load the embedding model."""
        loop = asyncio.get_event_loop()

        # Load Qdrant client
        await loop.run_in_executor(self._executor, self._connect)

        # Load sentence-transformers model (slow first load — run in thread)
        await loop.run_in_executor(self._executor, self._load_encoder)

        # Ensure collections exist
        await loop.run_in_executor(self._executor, self._ensure_collections)

        logger.info("QdrantService: ready (url={})", self._url)

    def _connect(self) -> None:
        from qdrant_client import QdrantClient
        kwargs = {"url": self._url}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._client = QdrantClient(**kwargs)
        logger.info("QdrantService: connected to {}", self._url)

    def _load_encoder(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("QdrantService: embedding model '{}' loaded", EMBEDDING_MODEL)

    def _ensure_collections(self) -> None:
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
        existing = {c.name for c in self._client.get_collections().collections}

        for name in (COLLECTION_LOGS, COLLECTION_MANUALS):
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                logger.info("QdrantService: created collection '{}'", name)
            else:
                logger.info("QdrantService: collection '{}' already exists", name)

        # Create keyword index on failure_type so filtered queries work
        try:
            self._client.create_payload_index(
                collection_name=COLLECTION_LOGS,
                field_name="failure_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("QdrantService: payload index on 'failure_type' ensured")
        except Exception:
            pass  # index may already exist — safe to ignore

    # ── Upsert ─────────────────────────────────────────────────────────────────

    async def upsert_logs(self, logs: list[MaintenanceLog]) -> int:
        """
        Embed and upsert MaintenanceLogs into 'maintenance_logs' collection.
        Returns number of logs upserted.
        """
        if not logs:
            return 0
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(self._executor, self._upsert_logs_sync, logs)
        return count

    def _upsert_logs_sync(self, logs: list[MaintenanceLog]) -> int:
        from qdrant_client.models import PointStruct

        texts   = [_log_to_embed_text(log) for log in logs]
        vectors = self._encoder.encode(texts, batch_size=64, show_progress_bar=False)

        points = []
        for log, vec in zip(logs, vectors):
            point_id = _make_point_id(log.log_id)
            payload  = log.model_dump(mode="json")
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        self._client.upsert(collection_name=COLLECTION_LOGS, points=points)
        logger.info("QdrantService: upserted {} logs into '{}'", len(points), COLLECTION_LOGS)
        return len(points)

    # ── Search ─────────────────────────────────────────────────────────────────

    async def search_similar_incidents(
        self,
        query: str,
        failure_type: Optional[str] = None,
        limit: int = 3,
    ) -> list[MaintenanceLog]:
        """
        Semantic search for past incidents similar to query.

        Args:
            query:        Natural language description of current anomaly.
            failure_type: Optional filter — only return incidents of this type.
            limit:        Max results to return.

        Returns:
            List of MaintenanceLog ordered by similarity (most similar first).
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self._search_sync,
            query,
            failure_type,
            limit,
        )
        return results

    def _search_sync(
        self,
        query: str,
        failure_type: Optional[str],
        limit: int,
    ) -> list[MaintenanceLog]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_vec = self._encoder.encode([query], show_progress_bar=False)[0].tolist()

        # Optional failure-type filter
        search_filter = None
        if failure_type and failure_type not in ("NONE", "none"):
            search_filter = Filter(
                must=[FieldCondition(key="failure_type", match=MatchValue(value=failure_type))]
            )

        # qdrant-client >= 1.7: query_points replaces search()
        results = self._client.query_points(
            collection_name=COLLECTION_LOGS,
            query=query_vec,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        hits = results.points

        logs = []
        for hit in hits:
            try:
                logs.append(MaintenanceLog(**hit.payload))
            except Exception as exc:
                logger.warning("QdrantService: failed to parse hit payload — {}", exc)
        return logs

    async def collection_count(self, name: str = COLLECTION_LOGS) -> int:
        """Return number of vectors in a collection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._client.count(collection_name=name).count,
        )

    @property
    def is_ready(self) -> bool:
        return self._client is not None and self._encoder is not None
