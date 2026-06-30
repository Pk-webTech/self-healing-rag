"""
self-healing-rag/ingestion/vector_store.py
Vector store abstraction layer.
Primary backend: ChromaDB (persistent)
Fallback backend: FAISS (in-memory, for testing)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.logger import logger
from core.models import Chunk, RetrievedChunk

settings = get_settings()


# ── base ──────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None: ...

    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


# ── ChromaDB backend ──────────────────────────────────────────

class ChromaVectorStore(BaseVectorStore):
    """
    Persistent ChromaDB store.
    Each chunk is stored with its embedding + all metadata fields.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:
            raise ImportError("pip install chromadb") from e

        cfg = settings.vector_store_cfg
        persist_dir = persist_dir or settings.chroma_persist_dir
        collection_name = collection_name or settings.chroma_collection

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": cfg.get("distance_metric", "cosine")},
        )
        logger.info(
            f"ChromaDB ready: collection='{collection_name}' "
            f"path='{persist_dir}' docs={self._col.count()}"
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids, embeddings, documents, metadatas = [], [], [], []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.chunk_id} has no embedding — run Embedder first")
            ids.append(c.chunk_id)
            embeddings.append(c.embedding)
            documents.append(c.content)
            # Chroma metadata must be flat str/int/float/bool
            meta = {
                "source": c.source,
                "doc_type": c.doc_type,
                "chunk_index": c.chunk_index,
                "quality_score": c.quality_score,
                "heal_flag": c.heal_flag,
                "failure_count": c.failure_count,
                "retrieval_count": c.retrieval_count,
            }
            # merge extra metadata (flatten lists/dicts to json strings)
            for k, v in c.metadata.items():
                if k not in meta:
                    meta[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            metadatas.append(meta)

        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks → ChromaDB (total={self._col.count()})")

    def query(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        where = {}
        if filters:
            where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._col.count() or 1),
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Chroma returns L2 or cosine distance; convert to similarity
            score = 1.0 - dist  # works for cosine space
            chunk = Chunk(
                chunk_id=results["ids"][0][i],
                content=doc,
                source=meta.get("source", ""),
                doc_type=meta.get("doc_type", "text"),
                chunk_index=int(meta.get("chunk_index", 0)),
                quality_score=float(meta.get("quality_score", 1.0)),
                heal_flag=bool(meta.get("heal_flag", False)),
                failure_count=int(meta.get("failure_count", 0)),
                retrieval_count=int(meta.get("retrieval_count", 0)),
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=i))
        return retrieved

    def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        self._col.update(ids=[chunk_id], metadatas=[updates])

    def delete_chunk(self, chunk_id: str) -> None:
        self._col.delete(ids=[chunk_id])
        logger.info(f"Deleted chunk {chunk_id[:8]} from ChromaDB")

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        name = self._col.name
        self._client.delete_collection(name)
        self._col = self._client.get_or_create_collection(name)
        logger.warning(f"ChromaDB collection '{name}' reset (all data deleted)")


# ── FAISS backend (lightweight / tests) ──────────────────────

class FAISSVectorStore(BaseVectorStore):
    """In-memory FAISS store — useful for local testing without persistence."""

    def __init__(self, dimension: int = 1536) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise ImportError("pip install faiss-cpu numpy") from e
        import faiss
        import numpy as np
        self._np = np
        self._dim = dimension
        self._index = faiss.IndexFlatIP(dimension)  # inner product = cosine after normalise
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        logger.info(f"FAISS in-memory store initialised (dim={dimension})")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        import faiss
        vecs = self._np.array([c.embedding for c in chunks], dtype="float32")
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        for c in chunks:
            self._chunks[c.chunk_id] = c
            self._ids.append(c.chunk_id)

    def query(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        import faiss
        vec = self._np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vec)
        scores, idxs = self._index.search(vec, min(k, len(self._ids)))
        result = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0])):
            if idx == -1:
                continue
            cid = self._ids[idx]
            result.append(RetrievedChunk(chunk=self._chunks[cid], score=float(score), rank=rank))
        return result

    def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        if chunk_id in self._chunks:
            self._chunks[chunk_id].metadata.update(updates)

    def delete_chunk(self, chunk_id: str) -> None:
        self._chunks.pop(chunk_id, None)

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks.clear()
        self._ids.clear()


# ── factory ───────────────────────────────────────────────────

def build_vector_store(provider: str | None = None) -> BaseVectorStore:
    prov = provider or settings.vector_store_cfg.get("provider", "chroma")
    if prov == "chroma":
        return ChromaVectorStore()
    if prov == "faiss":
        return FAISSVectorStore()
    raise ValueError(f"Unknown vector store provider: {prov}")