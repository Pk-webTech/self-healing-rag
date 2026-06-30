"""
self-healing-rag/ingestion/__init__.py
Ingestion pipeline — orchestrates Loader → Chunker → Embedder → Tagger → VectorStore.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from core.logger import logger
from core.models import Chunk, Document
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from ingestion.metadata import MetadataTagger
from ingestion.vector_store import BaseVectorStore, build_vector_store


class IngestionPipeline:
    """
    End-to-end ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline()
        stats = pipeline.ingest_files(["docs/paper.pdf", "docs/notes.md"])
        stats = pipeline.ingest_url("https://arxiv.org/abs/...")
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.loader = DocumentLoader()
        self.chunker = chunker or Chunker()
        self.embedder = embedder or Embedder()
        self.tagger = MetadataTagger()
        self.store = vector_store or build_vector_store()

    # ── internal ─────────────────────────────────────────────

    def _process_docs(self, docs: list[Document]) -> dict[str, Any]:
        t0 = time.perf_counter()
        chunks = self.chunker.chunk_many(docs)
        chunks = self.tagger.tag(chunks)
        chunks = self.embedder.embed_chunks(chunks)
        self.store.add_chunks(chunks)
        elapsed = time.perf_counter() - t0

        stats = {
            "documents": len(docs),
            "chunks": len(chunks),
            "avg_quality": round(
                sum(c.quality_score for c in chunks) / max(len(chunks), 1), 4
            ),
            "elapsed_s": round(elapsed, 2),
            "total_in_store": self.store.count(),
        }
        logger.info(f"Ingestion complete: {stats}")
        return stats

    # ── public API ────────────────────────────────────────────

    def ingest_files(self, paths: list[str | Path]) -> dict[str, Any]:
        docs = self.loader.load_many([str(p) for p in paths])
        return self._process_docs(docs)

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> dict[str, Any]:
        docs = self.loader.load_directory(directory, recursive=recursive)
        return self._process_docs(docs)

    def ingest_url(self, url: str) -> dict[str, Any]:
        doc = self.loader.load_url(url)
        return self._process_docs([doc])

    def ingest_text(self, text: str, source: str = "manual") -> dict[str, Any]:
        from core.models import Document
        doc = Document(content=text, source=source, doc_type="text")
        return self._process_docs([doc])