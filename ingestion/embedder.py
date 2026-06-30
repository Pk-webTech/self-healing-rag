"""
self-healing-rag/ingestion/embedder.py
Embedding wrapper supporting OpenAI and HuggingFace (sentence-transformers).
Includes batch processing, retry logic, and dimension validation.
"""
from __future__ import annotations

import time
from typing import Literal

from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logger import logger
from core.models import Chunk

settings = get_settings()


# ── OpenAI Embedder ───────────────────────────────────────────

class OpenAIEmbedder:
    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self.model = model or settings.embedding_cfg["openai_model"]
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.dimension = settings.embedding_cfg["dimension"]
        logger.info(f"OpenAIEmbedder initialised: model={self.model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


# ── HuggingFace Embedder ──────────────────────────────────────

class HuggingFaceEmbedder:
    def __init__(self, model: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e
        self.model_name = model or settings.embedding_cfg["hf_model"]
        self._model = SentenceTransformer(self.model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"HuggingFaceEmbedder initialised: model={self.model_name}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


# ── Factory + public Embedder class ──────────────────────────

EmbedderBackend = OpenAIEmbedder | HuggingFaceEmbedder


def _build_backend(
    provider: Literal["openai", "huggingface"] | None = None,
) -> EmbedderBackend:
    prov = provider or settings.embedding_provider
    if prov == "openai":
        return OpenAIEmbedder()
    if prov == "huggingface":
        return HuggingFaceEmbedder()
    raise ValueError(f"Unknown embedding provider: {prov}")


class Embedder:
    """
    High-level embedder that:
    - auto-batches up to `batch_size` texts per API call
    - attaches embeddings in-place to Chunk objects
    - exposes embed_query() for retrieval-time embedding

    Usage:
        embedder = Embedder()
        chunks = embedder.embed_chunks(chunks)
        vec = embedder.embed_query("what is RAG?")
    """

    def __init__(
        self,
        provider: Literal["openai", "huggingface"] | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._backend = _build_backend(provider)
        self.batch_size = batch_size or settings.embedding_cfg["batch_size"]
        self.dimension = self._backend.dimension

    def embed_query(self, text: str) -> list[float]:
        return self._backend.embed_query(text)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all chunks in batches; mutates Chunk.embedding in place."""
        total = len(chunks)
        logger.info(f"Embedding {total} chunks (batch_size={self.batch_size})")
        t0 = time.perf_counter()

        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [c.content for c in batch]
            vecs = self._backend.embed_batch(texts)
            for chunk, vec in zip(batch, vecs):
                chunk.embedding = vec

        elapsed = time.perf_counter() - t0
        logger.info(f"Embedded {total} chunks in {elapsed:.2f}s")
        return chunks