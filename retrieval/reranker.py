"""
self-healing-rag/retrieval/reranker.py
Cross-encoder re-ranker using sentence-transformers.
Takes (query, [chunks]) and returns chunks sorted by cross-encoder score.
"""
from __future__ import annotations

from core.config import get_settings
from core.logger import logger
from core.models import RetrievedChunk

settings = get_settings()


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model.
    Default: cross-encoder/ms-marco-MiniLM-L-6-v2

    Usage:
        reranker = CrossEncoderReranker()
        top_k = reranker.rerank(query, chunks, top_k=4)
    """

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e

        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.retrieval_cfg.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = CrossEncoder(self.model_name)
        logger.info(f"CrossEncoderReranker loaded: {self.model_name}")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        top_k = top_k or settings.retrieval_cfg.get("final_k", 4)
        pairs = [(query, rc.chunk.content) for rc in chunks]
        ce_scores = self._model.predict(pairs).tolist()

        for rc, score in zip(chunks, ce_scores):
            rc.score = float(score)  # overwrite with cross-encoder score

        reranked = sorted(chunks, key=lambda r: r.score, reverse=True)[:top_k]
        for i, rc in enumerate(reranked):
            rc.rank = i

        logger.debug(
            f"Reranked {len(chunks)} → top-{top_k}: "
            f"scores={[round(r.score,3) for r in reranked]}"
        )
        return reranked


class IdentityReranker:
    """No-op reranker — returns chunks unchanged (useful when CE is disabled)."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_cfg.get("final_k", 4)
        return chunks[:top_k]


def build_reranker(enabled: bool | None = None):
    use = enabled if enabled is not None else settings.retrieval_cfg.get("use_reranker", True)
    if use:
        return CrossEncoderReranker()
    return IdentityReranker()