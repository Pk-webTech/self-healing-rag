"""
self-healing-rag/retrieval/retriever.py
Hybrid retrieval: dense vector search + BM25 keyword search,
fused via Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

from collections import defaultdict

from rank_bm25 import BM25Okapi

from core.config import get_settings
from core.logger import logger
from core.models import RetrievedChunk
from ingestion.embedder import Embedder
from ingestion.vector_store import BaseVectorStore

settings = get_settings()


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Dense + BM25 hybrid retrieval with RRF fusion.

    Parameters
    ----------
    store       : vector store instance (ChromaDB / FAISS)
    embedder    : embedder instance for query embedding
    k           : number of candidates from each sub-retriever
    alpha       : weight for dense scores vs BM25 (0 = BM25 only, 1 = dense only)
    threshold   : minimum fused score to include in results
    """

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: Embedder,
        k: int | None = None,
        alpha: float | None = None,
        threshold: float | None = None,
    ) -> None:
        cfg = settings.retrieval_cfg
        self.store = store
        self.embedder = embedder
        self.k = k or cfg["k"]
        self.alpha = alpha if alpha is not None else cfg["alpha"]
        self.threshold = threshold if threshold is not None else cfg.get("score_threshold", 0.0)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[RetrievedChunk] = []

    # ── BM25 index ────────────────────────────────────────────

    def _build_bm25_index(self, chunks: list[RetrievedChunk]) -> None:
        """Build a lightweight BM25 index over the provided chunks."""
        corpus = [c.chunk.content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_chunks = chunks
        logger.debug(f"BM25 index built over {len(chunks)} chunks")

    def _bm25_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._bm25_chunks:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        result = []
        for rank, idx in enumerate(top_idxs):
            rc = self._bm25_chunks[idx]
            result.append(
                RetrievedChunk(
                    chunk=rc.chunk,
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return result

    # ── RRF fusion ────────────────────────────────────────────

    def _fuse(
        self,
        dense_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Combine dense and BM25 results via weighted RRF.
        fused_score = alpha * rrf(dense_rank) + (1-alpha) * rrf(bm25_rank)
        """
        scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, rc in enumerate(dense_results):
            cid = rc.chunk.chunk_id
            scores[cid] += self.alpha * _rrf_score(rank)
            chunk_map[cid] = rc

        for rank, rc in enumerate(bm25_results):
            cid = rc.chunk.chunk_id
            scores[cid] += (1 - self.alpha) * _rrf_score(rank)
            if cid not in chunk_map:
                chunk_map[cid] = rc

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        fused = []
        for new_rank, cid in enumerate(sorted_ids):
            rc = chunk_map[cid]
            fused.append(
                RetrievedChunk(chunk=rc.chunk, score=scores[cid], rank=new_rank)
            )
        return fused

    # ── public API ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieve for a single query string.
        Returns up to `k` chunks ranked by RRF-fused score.
        """
        k = k or self.k

        # 1. Dense retrieval
        query_vec = self.embedder.embed_query(query)
        dense = self.store.query(query_vec, k=k, filters=filters)

        # 2. BM25 retrieval (build index on-the-fly from dense candidates)
        #    For production use a persistent BM25 index; this is sufficient for Phase 1.
        if dense:
            self._build_bm25_index(dense)
        bm25 = self._bm25_retrieve(query, k=k)

        # 3. Fuse
        fused = self._fuse(dense, bm25)

        # 4. Filter by threshold
        filtered = [rc for rc in fused if rc.score >= self.threshold]
        logger.debug(
            f"Hybrid retrieve: dense={len(dense)} bm25={len(bm25)} "
            f"fused={len(fused)} after_threshold={len(filtered)}"
        )
        return filtered[:k]

    def retrieve_multi(
        self,
        queries: list[str],
        k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve for multiple query variants and de-duplicate by chunk_id.
        Used with multi-query or HyDE expansion.
        """
        seen: dict[str, RetrievedChunk] = {}
        for query in queries:
            for rc in self.retrieve(query, k=k, filters=filters):
                cid = rc.chunk.chunk_id
                if cid not in seen or rc.score > seen[cid].score:
                    seen[cid] = rc

        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        for i, rc in enumerate(merged):
            rc.rank = i
        return merged[: (k or self.k)]