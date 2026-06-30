"""
self-healing-rag/retrieval/__init__.py
Retrieval pipeline: QueryProcessor → HybridRetriever → Reranker → ContextBuilder
"""
from __future__ import annotations

from core.models import RetrievalResult
from ingestion.embedder import Embedder
from ingestion.vector_store import BaseVectorStore
from retrieval.context_builder import ContextBuilder
from retrieval.query_processor import QueryProcessor
from retrieval.reranker import build_reranker
from retrieval.retriever import HybridRetriever


class RetrievalPipeline:
    """
    Usage:
        pipeline = RetrievalPipeline(store=store, embedder=embedder)
        result: RetrievalResult = pipeline.run("What is self-healing RAG?")
    """

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: Embedder,
        use_reranker: bool | None = None,
        use_hyde: bool = False,
        use_multi_query: bool = False,
    ) -> None:
        self.query_processor = QueryProcessor(
            use_hyde=use_hyde, use_multi_query=use_multi_query
        )
        self.retriever = HybridRetriever(store=store, embedder=embedder)
        self.reranker = build_reranker(enabled=use_reranker)
        self.context_builder = ContextBuilder()

    def run(self, query: str, filters: dict | None = None) -> RetrievalResult:
        queries = self.query_processor.process(query)
        raw_chunks = self.retriever.retrieve_multi(queries, filters=filters)
        reranked = self.reranker.rerank(query, raw_chunks)
        return self.context_builder.build(query, reranked)