"""
tests/integration/test_pipeline_phase1.py
End-to-end integration test for Phase 1: Ingest → Retrieve → Generate
Uses FAISS (no Chroma needed) and mocks LLM calls.
"""
import pytest
from unittest.mock import patch, MagicMock

from core.models import Document
from ingestion import IngestionPipeline
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from ingestion.vector_store import FAISSVectorStore
from retrieval import RetrievalPipeline
from retrieval.reranker import IdentityReranker


SAMPLE_DOCS = [
    Document(
        content=(
            "Self-healing RAG is a system that automatically detects hallucinations "
            "in generated answers by running RAGAS evaluation judges including "
            "faithfulness, answer relevance, and context precision. "
            "When a failure is detected, the system triggers healing actions "
            "such as expanding the query, re-ranking with a different model, "
            "or flagging and re-indexing low-quality chunks."
        ),
        source="rag_paper.txt",
        doc_type="text",
        metadata={"filename": "rag_paper.txt"},
    ),
    Document(
        content=(
            "Vector databases store dense embeddings and support approximate "
            "nearest-neighbour (ANN) search. ChromaDB is a popular open-source "
            "option that supports cosine similarity and metadata filtering. "
            "FAISS is an in-memory alternative built by Meta for fast retrieval."
        ),
        source="vector_db.txt",
        doc_type="text",
        metadata={"filename": "vector_db.txt"},
    ),
]


def _mock_embedder(dim: int = 64) -> Embedder:
    """Create an Embedder that returns deterministic random embeddings."""
    import random

    emb = MagicMock(spec=Embedder)
    emb.dimension = dim

    def fake_embed_chunks(chunks):
        rng = random.Random(42)
        for c in chunks:
            c.embedding = [rng.uniform(-1, 1) for _ in range(dim)]
        return chunks

    def fake_embed_query(text):
        rng = random.Random(hash(text) % 1000)
        return [rng.uniform(-1, 1) for _ in range(dim)]

    emb.embed_chunks.side_effect = fake_embed_chunks
    emb.embed_query.side_effect = fake_embed_query
    return emb


def _mock_generator_fn():
    """Return a callable that produces a fake generation result."""
    from core.models import GenerationResult

    def gen(retrieval_result):
        return GenerationResult(
            answer=(
                "Self-healing RAG detects hallucinations using RAGAS judges "
                "[Source: rag_paper.txt] and then triggers healing actions."
            ),
            query=retrieval_result.query,
            context_text=retrieval_result.context_text,
            sources=["rag_paper.txt"],
            model="gpt-4o-mini-mock",
            prompt_tokens=120,
            completion_tokens=40,
            latency_ms=150.0,
        )

    return gen


# ── tests ─────────────────────────────────────────────────────

class TestPhase1Pipeline:
    """Full Phase 1 integration: ingest → retrieve → generate."""

    def setup_method(self):
        self.store = FAISSVectorStore(dimension=64)
        self.embedder = _mock_embedder(dim=64)
        self.chunker = Chunker(strategy="recursive", chunk_size=80, chunk_overlap=10)

        self.ingest = IngestionPipeline(
            vector_store=self.store,
            chunker=self.chunker,
            embedder=self.embedder,
        )

    def test_ingestion_produces_chunks(self):
        stats = self.ingest._process_docs(SAMPLE_DOCS)
        assert stats["documents"] == 2
        assert stats["chunks"] >= 2
        assert 0.0 <= stats["avg_quality"] <= 1.0
        assert self.store.count() >= 2

    def test_retrieval_returns_results(self):
        self.ingest._process_docs(SAMPLE_DOCS)

        retrieval_pipeline = RetrievalPipeline(
            store=self.store,
            embedder=self.embedder,
            use_reranker=False,  # avoid loading CE model in tests
        )
        result = retrieval_pipeline.run("What is self-healing RAG?")

        assert result.query == "What is self-healing RAG?"
        assert len(result.chunks) >= 1
        assert result.context_text != ""
        assert "Source 1:" in result.context_text

    def test_full_pipeline_ingest_retrieve_generate(self):
        """End-to-end: ingest → retrieve → mock-generate."""
        self.ingest._process_docs(SAMPLE_DOCS)

        retrieval_pipeline = RetrievalPipeline(
            store=self.store,
            embedder=self.embedder,
            use_reranker=False,
        )

        from generation.generator import Generator
        gen = MagicMock(spec=Generator)
        gen.generate.side_effect = _mock_generator_fn()

        result = retrieval_pipeline.run("How does self-healing RAG work?")
        gen_result = gen.generate(result)

        assert gen_result.answer != ""
        assert gen_result.model == "gpt-4o-mini-mock"
        assert gen_result.prompt_tokens > 0
        assert "Source" in gen_result.answer

    def test_store_persistence_across_ingestions(self):
        """Ingesting twice should increase chunk count."""
        self.ingest._process_docs([SAMPLE_DOCS[0]])
        count_after_first = self.store.count()
        self.ingest._process_docs([SAMPLE_DOCS[1]])
        count_after_second = self.store.count()
        assert count_after_second > count_after_first

    def test_quality_scores_are_valid(self):
        stats = self.ingest._process_docs(SAMPLE_DOCS)
        # All quality scores should be between 0 and 1
        assert 0.0 <= stats["avg_quality"] <= 1.0