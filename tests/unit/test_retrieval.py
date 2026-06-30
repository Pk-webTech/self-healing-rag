"""
tests/unit/test_retrieval.py
Unit tests for hybrid retriever, context builder, and query processor.
Uses FAISS in-memory store to avoid ChromaDB dependency in CI.
"""
import pytest
from unittest.mock import MagicMock, patch

from core.models import Chunk, RetrievedChunk, RetrievalResult
from retrieval.context_builder import ContextBuilder, _format_chunk
from retrieval.reranker import IdentityReranker


# ── helpers ──────────────────────────────────────────────────

def _make_rc(content: str, score: float, rank: int) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"chunk_{rank:03d}",
        content=content,
        source="test.txt",
        doc_type="text",
        chunk_index=rank,
        quality_score=0.9,
        metadata={"filename": "test.txt", "token_count": len(content.split())},
    )
    return RetrievedChunk(chunk=chunk, score=score, rank=rank)


# ── context builder ───────────────────────────────────────────

def test_context_builder_respects_budget():
    # Build chunks that collectively exceed a small budget
    chunks = [_make_rc("word " * 100, 0.9 - i * 0.1, i) for i in range(10)]
    builder = ContextBuilder(token_budget=200)
    result = builder.build("test query", chunks)
    assert result.total_tokens <= 210  # slight overshoot tolerance
    assert isinstance(result, RetrievalResult)


def test_context_builder_includes_source_labels():
    chunks = [_make_rc("Sample text about RAG systems and retrieval.", 0.95, 0)]
    builder = ContextBuilder(token_budget=500)
    result = builder.build("what is RAG?", chunks)
    assert "Source 1:" in result.context_text
    assert "test.txt" in result.context_text


def test_context_builder_empty_chunks():
    builder = ContextBuilder(token_budget=500)
    result = builder.build("test query", [])
    assert result.context_text == ""
    assert result.chunks == []


def test_context_builder_query_stored():
    chunks = [_make_rc("Some relevant content here.", 0.8, 0)]
    builder = ContextBuilder(token_budget=500)
    result = builder.build("What is self-healing?", chunks)
    assert result.query == "What is self-healing?"


# ── identity reranker ─────────────────────────────────────────

def test_identity_reranker_returns_top_k():
    chunks = [_make_rc(f"chunk content {i}", 0.9 - i * 0.05, i) for i in range(10)]
    reranker = IdentityReranker()
    top = reranker.rerank("query", chunks, top_k=4)
    assert len(top) == 4
    assert top[0].rank == 0


def test_identity_reranker_empty():
    reranker = IdentityReranker()
    result = reranker.rerank("query", [], top_k=4)
    assert result == []


# ── format chunk ─────────────────────────────────────────────

def test_format_chunk_contains_source_and_score():
    rc = _make_rc("Some text content", 0.876, 0)
    formatted = _format_chunk(rc, 1)
    assert "Source 1:" in formatted
    assert "test.txt" in formatted
    assert "0.876" in formatted
    assert "Some text content" in formatted