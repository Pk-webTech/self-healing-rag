"""
tests/unit/test_ingestion.py
Unit tests for chunker, metadata tagger, and ingestion pipeline.
"""
import pytest
from core.models import Document, Chunk
from ingestion.chunker import Chunker, _recursive_split, _semantic_split
from ingestion.metadata import MetadataTagger, _alpha_ratio, _repetition_penalty


# ── chunker ──────────────────────────────────────────────────

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines retrieval
of relevant documents with language model generation.

It was introduced to address the limitations of pure parametric knowledge
in large language models (LLMs). By grounding generation in retrieved context,
RAG reduces hallucinations and improves factual accuracy.

The pipeline consists of three main components: a document store, a retriever,
and a generator. These components work together to produce grounded answers.

Self-healing RAG extends this by adding an evaluation loop that detects
poor-quality generations and automatically triggers retrieval or re-indexing
to correct them.
""".strip()


def test_recursive_split_produces_chunks():
    chunks = _recursive_split(SAMPLE_TEXT, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.strip()) > 0


def test_semantic_split_produces_chunks():
    chunks = _semantic_split(SAMPLE_TEXT, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 1


def test_chunker_chunk_returns_chunk_objects():
    doc = Document(content=SAMPLE_TEXT, source="test.txt", doc_type="text")
    chunker = Chunker(strategy="recursive", chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.chunk_id for c in chunks)
    assert all(c.source == "test.txt" for c in chunks)


def test_chunker_chunk_indexes_are_sequential():
    doc = Document(content=SAMPLE_TEXT, source="test.txt", doc_type="text")
    chunker = Chunker(strategy="recursive", chunk_size=80, chunk_overlap=10)
    chunks = chunker.chunk(doc)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunker_semantic_strategy():
    doc = Document(content=SAMPLE_TEXT, source="test.md", doc_type="markdown")
    chunker = Chunker(strategy="semantic", chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1


def test_chunk_many_aggregates():
    docs = [
        Document(content=SAMPLE_TEXT, source=f"doc_{i}.txt", doc_type="text")
        for i in range(3)
    ]
    chunker = Chunker(strategy="recursive", chunk_size=100, chunk_overlap=20)
    all_chunks = chunker.chunk_many(docs)
    assert len(all_chunks) > 3  # more than one chunk per doc


# ── metadata tagger ───────────────────────────────────────────

def test_alpha_ratio_clean_text():
    score = _alpha_ratio("Hello world this is clean text")
    assert score > 0.7


def test_alpha_ratio_garbled():
    score = _alpha_ratio("123 456 789 !!! ??? ###")
    assert score < 0.3


def test_repetition_penalty_unique():
    score = _repetition_penalty("The quick brown fox jumps over the lazy dog")
    assert score > 0.7


def test_repetition_penalty_repeated():
    score = _repetition_penalty("the the the the the the the the the the")
    assert score < 0.3


def test_metadata_tagger_sets_quality_score():
    doc = Document(content=SAMPLE_TEXT, source="test.txt", doc_type="text")
    chunker = Chunker(strategy="recursive", chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)
    tagger = MetadataTagger()
    tagged = tagger.tag(chunks)
    for c in tagged:
        assert 0.0 <= c.quality_score <= 1.0
        assert "indexed_at" in c.metadata
        assert "quality_score" in c.metadata


def test_metadata_tagger_low_quality_flagged(caplog):
    """A chunk with only numbers should receive a low quality score."""
    bad_chunk = Chunk(
        chunk_id="bad001",
        content="123 456 789 000 111 222 333 444 555 666",
        source="test.txt",
        doc_type="text",
        chunk_index=0,
    )
    tagger = MetadataTagger()
    tagger.tag([bad_chunk])
    assert bad_chunk.quality_score < 0.5