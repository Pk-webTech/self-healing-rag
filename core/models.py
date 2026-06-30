"""
self-healing-rag/core/models.py
Shared dataclasses / typed containers used across all phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ── Ingestion ────────────────────────────────────────────────

@dataclass
class Document:
    """Raw document before chunking."""
    content: str
    source: str                          # file path or URL
    doc_type: str = "text"              # pdf | text | markdown | html
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    chunk_id: str
    content: str
    source: str
    doc_type: str
    chunk_index: int
    quality_score: float = 1.0          # initial quality (degraded by healing)
    heal_flag: bool = False
    failure_count: int = 0
    retrieval_count: int = 0
    last_healed: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


# ── Retrieval ────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store with a similarity score."""
    chunk: Chunk
    score: float                         # cosine similarity (0–1)
    rank: int = 0                        # rank after re-ranking


@dataclass
class RetrievalResult:
    """Output of the full retrieval pipeline."""
    query: str
    chunks: list[RetrievedChunk]
    context_text: str                    # concatenated context for LLM
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Generation ───────────────────────────────────────────────

@dataclass
class GenerationResult:
    """LLM response with metadata."""
    answer: str
    query: str
    context_text: str
    sources: list[str]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Evaluation (Phase 2) ─────────────────────────────────────

class Verdict(str, Enum):
    PASS = "PASS"
    SOFT_FAIL = "SOFT_FAIL"
    HARD_FAIL = "HARD_FAIL"


@dataclass
class EvalScore:
    """Score from a single judge."""
    judge_name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregated result from all judges."""
    verdict: Verdict
    scores: list[EvalScore]
    weighted_score: float
    generation: GenerationResult
    retrieval: RetrievalResult
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Pipeline Response ─────────────────────────────────────────

@dataclass
class PipelineResponse:
    """Final response returned to the user / API layer."""
    query: str
    answer: str
    sources: list[str]
    retrieval: RetrievalResult
    generation: GenerationResult
    eval_result: EvalResult | None = None
    heal_rounds: int = 0
    total_latency_ms: float = 0.0