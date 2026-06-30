"""
self-healing-rag/api/schemas.py
Pydantic v2 request/response models for all API endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# ── Ingest ────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text to ingest")
    source: str = Field(default="manual", description="Identifier for this source")


class IngestURLRequest(BaseModel):
    url: str = Field(..., description="URL to fetch and ingest")


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    avg_quality: float
    elapsed_s: float
    total_in_store: int


# ── Query ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2048)
    k: int = Field(default=10, ge=1, le=50, description="Candidates to retrieve")
    final_k: int = Field(default=4, ge=1, le=20, description="Top-k after re-ranking")
    use_reranker: bool = True
    use_hyde: bool = False
    use_multi_query: bool = False
    filters: dict[str, Any] | None = None


class ChunkInfo(BaseModel):
    chunk_id: str
    source: str
    score: float
    rank: int
    quality_score: float
    token_count: int | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    chunks: list[ChunkInfo]
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    total_latency_ms: float
    heal_rounds: int = 0
    verdict: str | None = None


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Store stats ───────────────────────────────────────────────

class StoreStatsResponse(BaseModel):
    total_chunks: int
    provider: str