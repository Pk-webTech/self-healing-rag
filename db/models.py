"""
self-healing-rag/db/models.py
SQLAlchemy 2.x ORM models.

Tables
------
query_logs   — one row per user query (with final verdict + latency)
heal_events  — one row per healing action taken (FK → query_logs.id)

Design notes
------------
- Uses mapped_column / Mapped syntax (SQLAlchemy 2.x) for full typing.
- JSON columns store dicts as TEXT (SQLite-compatible; Postgres uses JSONB natively).
- All timestamps are UTC stored as strings for portability.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    # Evaluation
    verdict: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    weighted_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    judge_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Pipeline performance
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    heal_rounds: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    # Relationship
    heal_events: Mapped[list[HealEvent]] = relationship(
        "HealEvent", back_populates="query_log", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} verdict={self.verdict} query={self.query[:40]!r}>"


class HealEvent(Base):
    __tablename__ = "heal_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_log_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("query_logs.id", ondelete="SET NULL"), nullable=True
    )

    # What triggered healing
    query: Mapped[str] = mapped_column(Text, nullable=False)
    verdict_before: Mapped[str] = mapped_column(String(32), nullable=False)
    weighted_score_before: Mapped[float] = mapped_column(Float, default=0.0)
    failed_judges: Mapped[list] = mapped_column(JSON, default=list)  # ["faithfulness", ...]

    # What action was taken
    action: Mapped[str] = mapped_column(String(64), nullable=False)   # e.g. "expand_query"
    round_number: Mapped[int] = mapped_column(Integer, default=1)

    # What chunk(s) were affected (nullable — not all actions target a specific chunk)
    chunk_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chunk_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Outcome
    verdict_after: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    weighted_score_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    healed: Mapped[bool] = mapped_column(Boolean, default=False)

    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    # Relationship
    query_log: Mapped[Optional[QueryLog]] = relationship(
        "QueryLog", back_populates="heal_events"
    )

    def __repr__(self) -> str:
        return (
            f"<HealEvent id={self.id} action={self.action!r} "
            f"round={self.round_number} healed={self.healed}>"
        )


class ChunkQualityHistory(Base):
    """
    Time-series record of a chunk's quality score after each eval cycle.
    Used by the QualityTracker's EWM rolling average and the ChunkEvolver.
    One row per (chunk_id, eval cycle) — not per query (too noisy).
    """
    __tablename__ = "chunk_quality_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Score at this point in time
    raw_quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    ewm_quality_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Context that caused the update
    verdict: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    weighted_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    heal_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    # What the evolver decided (if anything)
    evolver_action: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # "drop" | "keep" | "flag"

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkQualityHistory chunk={self.chunk_id[:8]} "
            f"ewm={self.ewm_quality_score:.3f} action={self.evolver_action}>"
        )