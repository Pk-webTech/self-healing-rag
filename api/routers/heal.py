"""
self-healing-rag/api/routers/heal.py
GET  /heal/events         — list recent heal events
GET  /heal/events/{id}    — single heal event
GET  /heal/stats          — aggregate healing statistics
GET  /heal/query-logs     — list recent query logs
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import HealEvent, QueryLog
from db.session import get_session

router = APIRouter(prefix="/heal", tags=["Healing"])


# ── response schemas ─────────────────────────────────────────────────

class HealEventOut(BaseModel):
    id: int
    query_log_id: int | None
    query: str
    verdict_before: str
    weighted_score_before: float
    failed_judges: list[str]
    action: str
    round_number: int
    chunk_id: str | None
    verdict_after: str | None
    weighted_score_after: float | None
    healed: bool
    extra: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryLogOut(BaseModel):
    id: int
    query: str
    answer: str
    model: str
    verdict: str | None
    weighted_score: float | None
    heal_rounds: int
    latency_ms: float
    total_latency_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HealStats(BaseModel):
    total_queries: int
    total_heal_events: int
    healed_count: int
    heal_success_rate: float          # healed_count / total_heal_events (or 0)
    avg_heal_rounds: float
    verdict_distribution: dict[str, int]
    action_distribution: dict[str, int]


# ── endpoints ────────────────────────────────────────────────────────

@router.get("/events", response_model=list[HealEventOut])
async def list_heal_events(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[HealEventOut]:
    """List heal events in reverse-chronological order."""
    result = await session.execute(
        select(HealEvent)
        .order_by(HealEvent.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = result.scalars().all()
    return [HealEventOut.model_validate(r) for r in rows]


@router.get("/events/{event_id}", response_model=HealEventOut)
async def get_heal_event(
    event_id: int,
    session: AsyncSession = Depends(get_session),
) -> HealEventOut:
    """Fetch a single HealEvent by id."""
    result = await session.execute(
        select(HealEvent).where(HealEvent.id == event_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"HealEvent {event_id} not found")
    return HealEventOut.model_validate(row)


@router.get("/query-logs", response_model=list[QueryLogOut])
async def list_query_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    verdict: str | None = Query(default=None, description="Filter by verdict (PASS/SOFT_FAIL/HARD_FAIL)"),
    session: AsyncSession = Depends(get_session),
) -> list[QueryLogOut]:
    """List query logs, optionally filtered by verdict."""
    q = select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit).offset(offset)
    if verdict:
        q = q.where(QueryLog.verdict == verdict.upper())
    result = await session.execute(q)
    rows = result.scalars().all()
    return [QueryLogOut.model_validate(r) for r in rows]


@router.get("/stats", response_model=HealStats)
async def heal_stats(
    session: AsyncSession = Depends(get_session),
) -> HealStats:
    """Aggregate statistics across all heal events and query logs."""

    # Total queries
    total_queries = (await session.execute(
        select(func.count()).select_from(QueryLog)
    )).scalar_one()

    # Total heal events
    total_events = (await session.execute(
        select(func.count()).select_from(HealEvent)
    )).scalar_one()

    # Healed count
    healed_count = (await session.execute(
        select(func.count()).select_from(HealEvent).where(HealEvent.healed == True)  # noqa: E712
    )).scalar_one()

    # Average heal rounds (from query logs that had at least 1 round)
    avg_rounds_result = (await session.execute(
        select(func.avg(QueryLog.heal_rounds)).where(QueryLog.heal_rounds > 0)
    )).scalar_one()
    avg_rounds = round(float(avg_rounds_result or 0.0), 2)

    # Verdict distribution
    verdict_rows = (await session.execute(
        select(QueryLog.verdict, func.count()).group_by(QueryLog.verdict)
    )).all()
    verdict_dist = {(v or "unknown"): c for v, c in verdict_rows}

    # Action distribution
    action_rows = (await session.execute(
        select(HealEvent.action, func.count()).group_by(HealEvent.action)
    )).all()
    action_dist = {a: c for a, c in action_rows}

    success_rate = round(healed_count / total_events, 4) if total_events > 0 else 0.0

    return HealStats(
        total_queries=total_queries,
        total_heal_events=total_events,
        healed_count=healed_count,
        heal_success_rate=success_rate,
        avg_heal_rounds=avg_rounds,
        verdict_distribution=verdict_dist,
        action_distribution=action_dist,
    )