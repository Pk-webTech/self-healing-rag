"""
self-healing-rag/api/routers/observability.py
Phase 5 observability endpoints.

GET  /metrics              — Prometheus text format (for Prometheus scraping)
GET  /traces               — recent request traces (JSONL → JSON list)
GET  /alerts/history       — fired alerts from current process lifetime
GET  /alerts/window-stats  — current rolling window statistics
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from api.deps import get_observability_manager
from observability import ObservabilityManager
from observability.metrics import render_metrics

router = APIRouter(tags=["Observability"])


# ── response schemas ─────────────────────────────────────────────────

class AlertOut(BaseModel):
    alert_type: str
    message: str
    value: float
    threshold: float
    fired_at: str


class WindowStatsOut(BaseModel):
    window_size: int
    queries_in_window: int
    avg_weighted_score: float | None
    heal_rate: float
    p95_latency_ms: float | None
    alerts_fired_total: int


# ── endpoints ─────────────────────────────────────────────────────────

@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus metrics scrape endpoint.
    Returns text/plain in Prometheus exposition format.
    """
    try:
        body, content_type = render_metrics()
        return PlainTextResponse(content=body.decode("utf-8"), media_type=content_type)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="prometheus_client not installed. Run: pip install prometheus-client",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/traces", response_model=list[dict[str, Any]])
async def get_traces(
    limit: int = Query(default=50, ge=1, le=500),
    manager: ObservabilityManager = Depends(get_observability_manager),
) -> list[dict[str, Any]]:
    """Return the most recent request traces (newest first)."""
    return manager.tracer.read_recent(n=limit)


@router.get("/alerts/history", response_model=list[AlertOut])
async def alert_history(
    limit: int = Query(default=100, ge=1, le=1000),
    manager: ObservabilityManager = Depends(get_observability_manager),
) -> list[AlertOut]:
    """Return fired alerts from the current process lifetime (newest first)."""
    alerts = manager.alerting.get_history(limit=limit)
    return [
        AlertOut(
            alert_type=a.alert_type,
            message=a.message,
            value=a.value,
            threshold=a.threshold,
            fired_at=a.fired_at,
        )
        for a in alerts
    ]


@router.get("/alerts/window-stats", response_model=WindowStatsOut)
async def window_stats(
    manager: ObservabilityManager = Depends(get_observability_manager),
) -> WindowStatsOut:
    """Current rolling window statistics used by the alert engine."""
    engine = manager.alerting
    window = list(engine._window)

    scores = [s.weighted_score for s in window if s.weighted_score is not None]
    avg_score = round(sum(scores) / len(scores), 4) if scores else None

    healed = sum(1 for s in window if s.heal_rounds > 0)
    heal_rate = round(healed / len(window), 4) if window else 0.0

    latencies = sorted(s.total_latency_ms for s in window)
    p95 = None
    if latencies:
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95 = round(latencies[p95_idx], 1)

    return WindowStatsOut(
        window_size=engine._window_size,
        queries_in_window=len(window),
        avg_weighted_score=avg_score,
        heal_rate=heal_rate,
        p95_latency_ms=p95,
        alerts_fired_total=len(engine._history),
    )