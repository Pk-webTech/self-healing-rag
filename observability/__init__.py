"""
self-healing-rag/observability/__init__.py
ObservabilityManager: single call-site for all Phase 5 telemetry.

After every query the caller does:
    obs_manager.record(retrieval, generation, eval_result, ...)

This internally:
  1. Calls RAGMetrics.observe_query()          (Prometheus counters/histograms)
  2. Calls RequestTracer.trace()               (JSONL file)
  3. Calls AlertEngine.record()                (threshold checks)

All three are fail-safe — exceptions are caught per-component.
The manager itself also never raises.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import get_settings
from core.logger import logger
from observability.alerting import Alert, AlertEngine
from observability.metrics import RAGMetrics, get_metrics
from observability.tracer import RequestTracer

if TYPE_CHECKING:
    from core.models import EvalResult, GenerationResult, RetrievalResult

settings = get_settings()


class ObservabilityManager:
    """
    Single facade for all Phase 5 telemetry.

    Usage:
        manager = ObservabilityManager()
        manager.record(
            retrieval=retrieval_result,
            generation=generation_result,
            eval_result=eval_result,       # None if evaluate=False
            total_latency_ms=345.2,
            heal_rounds=1,
            actions_taken=["r1:expand_query"],
        )
    """

    def __init__(self) -> None:
        self.metrics: RAGMetrics | None = get_metrics()
        self.tracer = RequestTracer()
        self.alerting = AlertEngine()

    def record(
        self,
        retrieval: "RetrievalResult",
        generation: "GenerationResult",
        total_latency_ms: float,
        eval_result: "EvalResult | None" = None,
        heal_rounds: int = 0,
        actions_taken: list[str] | None = None,
        error: str | None = None,
    ) -> str:
        """
        Record telemetry for one completed query.
        Returns the trace_id string. Never raises.
        """
        verdict = eval_result.verdict.value if eval_result else None
        weighted_score = eval_result.weighted_score if eval_result else None
        trace_id = "n/a"

        # ── 1. Prometheus metrics ─────────────────────────────────────
        if self.metrics is not None:
            try:
                self.metrics.observe_query(
                    total_latency_ms=total_latency_ms,
                    verdict=verdict,
                    heal_rounds=heal_rounds,
                    generation=generation,
                    eval_result=eval_result,
                )
                # Record each healing action separately
                for action in (actions_taken or []):
                    # actions_taken entries are "r1:expand_query" — strip round prefix
                    action_name = action.split(":")[-1] if ":" in action else action
                    self.metrics.observe_heal_action(action_name)
            except Exception as exc:
                logger.error(f"[ObservabilityManager] metrics failed: {exc}")

        # ── 2. Tracer ────────────────────────────────────────────────
        try:
            trace_id = self.tracer.trace(
                retrieval=retrieval,
                generation=generation,
                eval_result=eval_result,
                total_latency_ms=total_latency_ms,
                heal_rounds=heal_rounds,
                actions_taken=actions_taken,
                error=error,
            )
        except Exception as exc:
            logger.error(f"[ObservabilityManager] tracer failed: {exc}")

        # ── 3. Alert engine ───────────────────────────────────────────
        try:
            fired = self.alerting.record(
                weighted_score=weighted_score,
                heal_rounds=heal_rounds,
                total_latency_ms=total_latency_ms,
            )
            if fired:
                logger.info(
                    f"[ObservabilityManager] {len(fired)} alert(s) fired: "
                    f"{[a.alert_type for a in fired]}"
                )
        except Exception as exc:
            logger.error(f"[ObservabilityManager] alerting failed: {exc}")

        return trace_id

    def update_store_metrics(self, store_count: int, avg_quality: float | None = None) -> None:
        """Update store-level gauges. Call after ingest or evolver runs."""
        if self.metrics is None:
            return
        try:
            self.metrics.update_store_size(store_count)
            if avg_quality is not None:
                self.metrics.update_chunk_quality(avg_quality)
        except Exception as exc:
            logger.error(f"[ObservabilityManager] update_store_metrics failed: {exc}")