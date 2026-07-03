"""
self-healing-rag/observability/metrics.py
Prometheus metrics for the Self-Healing RAG pipeline.

All metrics use a single module-level CollectorRegistry (not the default
global registry) so tests can create isolated instances without
"Duplicated timeseries" errors from prometheus_client.

Metrics exposed
───────────────
  shr_queries_total              Counter   — total queries by verdict
  shr_query_latency_ms           Histogram — end-to-end query latency
  shr_generation_latency_ms      Histogram — LLM generation latency only
  shr_heal_events_total          Counter   — healing actions by action name
  shr_heal_rounds_histogram      Histogram — rounds per healed query
  shr_weighted_score             Gauge     — latest weighted_score (rolling)
  shr_chunk_quality_gauge        Gauge     — avg chunk quality in store
  shr_vector_store_size          Gauge     — total chunks in vector store
  shr_tokens_total               Counter   — LLM tokens consumed by type

Bug-prevention notes
────────────────────
- All label cardinality is bounded: verdict ∈ {PASS, SOFT_FAIL, HARD_FAIL, none},
  action ∈ fixed set, model is capped at first 32 chars to prevent label explosion.
- observe() is entirely fail-safe — never raises, never blocks the request path.
- We avoid prometheus_client's push_to_gateway (requires a running PushGateway)
  and instead expose a /metrics HTTP endpoint that Prometheus scrapes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

from core.config import get_settings
from core.logger import logger

if TYPE_CHECKING:
    from core.models import EvalResult, GenerationResult

settings = get_settings()

# ── registry ──────────────────────────────────────────────────────────
# Module-level singleton — created once per process.
# Using a dedicated registry (not prometheus_client.REGISTRY) means tests
# can instantiate RAGMetrics without polluting the global state.
_REGISTRY: "CollectorRegistry | None" = None


def get_registry() -> "CollectorRegistry":
    global _REGISTRY
    if _REGISTRY is None:
        if not _PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client not installed. "
                "Run: pip install prometheus-client"
            )
        _REGISTRY = CollectorRegistry()
    return _REGISTRY


# ── metric definitions ────────────────────────────────────────────────

class RAGMetrics:
    """
    Container for all Prometheus metrics.
    Instantiated once as a module-level singleton via get_metrics().
    """

    def __init__(self, registry: "CollectorRegistry") -> None:
        self.queries_total = Counter(
            "shr_queries_total",
            "Total RAG queries processed",
            labelnames=["verdict"],
            registry=registry,
        )
        self.query_latency_ms = Histogram(
            "shr_query_latency_ms",
            "End-to-end query latency in milliseconds",
            buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000],
            registry=registry,
        )
        self.generation_latency_ms = Histogram(
            "shr_generation_latency_ms",
            "LLM generation latency in milliseconds",
            buckets=[50, 100, 250, 500, 1000, 2000, 5000],
            registry=registry,
        )
        self.heal_events_total = Counter(
            "shr_heal_events_total",
            "Total healing actions triggered",
            labelnames=["action"],
            registry=registry,
        )
        self.heal_rounds_histogram = Histogram(
            "shr_heal_rounds",
            "Number of healing rounds per query",
            buckets=[1, 2, 3, 4, 5],  # +Inf appended automatically
            registry=registry,
        )
        self.weighted_score_gauge = Gauge(
            "shr_weighted_score_latest",
            "Latest weighted evaluation score (rolling update)",
            registry=registry,
        )
        self.chunk_quality_gauge = Gauge(
            "shr_avg_chunk_quality",
            "Average chunk quality score in the vector store",
            registry=registry,
        )
        self.vector_store_size = Gauge(
            "shr_vector_store_chunks_total",
            "Total number of chunks in the vector store",
            registry=registry,
        )
        self.tokens_total = Counter(
            "shr_tokens_total",
            "Total LLM tokens consumed",
            labelnames=["type"],  # "prompt" | "completion"
            registry=registry,
        )

    def observe_query(
        self,
        total_latency_ms: float,
        verdict: str | None,
        heal_rounds: int,
        generation: "GenerationResult | None" = None,
        eval_result: "EvalResult | None" = None,
    ) -> None:
        """Record metrics for one completed query. Never raises."""
        try:
            label = verdict or "none"
            self.queries_total.labels(verdict=label).inc()
            self.query_latency_ms.observe(total_latency_ms)
            self.heal_rounds_histogram.observe(heal_rounds)

            if generation is not None:
                self.generation_latency_ms.observe(generation.latency_ms)
                self.tokens_total.labels(type="prompt").inc(generation.prompt_tokens)
                self.tokens_total.labels(type="completion").inc(generation.completion_tokens)

            if eval_result is not None:
                self.weighted_score_gauge.set(eval_result.weighted_score)

        except Exception as exc:
            logger.error(f"[Metrics] observe_query failed: {exc}")

    def observe_heal_action(self, action: str) -> None:
        """Record one healing action. Never raises."""
        try:
            self.heal_events_total.labels(action=action[:64]).inc()
        except Exception as exc:
            logger.error(f"[Metrics] observe_heal_action failed: {exc}")

    def update_store_size(self, count: int) -> None:
        """Update vector store size gauge. Never raises."""
        try:
            self.vector_store_size.set(count)
        except Exception as exc:
            logger.error(f"[Metrics] update_store_size failed: {exc}")

    def update_chunk_quality(self, avg_quality: float) -> None:
        """Update average chunk quality gauge. Never raises."""
        try:
            self.chunk_quality_gauge.set(avg_quality)
        except Exception as exc:
            logger.error(f"[Metrics] update_chunk_quality failed: {exc}")


# ── singleton ─────────────────────────────────────────────────────────
_METRICS: RAGMetrics | None = None


def get_metrics() -> RAGMetrics | None:
    """
    Return the singleton RAGMetrics, or None if prometheus_client is not installed.
    Callers must handle the None case gracefully.
    """
    global _METRICS
    if _METRICS is None:
        if not _PROMETHEUS_AVAILABLE:
            logger.warning(
                "[Metrics] prometheus_client not installed — metrics disabled. "
                "Run: pip install prometheus-client"
            )
            return None
        cfg = settings.observability_cfg
        if not cfg.get("metrics_enabled", True):
            return None
        _METRICS = RAGMetrics(registry=get_registry())
    return _METRICS


def render_metrics() -> tuple[bytes, str]:
    """
    Render current metrics as Prometheus text format.
    Returns (body_bytes, content_type_string).
    Raises ImportError if prometheus_client is not installed.
    """
    if not _PROMETHEUS_AVAILABLE:
        raise ImportError("prometheus_client not installed")
    body = generate_latest(get_registry())
    return body, CONTENT_TYPE_LATEST