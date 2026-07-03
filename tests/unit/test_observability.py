"""
tests/unit/test_observability.py
Unit tests for Phase 5 observability components.
Uses isolated Prometheus registries so tests don't pollute each other.
No file I/O for tracer — uses tmp_path fixture.
No real HTTP calls for alerting backends.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.models import (
    Chunk, EvalResult, EvalScore, GenerationResult,
    RetrievalResult, RetrievedChunk, Verdict,
)


# ── shared helpers ────────────────────────────────────────────────────

def _chunk():
    return Chunk(
        chunk_id="c001", content="test content", source="doc.txt",
        doc_type="text", chunk_index=0, quality_score=0.8,
    )

def _rc():
    return RetrievedChunk(chunk=_chunk(), score=0.85, rank=0)

def _retrieval(query="What is RAG?"):
    return RetrievalResult(
        query=query,
        chunks=[_rc()],
        context_text="[Source 1: doc.txt | score=0.850]\nRAG is retrieval augmented generation.",
    )

def _generation():
    return GenerationResult(
        answer="RAG is retrieval augmented generation.",
        query="What is RAG?", context_text="ctx",
        sources=["doc.txt"], model="gpt-4o-mini",
        prompt_tokens=100, completion_tokens=30, latency_ms=250.0,
    )

def _eval(verdict=Verdict.PASS, weighted=0.88):
    scores = [
        EvalScore(judge_name="faithfulness", score=0.9, passed=True),
        EvalScore(judge_name="relevance", score=0.85, passed=True),
        EvalScore(judge_name="grounding", score=0.88, passed=True),
    ]
    return EvalResult(
        verdict=verdict, scores=scores, weighted_score=weighted,
        generation=_generation(), retrieval=_retrieval(),
        metadata={"failed_judges": []},
    )


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

class TestRAGMetrics:
    """Tests use an isolated CollectorRegistry per test to avoid conflicts."""

    def _metrics(self):
        try:
            from prometheus_client import CollectorRegistry
            from observability.metrics import RAGMetrics
            registry = CollectorRegistry()
            return RAGMetrics(registry=registry)
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_observe_query_increments_counter(self):
        m = self._metrics()
        m.observe_query(
            total_latency_ms=300.0,
            verdict="PASS",
            heal_rounds=0,
            generation=_generation(),
            eval_result=_eval(),
        )
        # Collect all samples and find the _total sample for verdict=PASS
        samples = list(m.queries_total.labels(verdict="PASS").collect())[0].samples
        count = next(s.value for s in samples if s.name.endswith("_total") and not s.name.endswith("_created"))
        assert count == 1.0

    def test_observe_query_none_verdict_uses_none_label(self):
        m = self._metrics()
        m.observe_query(total_latency_ms=100.0, verdict=None, heal_rounds=0)
        samples = list(m.queries_total.labels(verdict="none").collect())[0].samples
        count = next(s.value for s in samples if s.name.endswith("_total") and not s.name.endswith("_created"))
        assert count == 1.0

    def test_observe_query_never_raises_on_bad_input(self):
        m = self._metrics()
        # Pass garbage values — must not raise
        m.observe_query(
            total_latency_ms=-1.0,
            verdict=None,
            heal_rounds=-1,
            generation=None,
            eval_result=None,
        )

    def test_observe_heal_action_increments_counter(self):
        m = self._metrics()
        m.observe_heal_action("expand_query")
        m.observe_heal_action("expand_query")
        m.observe_heal_action("quarantine_chunk")
        samples = list(m.heal_events_total.labels(action="expand_query").collect())[0].samples
        count = next(s.value for s in samples if s.name.endswith("_total") and not s.name.endswith("_created"))
        assert count == 2.0

    def test_update_store_size_sets_gauge(self):
        m = self._metrics()
        m.update_store_size(42)
        samples = list(m.vector_store_size.collect())[0].samples
        val = next(s.value for s in samples if not s.name.endswith("_created"))
        assert val == 42.0

    def test_tokens_recorded_correctly(self):
        m = self._metrics()
        gen = _generation()
        m.observe_query(
            total_latency_ms=200.0, verdict="PASS",
            heal_rounds=0, generation=gen,
        )
        prompt_samples = list(m.tokens_total.labels(type="prompt").collect())[0].samples
        prompt_count = next(s.value for s in prompt_samples if s.name.endswith("_total") and not s.name.endswith("_created"))
        assert prompt_count == gen.prompt_tokens


# ══════════════════════════════════════════════════════════════════════
# Tracer
# ══════════════════════════════════════════════════════════════════════

class TestRequestTracer:
    def _tracer(self, tmp_path):
        from observability.tracer import RequestTracer
        trace_file = tmp_path / "traces.jsonl"
        with patch("observability.tracer.settings") as mock_s:
            mock_s.observability_cfg = {
                "tracer_enabled": True,
                "trace_log_path": str(trace_file),
                "trace_max_file_mb": 10,
            }
            tracer = RequestTracer(trace_path=trace_file)
        return tracer, trace_file

    def test_trace_writes_jsonl_record(self, tmp_path):
        tracer, trace_file = self._tracer(tmp_path)
        trace_id = tracer.trace(
            retrieval=_retrieval(),
            generation=_generation(),
            eval_result=_eval(),
            total_latency_ms=350.0,
            heal_rounds=1,
            actions_taken=["r1:expand_query"],
        )
        assert trace_file.exists()
        line = trace_file.read_text().strip()
        record = json.loads(line)
        assert record["trace_id"] == trace_id
        assert record["query"] == "What is RAG?"
        assert record["verdict"] == "PASS"
        assert record["heal_rounds"] == 1
        assert record["actions_taken"] == ["r1:expand_query"]
        assert record["latency_ms"] == 350.0

    def test_trace_truncates_long_query(self, tmp_path):
        tracer, trace_file = self._tracer(tmp_path)
        long_query = "x" * 1000
        ret = RetrievalResult(query=long_query, chunks=[], context_text="")
        tracer.trace(retrieval=ret, generation=_generation())
        record = json.loads(trace_file.read_text().strip())
        assert len(record["query"]) <= 500

    def test_read_recent_returns_newest_first(self, tmp_path):
        tracer, trace_file = self._tracer(tmp_path)
        for i in range(5):
            ret = RetrievalResult(query=f"query {i}", chunks=[], context_text="")
            tracer.trace(retrieval=ret, generation=_generation())
        records = tracer.read_recent(n=5)
        # Newest first — last written query should be first
        assert records[0]["query"] == "query 4"
        assert records[-1]["query"] == "query 0"

    def test_read_recent_returns_empty_when_no_file(self, tmp_path):
        from observability.tracer import RequestTracer
        tracer = RequestTracer(trace_path=tmp_path / "nonexistent.jsonl")
        result = tracer.read_recent()
        assert result == []

    def test_trace_rotates_on_size_exceeded(self, tmp_path):
        from observability.tracer import RequestTracer
        trace_file = tmp_path / "traces.jsonl"
        # Set max to 1 byte so any write triggers rotation
        tracer = RequestTracer(trace_path=trace_file)
        tracer._max_bytes = 1
        # Write two traces — second should trigger rotation
        tracer.trace(retrieval=_retrieval(), generation=_generation())
        tracer.trace(retrieval=_retrieval(), generation=_generation())
        backup = trace_file.with_suffix(".jsonl.1")
        assert backup.exists()

    def test_trace_never_raises_on_bad_eval_result(self, tmp_path):
        tracer, _ = self._tracer(tmp_path)
        # eval_result=None is valid (evaluate=False path)
        trace_id = tracer.trace(
            retrieval=_retrieval(),
            generation=_generation(),
            eval_result=None,
        )
        assert isinstance(trace_id, str)


# ══════════════════════════════════════════════════════════════════════
# Alert Engine
# ══════════════════════════════════════════════════════════════════════

class TestAlertEngine:
    def _engine(self, overrides=None):
        from observability.alerting import AlertEngine
        cfg = {
            "alerting_enabled": True,
            "alert_window_queries": 10,
            "alert_score_drop_threshold": 0.15,
            "alert_heal_rate_threshold": 0.5,
            "alert_latency_p95_ms": 1000.0,
            "alert_backend": "log",
            "alert_webhook_url": "",
            "alert_slack_webhook_url": "",
        }
        if overrides:
            cfg.update(overrides)
        with patch("observability.alerting.settings") as mock_s:
            mock_s.observability_cfg = cfg
            engine = AlertEngine(cooldown_queries=0)  # no cooldown for testing
        return engine

    def test_no_alerts_below_window_minimum(self):
        engine = self._engine()
        # Only 1 record — below minimum of window_size // 5 = 2
        alerts = engine.record(weighted_score=0.2, heal_rounds=5, total_latency_ms=9999.0)
        assert alerts == []

    def test_high_heal_rate_fires_alert(self):
        engine = self._engine()
        # Fill window: 8 out of 10 queries need healing → 80% > 50% threshold
        for i in range(10):
            engine.record(
                weighted_score=0.8,
                heal_rounds=1 if i < 8 else 0,
                total_latency_ms=200.0,
            )
        history = engine.get_history()
        types = [a.alert_type for a in history]
        assert "HIGH_HEAL_RATE" in types

    def test_high_latency_fires_alert(self):
        engine = self._engine()
        # All queries very slow
        for _ in range(10):
            engine.record(weighted_score=0.9, heal_rounds=0, total_latency_ms=5000.0)
        history = engine.get_history()
        types = [a.alert_type for a in history]
        assert "HIGH_LATENCY" in types

    def test_score_drop_fires_after_baseline_established(self):
        engine = self._engine({"alert_window_queries": 10})
        # Fill full window with high scores to establish baseline
        for _ in range(10):
            engine.record(weighted_score=0.9, heal_rounds=0, total_latency_ms=200.0)
        # Force baseline
        engine._baseline_score = 0.9
        # Now inject low scores
        for _ in range(10):
            engine.record(weighted_score=0.6, heal_rounds=0, total_latency_ms=200.0)
        history = engine.get_history()
        types = [a.alert_type for a in history]
        assert "SCORE_DROP" in types

    def test_alert_disabled_fires_nothing(self):
        engine = self._engine({"alerting_enabled": False})
        for _ in range(20):
            engine.record(weighted_score=0.0, heal_rounds=5, total_latency_ms=99999.0)
        assert engine.get_history() == []

    def test_cooldown_prevents_alert_storm(self):
        from observability.alerting import AlertEngine
        cfg = {
            "alerting_enabled": True,
            "alert_window_queries": 5,
            "alert_score_drop_threshold": 0.15,
            "alert_heal_rate_threshold": 0.3,
            "alert_latency_p95_ms": 100.0,
            "alert_backend": "log",
            "alert_webhook_url": "",
            "alert_slack_webhook_url": "",
        }
        with patch("observability.alerting.settings") as mock_s:
            mock_s.observability_cfg = cfg
            engine = AlertEngine(cooldown_queries=5)  # 5-query cooldown

        # Trigger condition for HIGH_HEAL_RATE
        for _ in range(30):
            engine.record(weighted_score=0.8, heal_rounds=1, total_latency_ms=50.0)

        # Should have fired at most ceil(30/5) = 6 times, not 30 times
        heal_rate_alerts = [a for a in engine.get_history() if a.alert_type == "HIGH_HEAL_RATE"]
        assert len(heal_rate_alerts) <= 6

    def test_get_history_respects_limit(self):
        engine = self._engine()
        engine._baseline_score = 0.9
        for _ in range(20):
            engine.record(weighted_score=0.0, heal_rounds=5, total_latency_ms=9999.0)
        history = engine.get_history(limit=3)
        assert len(history) <= 3


# ══════════════════════════════════════════════════════════════════════
# ObservabilityManager
# ══════════════════════════════════════════════════════════════════════

class TestObservabilityManager:
    def test_record_returns_trace_id_string(self, tmp_path):
        from observability import ObservabilityManager
        manager = ObservabilityManager()
        manager.tracer._path = tmp_path / "traces.jsonl"
        manager.metrics = None  # disable prometheus for this test

        trace_id = manager.record(
            retrieval=_retrieval(),
            generation=_generation(),
            eval_result=_eval(),
            total_latency_ms=400.0,
            heal_rounds=0,
        )
        assert isinstance(trace_id, str)
        assert len(trace_id) == 32  # uuid4 hex

    def test_record_never_raises_when_all_components_crash(self):
        from observability import ObservabilityManager
        manager = ObservabilityManager()
        # Make all components raise
        manager.metrics = MagicMock()
        manager.metrics.observe_query.side_effect = RuntimeError("metrics crash")
        manager.tracer.trace = MagicMock(side_effect=RuntimeError("tracer crash"))
        manager.alerting.record = MagicMock(side_effect=RuntimeError("alerting crash"))

        # Must not raise
        result = manager.record(
            retrieval=_retrieval(),
            generation=_generation(),
            total_latency_ms=100.0,
        )
        assert result == "n/a"  # returned because tracer crashed

    def test_action_name_stripped_of_round_prefix(self, tmp_path):
        from observability import ObservabilityManager
        manager = ObservabilityManager()
        manager.tracer._path = tmp_path / "traces.jsonl"

        recorded_actions = []
        def capture_action(action):
            recorded_actions.append(action)
        manager.metrics = MagicMock()
        manager.metrics.observe_query = MagicMock()
        manager.metrics.observe_heal_action.side_effect = capture_action

        manager.record(
            retrieval=_retrieval(),
            generation=_generation(),
            total_latency_ms=200.0,
            actions_taken=["r1:expand_query", "r2:quarantine_chunk"],
        )
        assert "expand_query" in recorded_actions
        assert "quarantine_chunk" in recorded_actions

    def test_update_store_metrics_safe_when_no_prometheus(self):
        from observability import ObservabilityManager
        manager = ObservabilityManager()
        manager.metrics = None  # simulate prometheus_client not installed
        # Must not raise
        manager.update_store_metrics(store_count=100, avg_quality=0.75)