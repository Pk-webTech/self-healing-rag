"""
tests/unit/test_healing.py
Unit tests for dispatcher routing logic and action contracts.
No LLM calls, no DB, no vector store — all mocked.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from functools import partial

from core.models import (
    Chunk, EvalResult, EvalScore, GenerationResult,
    RetrievalResult, RetrievedChunk, Verdict,
)
from healing.actions import ActionResult, quarantine_chunk, re_retrieve
from healing.dispatcher import dispatch


# ── fixtures ─────────────────────────────────────────────────────────

def _chunk(chunk_id="c001", quality=0.8):
    return Chunk(
        chunk_id=chunk_id, content="Test chunk content",
        source="test.txt", doc_type="text", chunk_index=0,
        quality_score=quality, failure_count=0,
    )

def _rc(chunk_id="c001", quality=0.8, score=0.85):
    return RetrievedChunk(chunk=_chunk(chunk_id, quality), score=score, rank=0)

def _eval(verdict, failed_judges, weighted=0.5):
    scores = []
    for name in ["faithfulness", "relevance", "grounding"]:
        passed = name not in failed_judges
        s = 0.9 if passed else 0.4
        scores.append(EvalScore(judge_name=name, score=s, passed=passed))

    retrieval = RetrievalResult(
        query="test query", chunks=[_rc()], context_text="some context"
    )
    generation = GenerationResult(
        answer="test answer", query="test query", context_text="some context",
        sources=[], model="test",
    )
    return EvalResult(
        verdict=verdict,
        scores=scores,
        weighted_score=weighted,
        generation=generation,
        retrieval=retrieval,
        metadata={"failed_judges": failed_judges},
    )


# ── dispatcher routing ────────────────────────────────────────────────

class TestDispatcher:
    def _deps(self):
        return MagicMock(), MagicMock(), MagicMock()  # pipeline, store, embedder

    def test_pass_returns_empty_list(self):
        ev = _eval(Verdict.PASS, [])
        pipeline, store, embedder = self._deps()
        actions = dispatch(ev, pipeline, store, embedder)
        assert actions == []

    def test_soft_fail_relevance_dispatches_expand_query(self):
        ev = _eval(Verdict.SOFT_FAIL, ["relevance"])
        pipeline, store, embedder = self._deps()
        actions = dispatch(ev, pipeline, store, embedder)
        names = [a.func.__name__ for a in actions]
        assert "expand_query" in names

    def test_soft_fail_grounding_dispatches_re_retrieve(self):
        ev = _eval(Verdict.SOFT_FAIL, ["grounding"])
        pipeline, store, embedder = self._deps()
        actions = dispatch(ev, pipeline, store, embedder)
        names = [a.func.__name__ for a in actions]
        assert "re_retrieve" in names

    def test_hard_fail_faithfulness_dispatches_quarantine_then_expand(self):
        ev = _eval(Verdict.HARD_FAIL, ["faithfulness"])
        pipeline, store, embedder = self._deps()
        actions = dispatch(ev, pipeline, store, embedder)
        names = [a.func.__name__ for a in actions]
        assert names[0] == "quarantine_chunk"
        assert "expand_query" in names

    def test_hard_fail_grounding_and_relevance_dispatches_re_retrieve(self):
        ev = _eval(Verdict.HARD_FAIL, ["grounding", "relevance"])
        pipeline, store, embedder = self._deps()
        actions = dispatch(ev, pipeline, store, embedder)
        names = [a.func.__name__ for a in actions]
        assert "re_retrieve" in names

    def test_all_actions_are_zero_arg_callables(self):
        """Every returned action must be callable with zero args."""
        for verdict, failed in [
            (Verdict.SOFT_FAIL, ["relevance"]),
            (Verdict.HARD_FAIL, ["faithfulness"]),
            (Verdict.HARD_FAIL, ["grounding", "relevance"]),
        ]:
            ev = _eval(verdict, failed)
            pipeline, store, embedder = self._deps()
            actions = dispatch(ev, pipeline, store, embedder)
            for a in actions:
                assert callable(a), f"Action {a} is not callable"


# ── action contracts ──────────────────────────────────────────────────

class TestQuarantineChunk:
    def test_marks_chunk_healed_in_store(self):
        store = MagicMock()
        ev = _eval(Verdict.HARD_FAIL, ["faithfulness"])
        result = quarantine_chunk(ev, store)
        assert result.action == "quarantine_chunk"
        assert result.success is True
        assert len(result.affected_chunk_ids) > 0
        store.update_chunk_metadata.assert_called()

    def test_decrements_quality_score(self):
        store = MagicMock()
        ev = _eval(Verdict.HARD_FAIL, ["faithfulness"], weighted=0.3)
        # Inject a chunk with known quality score
        ev.retrieval.chunks[0].chunk.quality_score = 0.8
        quarantine_chunk(ev, store, quality_penalty=0.2)
        # Check the update call had new_quality = 0.8 - 0.2 = 0.6
        call_kwargs = store.update_chunk_metadata.call_args[0][1]
        assert call_kwargs["quality_score"] == pytest.approx(0.6)
        assert call_kwargs["heal_flag"] is True

    def test_no_chunks_returns_failure(self):
        store = MagicMock()
        ev = _eval(Verdict.HARD_FAIL, ["faithfulness"])
        ev.retrieval.chunks = []  # strip all chunks
        result = quarantine_chunk(ev, store)
        assert result.success is False
        store.update_chunk_metadata.assert_not_called()

    def test_store_exception_returns_failure_not_raise(self):
        store = MagicMock()
        store.update_chunk_metadata.side_effect = RuntimeError("DB error")
        ev = _eval(Verdict.HARD_FAIL, ["faithfulness"])
        result = quarantine_chunk(ev, store)
        assert result.success is False
        assert "error" in result.details


class TestReRetrieve:
    def test_restores_original_k_after_success(self):
        pipeline = MagicMock()
        pipeline.retriever.k = 10
        new_ret = RetrievalResult(
            query="q", chunks=[_rc()], context_text="ctx"
        )
        pipeline.run.return_value = new_ret
        ev = _eval(Verdict.SOFT_FAIL, ["grounding"])

        result = re_retrieve(ev, pipeline)
        assert result.success is True
        assert pipeline.retriever.k == 10  # must be restored

    def test_restores_original_k_after_failure(self):
        pipeline = MagicMock()
        pipeline.retriever.k = 10
        pipeline.run.side_effect = RuntimeError("retrieval failed")
        ev = _eval(Verdict.SOFT_FAIL, ["grounding"])

        result = re_retrieve(ev, pipeline)
        assert result.success is False
        assert pipeline.retriever.k == 10  # must still be restored

    def test_boosts_k_during_call(self):
        pipeline = MagicMock()
        pipeline.retriever.k = 10
        captured_k = []

        def capture_run(query):
            captured_k.append(pipeline.retriever.k)
            return RetrievalResult(query=query, chunks=[_rc()], context_text="ctx")

        pipeline.run.side_effect = capture_run
        ev = _eval(Verdict.SOFT_FAIL, ["grounding"])
        re_retrieve(ev, pipeline)
        assert captured_k[0] > 10   # k was boosted during the call

    def test_new_retrieval_is_returned(self):
        pipeline = MagicMock()
        pipeline.retriever.k = 10
        new_ret = RetrievalResult(query="q", chunks=[_rc()], context_text="new ctx")
        pipeline.run.return_value = new_ret
        ev = _eval(Verdict.SOFT_FAIL, ["grounding"])
        result = re_retrieve(ev, pipeline)
        assert result.new_retrieval is new_ret