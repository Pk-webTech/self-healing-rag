"""
tests/integration/test_healing_loop.py
Integration tests for HealingLoop.
Uses mock DB session, mock evaluator, and mock generator — no real LLM/DB calls.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.models import (
    Chunk, EvalResult, EvalScore, GenerationResult,
    RetrievalResult, RetrievedChunk, Verdict,
)
from healing import HealingLoop
from healing.actions import ActionResult


# ── shared helpers ────────────────────────────────────────────────────

def _chunk(chunk_id="c001", quality=0.8):
    return Chunk(
        chunk_id=chunk_id, content="Relevant test content about RAG",
        source="doc.txt", doc_type="text", chunk_index=0,
        quality_score=quality, failure_count=0,
    )

def _rc(chunk_id="c001", quality=0.8):
    return RetrievedChunk(chunk=_chunk(chunk_id, quality), score=0.85, rank=0)

def _retrieval(query="What is RAG?"):
    return RetrievalResult(
        query=query, chunks=[_rc()],
        context_text="[Source 1: doc.txt | score=0.850]\nRAG is retrieval augmented generation.",
    )

def _generation(answer="RAG stands for Retrieval Augmented Generation."):
    return GenerationResult(
        answer=answer, query="What is RAG?",
        context_text="some context", sources=["doc.txt"],
        model="test-model", prompt_tokens=100, completion_tokens=30,
    )

def _eval_result(verdict, failed_judges, weighted=0.85):
    scores = [
        EvalScore(
            judge_name=name,
            score=0.9 if name not in failed_judges else 0.4,
            passed=name not in failed_judges,
        )
        for name in ["faithfulness", "relevance", "grounding"]
    ]
    return EvalResult(
        verdict=verdict,
        scores=scores,
        weighted_score=weighted,
        generation=_generation(),
        retrieval=_retrieval(),
        metadata={"failed_judges": failed_judges},
    )

def _mock_session():
    """Async mock session — all DB calls are no-ops."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# ── tests ─────────────────────────────────────────────────────────────

class TestHealingLoopPass:
    @pytest.mark.asyncio
    async def test_no_healing_when_pass_on_first_eval(self):
        """If the first evaluation is PASS, healing loop exits immediately."""
        evaluator = MagicMock()
        evaluator.run.return_value = _eval_result(Verdict.PASS, [])

        loop = HealingLoop(
            retrieval_pipeline=MagicMock(),
            generator=MagicMock(),
            evaluator=evaluator,
            store=MagicMock(),
            embedder=MagicMock(),
        )
        session = _mock_session()
        result = await loop.run(_retrieval(), _generation(), session)

        assert result.healed is True
        assert result.heal_rounds == 0
        assert result.actions_taken == []
        evaluator.run.assert_called_once()  # only 1 eval call, no retry


class TestHealingLoopSoftFail:
    @pytest.mark.asyncio
    async def test_soft_fail_heals_on_second_eval(self):
        """SOFT_FAIL triggers one action which produces new retrieval → re-eval → PASS."""
        fail_eval = _eval_result(Verdict.SOFT_FAIL, ["relevance"], weighted=0.55)
        pass_eval = _eval_result(Verdict.PASS, [], weighted=0.92)

        evaluator = MagicMock()
        # First call (initial eval): fail; second call (after re-retrieval): pass
        evaluator.run.side_effect = [fail_eval, pass_eval]

        new_retrieval = _retrieval("What is RAG? (expanded)")
        new_generation = _generation("Expanded answer about RAG.")

        retrieval_pipeline = MagicMock()
        retrieval_pipeline.run.return_value = new_retrieval
        retrieval_pipeline.retriever.k = 10

        generator = MagicMock()
        generator.generate.return_value = new_generation

        loop = HealingLoop(
            retrieval_pipeline=retrieval_pipeline,
            generator=generator,
            evaluator=evaluator,
            store=MagicMock(),
            embedder=MagicMock(),
            max_rounds=3,
        )

        session = _mock_session()

        # Patch dispatch to return a single action that yields new_retrieval.
        # We use a real function (not a partial) so action_fn.func.__name__ works.
        def fake_action():
            return ActionResult(
                action="expand_query",
                success=True,
                new_retrieval=new_retrieval,
                details={"expanded_query": "What is RAG expanded?"},
            )
        fake_action.func = type("F", (), {"__name__": "expand_query"})()

        with patch("healing.dispatch", return_value=[fake_action]):
            result = await loop.run(_retrieval(), _generation(), session)

        assert result.healed is True
        assert result.heal_rounds >= 1
        assert evaluator.run.call_count == 2  # initial + post-heal


class TestHealingLoopMaxRounds:
    @pytest.mark.asyncio
    async def test_exhausts_max_rounds_returns_best_result(self):
        """When all rounds fail, best result is returned without crashing."""
        fail_eval = _eval_result(Verdict.HARD_FAIL, ["faithfulness"], weighted=0.3)

        evaluator = MagicMock()
        evaluator.run.return_value = fail_eval  # always fail

        # Store's update_chunk_metadata is a no-op
        store = MagicMock()

        # All action calls return failed results (no new retrieval)
        with patch("healing.actions.quarantine_chunk") as mock_q:
            mock_q.return_value = ActionResult(
                action="quarantine_chunk", success=True, affected_chunk_ids=["c001"]
            )
            with patch("healing.actions.expand_query") as mock_e:
                mock_e.return_value = ActionResult(
                    action="expand_query", success=False,
                    details={"reason": "no change"}
                )
                loop = HealingLoop(
                    retrieval_pipeline=MagicMock(),
                    generator=MagicMock(),
                    evaluator=evaluator,
                    store=store,
                    embedder=MagicMock(),
                    max_rounds=2,
                )
                session = _mock_session()
                result = await loop.run(_retrieval(), _generation(), session)

        # Must not raise; must return something
        assert result is not None
        assert result.heal_rounds <= 2
        # Best eval is still the failed one (no better result was produced)
        assert result.eval_result.verdict != Verdict.PASS


class TestHealingLoopDBLogging:
    @pytest.mark.asyncio
    async def test_query_log_is_persisted_on_pass(self):
        """DB session.add() must be called at least once (for QueryLog)."""
        evaluator = MagicMock()
        evaluator.run.return_value = _eval_result(Verdict.PASS, [])

        loop = HealingLoop(
            retrieval_pipeline=MagicMock(),
            generator=MagicMock(),
            evaluator=evaluator,
            store=MagicMock(),
            embedder=MagicMock(),
        )
        session = _mock_session()
        await loop.run(_retrieval(), _generation(), session)
        session.add.assert_called()   # QueryLog row added
        session.flush.assert_called()


class TestActionResult:
    def test_action_result_defaults(self):
        ar = ActionResult(action="test", success=True)
        assert ar.new_retrieval is None
        assert ar.affected_chunk_ids == []
        assert ar.details == {}

    def test_action_result_with_new_retrieval(self):
        ret = _retrieval()
        ar = ActionResult(action="re_retrieve", success=True, new_retrieval=ret)
        assert ar.new_retrieval is ret