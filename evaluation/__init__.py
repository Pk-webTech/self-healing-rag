"""
self-healing-rag/evaluation/__init__.py
Evaluation pipeline: run all judges in parallel-ish fashion (sequential
calls, but each is independently fail-safe) and aggregate into a verdict.
"""
from __future__ import annotations

import time

from core.logger import logger
from core.models import EvalResult, GenerationResult, RetrievalResult
from evaluation.judges.base import BaseJudge
from evaluation.judges.faithfulness import FaithfulnessJudge
from evaluation.judges.grounding import GroundingJudge
from evaluation.judges.relevance import RelevanceJudge
from evaluation.verdict import build_eval_result


class EvaluationPipeline:
    """
    Usage:
        pipeline = EvaluationPipeline()
        eval_result = pipeline.run(retrieval_result, generation_result)
        if eval_result.verdict != Verdict.PASS:
            ... trigger healing (Phase 3) ...
    """

    def __init__(self, judges: list[BaseJudge] | None = None) -> None:
        self.judges: list[BaseJudge] = judges or [
            FaithfulnessJudge(),
            RelevanceJudge(),
            GroundingJudge(),
        ]

    def run(
        self,
        retrieval: RetrievalResult,
        generation: GenerationResult,
    ) -> EvalResult:
        t0 = time.perf_counter()
        scores = []
        for judge in self.judges:
            score = judge.evaluate(retrieval, generation)
            scores.append(score)
            logger.debug(
                f"Judge '{score.judge_name}' → score={score.score} passed={score.passed}"
            )

        result = build_eval_result(scores, generation, retrieval)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result.metadata["eval_latency_ms"] = round(elapsed_ms, 1)
        return result