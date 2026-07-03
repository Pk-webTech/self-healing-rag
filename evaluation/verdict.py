"""
self-healing-rag/evaluation/verdict.py
Aggregates individual judge scores into a single Verdict.

Verdict rules (deliberately asymmetric — faithfulness failures are the
most severe class because they represent actual hallucination, not just
suboptimal retrieval):

  HARD_FAIL  if faithfulness fails (score < faithfulness_threshold)
             OR 2+ judges fail simultaneously
  SOFT_FAIL  if exactly 1 non-faithfulness judge fails
             (relevance OR grounding, but not both, and faithfulness passed)
  PASS       if all judges pass

weighted_score = 0.5 * faithfulness + 0.25 * relevance + 0.25 * grounding
(faithfulness weighted highest since hallucination is the costliest failure mode)
"""
from __future__ import annotations

from core.logger import logger
from core.models import EvalResult, EvalScore, GenerationResult, RetrievalResult, Verdict

# weight map — must sum to 1.0 (validated below at import time)
JUDGE_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.50,
    "relevance": 0.25,
    "grounding": 0.25,
}

_weight_sum = round(sum(JUDGE_WEIGHTS.values()), 6)
if _weight_sum != 1.0:
    raise ValueError(f"JUDGE_WEIGHTS must sum to 1.0, got {_weight_sum}")


def _compute_weighted_score(scores: list[EvalScore]) -> float:
    total = 0.0
    matched_weight = 0.0
    for s in scores:
        w = JUDGE_WEIGHTS.get(s.judge_name)
        if w is None:
            logger.warning(f"No weight configured for judge '{s.judge_name}', skipping in aggregate")
            continue
        total += w * s.score
        matched_weight += w

    if matched_weight == 0:
        # No recognised judges contributed — cannot trust this aggregate.
        return 0.0

    # Renormalise in case some judges were missing (e.g. only 2 of 3 ran)
    return round(total / matched_weight, 4)


def aggregate(scores: list[EvalScore]) -> Verdict:
    """Apply the verdict rule described in the module docstring."""
    by_name = {s.judge_name: s for s in scores}
    faithfulness = by_name.get("faithfulness")

    failed = [s for s in scores if not s.passed]
    n_failed = len(failed)

    # Faithfulness is a hard gate — hallucination always escalates to HARD_FAIL,
    # regardless of how many other judges passed.
    if faithfulness is not None and not faithfulness.passed:
        return Verdict.HARD_FAIL

    if n_failed >= 2:
        return Verdict.HARD_FAIL

    if n_failed == 1:
        return Verdict.SOFT_FAIL

    return Verdict.PASS


def build_eval_result(
    scores: list[EvalScore],
    generation: GenerationResult,
    retrieval: RetrievalResult,
) -> EvalResult:
    """Combine judge scores into a full EvalResult with verdict + weighted score."""
    if not scores:
        # No judges ran at all — fail closed rather than silently passing.
        logger.error("build_eval_result called with zero scores — failing closed")
        return EvalResult(
            verdict=Verdict.HARD_FAIL,
            scores=[],
            weighted_score=0.0,
            generation=generation,
            retrieval=retrieval,
            metadata={"error": "no_judges_ran"},
        )

    verdict = aggregate(scores)
    weighted = _compute_weighted_score(scores)

    logger.info(
        f"Eval verdict={verdict.value} weighted_score={weighted:.3f} "
        f"scores={ {s.judge_name: s.score for s in scores} }"
    )

    return EvalResult(
        verdict=verdict,
        scores=scores,
        weighted_score=weighted,
        generation=generation,
        retrieval=retrieval,
        metadata={
            "failed_judges": [s.judge_name for s in scores if not s.passed],
        },
    )