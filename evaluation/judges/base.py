"""
self-healing-rag/evaluation/judges/base.py
Abstract base class for evaluation judges.

Design contract:
- Every judge MUST return an EvalScore, never raise to the caller.
- On any internal failure (LLM error, malformed JSON, etc.) the judge
  returns a score of 0.0 with passed=False and the error recorded in
  `details["error"]` — this is a deliberate fail-closed design: a judge
  that cannot evaluate is treated as a failure, not silently skipped,
  so the healing pipeline always has a defined verdict to act on.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.logger import logger
from core.models import EvalScore, GenerationResult, RetrievalResult


class BaseJudge(ABC):
    """All judges implement `name` and `evaluate()`."""

    name: str = "base_judge"
    threshold: float = 0.5

    @abstractmethod
    def _score(
        self, retrieval: RetrievalResult, generation: GenerationResult
    ) -> tuple[float, dict]:
        """
        Compute the raw score in [0.0, 1.0] plus any extra details.
        Subclasses implement this; may raise — evaluate() catches it.
        """
        raise NotImplementedError

    def evaluate(
        self, retrieval: RetrievalResult, generation: GenerationResult
    ) -> EvalScore:
        try:
            score, details = self._score(retrieval, generation)
            score = max(0.0, min(1.0, float(score)))  # clamp defensively
            passed = score >= self.threshold
            return EvalScore(
                judge_name=self.name,
                score=round(score, 4),
                passed=passed,
                details=details,
            )
        except Exception as exc:
            logger.error(f"Judge '{self.name}' failed: {exc}")
            return EvalScore(
                judge_name=self.name,
                score=0.0,
                passed=False,
                details={"error": str(exc)},
            )