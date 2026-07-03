"""
self-healing-rag/evaluation/judges/grounding.py
Grounding / context-precision judge: what fraction of the RETRIEVED
context is actually relevant to the question? Catches noisy retrieval
even when the generator manages to produce a reasonable answer anyway.
"""
from __future__ import annotations

from core.config import get_settings
from core.llm_client import call_llm_json
from core.logger import logger
from core.models import GenerationResult, RetrievalResult
from evaluation.judges.base import BaseJudge

settings = get_settings()


class GroundingJudge(BaseJudge):
    name = "grounding"

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = (
            threshold
            if threshold is not None
            else settings.evaluation_cfg["grounding_threshold"]
        )

    def _score(
        self, retrieval: RetrievalResult, generation: GenerationResult
    ) -> tuple[float, dict]:
        if not retrieval.context_text.strip():
            # Nothing was retrieved — there is nothing to be "precise" about.
            # This is a HARD signal that retrieval itself failed, not a
            # generation quality issue, so we score 0.0 rather than skipping.
            return 0.0, {"error": "empty_context", "irrelevant_source_count": 0}

        prompts = settings.prompts
        result = call_llm_json(
            system=prompts["grounding_system"],
            user=prompts["grounding_user"].format(
                query=retrieval.query,
                context=retrieval.context_text,
            ),
            temperature=0.0,
            max_tokens=350,
        )

        score = float(result.get("score", 0.0))
        details = {
            "irrelevant_source_count": int(result.get("irrelevant_source_count", 0)),
            "reasoning": result.get("reasoning", ""),
            "num_chunks_retrieved": len(retrieval.chunks),
        }
        return score, details