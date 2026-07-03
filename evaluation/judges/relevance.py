"""
self-healing-rag/evaluation/judges/relevance.py
Answer-relevance judge: does the generated answer actually address
the user's question (independent of factual grounding)?
"""
from __future__ import annotations

from core.config import get_settings
from core.llm_client import call_llm_json
from core.logger import logger
from core.models import GenerationResult, RetrievalResult
from evaluation.judges.base import BaseJudge

settings = get_settings()


class RelevanceJudge(BaseJudge):
    name = "relevance"

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = (
            threshold
            if threshold is not None
            else settings.evaluation_cfg["relevance_threshold"]
        )

    def _score(
        self, retrieval: RetrievalResult, generation: GenerationResult
    ) -> tuple[float, dict]:
        if not generation.answer.strip():
            return 0.0, {"error": "empty_answer"}
        if not retrieval.query.strip():
            return 0.0, {"error": "empty_query"}

        prompts = settings.prompts
        result = call_llm_json(
            system=prompts["relevance_system"],
            user=prompts["relevance_user"].format(
                query=retrieval.query,
                answer=generation.answer,
            ),
            temperature=0.0,
            max_tokens=300,
        )

        score = float(result.get("score", 0.0))
        details = {"reasoning": result.get("reasoning", "")}
        return score, details