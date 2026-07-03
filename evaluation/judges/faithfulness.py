"""
self-healing-rag/evaluation/judges/faithfulness.py
Faithfulness judge: does the generated answer's claims hold up against
the retrieved context? (a.k.a. hallucination detection)

Implemented as LLM-as-judge rather than the `ragas` library directly —
ragas pins specific langchain versions that can conflict with Phase 1's
dependency set, and an LLM-as-judge call against our own provider
abstraction (core/llm_client.py) keeps Phase 2 dependency-free and
provider-agnostic (works identically with OpenAI/Anthropic/Ollama).
"""
from __future__ import annotations

from core.config import get_settings
from core.llm_client import call_llm_json
from core.logger import logger
from core.models import GenerationResult, RetrievalResult
from evaluation.judges.base import BaseJudge

settings = get_settings()


class FaithfulnessJudge(BaseJudge):
    name = "faithfulness"

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = (
            threshold
            if threshold is not None
            else settings.evaluation_cfg["faithfulness_threshold"]
        )

    def _score(
        self, retrieval: RetrievalResult, generation: GenerationResult
    ) -> tuple[float, dict]:
        # Guard: an empty answer or empty context cannot be evaluated meaningfully.
        if not generation.answer.strip():
            return 0.0, {"error": "empty_answer"}
        if not retrieval.context_text.strip():
            # No context retrieved at all → answer cannot be grounded by definition.
            return 0.0, {"error": "empty_context"}

        prompts = settings.prompts
        result = call_llm_json(
            system=prompts["faithfulness_system"],
            user=prompts["faithfulness_user"].format(
                context=retrieval.context_text,
                answer=generation.answer,
            ),
            temperature=0.0,
            max_tokens=500,
        )

        score = float(result.get("score", 0.0))
        details = {
            "unsupported_claims": result.get("unsupported_claims", []),
            "reasoning": result.get("reasoning", ""),
        }

        if details["unsupported_claims"]:
            logger.warning(
                f"Faithfulness judge flagged {len(details['unsupported_claims'])} "
                f"unsupported claim(s): {details['unsupported_claims']}"
            )

        return score, details