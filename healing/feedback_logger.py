"""
self-healing-rag/healing/feedback_logger.py
Async logger that persists QueryLog and HealEvent rows.

Design
------
- All DB calls are async (aiosqlite + SQLAlchemy async).
- Logging failure must NEVER crash the pipeline — every public method
  wraps its body in try/except and logs the error instead of re-raising.
- Returns the created ORM row IDs so the HealingLoop can link HealEvents
  to their parent QueryLog.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.logger import logger
from core.models import EvalResult, GenerationResult, RetrievalResult
from db.models import HealEvent, QueryLog
from healing.actions import ActionResult


class FeedbackLogger:
    """
    Usage (inside an async context with an AsyncSession available):

        logger = FeedbackLogger(session)
        qlog_id = await logger.log_query(retrieval, generation, eval_result, heal_rounds=1)
        await logger.log_heal_event(qlog_id, eval_result, action_result, round_number=1)
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_query(
        self,
        retrieval: RetrievalResult,
        generation: GenerationResult,
        eval_result: EvalResult | None,
        heal_rounds: int = 0,
        total_latency_ms: float = 0.0,
    ) -> int | None:
        """
        Persist a QueryLog row.
        Returns the new row id, or None on failure.
        """
        try:
            judge_scores_json: list[dict[str, Any]] | None = None
            verdict: str | None = None
            weighted_score: float | None = None

            if eval_result is not None:
                verdict = eval_result.verdict.value
                weighted_score = eval_result.weighted_score
                judge_scores_json = [
                    {
                        "judge": s.judge_name,
                        "score": s.score,
                        "passed": s.passed,
                        "details": s.details,
                    }
                    for s in eval_result.scores
                ]

            row = QueryLog(
                query=retrieval.query,
                answer=generation.answer,
                model=generation.model,
                verdict=verdict,
                weighted_score=weighted_score,
                judge_scores=judge_scores_json,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                latency_ms=generation.latency_ms,
                total_latency_ms=total_latency_ms,
                heal_rounds=heal_rounds,
            )
            self._session.add(row)
            await self._session.flush()  # gets the auto-assigned id without full commit
            logger.debug(f"QueryLog persisted: id={row.id} verdict={verdict}")
            return row.id
        except Exception as exc:
            logger.error(f"FeedbackLogger.log_query failed: {exc}")
            return None

    async def log_heal_event(
        self,
        query_log_id: int | None,
        eval_before: EvalResult,
        action_result: ActionResult,
        round_number: int,
        eval_after: EvalResult | None = None,
    ) -> None:
        """
        Persist a HealEvent row linked to a QueryLog.
        """
        try:
            healed = (
                eval_after is not None
                and eval_after.verdict.value == "PASS"
            )
            row = HealEvent(
                query_log_id=query_log_id,
                query=eval_before.retrieval.query,
                verdict_before=eval_before.verdict.value,
                weighted_score_before=eval_before.weighted_score,
                failed_judges=eval_before.metadata.get("failed_judges", []),
                action=action_result.action,
                round_number=round_number,
                # Best-effort: grab first affected chunk's id/source if available
                chunk_id=action_result.affected_chunk_ids[0]
                if action_result.affected_chunk_ids
                else None,
                chunk_source=(
                    eval_before.retrieval.chunks[0].chunk.source
                    if eval_before.retrieval.chunks
                    else None
                ),
                verdict_after=eval_after.verdict.value if eval_after else None,
                weighted_score_after=eval_after.weighted_score if eval_after else None,
                healed=healed,
                extra={
                    "action_success": action_result.success,
                    "action_details": action_result.details,
                    "affected_chunk_ids": action_result.affected_chunk_ids,
                },
            )
            self._session.add(row)
            await self._session.flush()
            logger.debug(
                f"HealEvent persisted: id={row.id} action={row.action} "
                f"round={round_number} healed={healed}"
            )
        except Exception as exc:
            logger.error(f"FeedbackLogger.log_heal_event failed: {exc}")