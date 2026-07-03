"""
self-healing-rag/healing/__init__.py
HealingLoop: the Phase 3 core controller.

Flow per query
──────────────
1.  EvaluationPipeline runs → EvalResult with PASS / SOFT_FAIL / HARD_FAIL
2.  If PASS → done immediately (no healing needed)
3.  Dispatcher selects an ordered action list for the verdict
4.  For each action (up to max_heal_rounds total):
      a.  Run the action → ActionResult
      b.  If action produced a new RetrievalResult:
            - Re-generate the answer with the new context
            - Re-evaluate the new answer
            - Log the HealEvent to DB
            - If new verdict == PASS → return healed PipelineResponse
      c.  If action did NOT produce new retrieval (e.g. quarantine, re_embed):
            - Log the event (chunk was mutated in the store)
            - Continue to the next action
5.  After all rounds exhausted → return best result seen so far
    (lowest weighted_score failure is still better than a crash)

Key safety contracts
────────────────────
- max_heal_rounds is read from config; default 3.
- Each round increments the round counter regardless of action type.
- DB logging never raises (FeedbackLogger is fail-safe).
- Generator and Evaluator errors abort that round but do not crash the loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logger import logger
from core.models import (
    EvalResult,
    GenerationResult,
    RetrievalResult,
    Verdict,
)
from evaluation import EvaluationPipeline
from generation.generator import Generator
from healing.dispatcher import dispatch
from healing.feedback_logger import FeedbackLogger
from healing.actions import ActionResult

if TYPE_CHECKING:
    from ingestion.embedder import Embedder
    from ingestion.vector_store import BaseVectorStore
    from retrieval import RetrievalPipeline

settings = get_settings()


@dataclass
class HealingResult:
    """What the HealingLoop returns — richer than PipelineResponse for logging."""
    retrieval: RetrievalResult
    generation: GenerationResult
    eval_result: EvalResult
    heal_rounds: int = 0
    healed: bool = False           # True if final verdict == PASS
    actions_taken: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0


class HealingLoop:
    """
    Usage:
        loop = HealingLoop(
            retrieval_pipeline=retrieval,
            generator=generator,
            evaluator=evaluator,
            store=store,
            embedder=embedder,
        )
        result = await loop.run(initial_retrieval, initial_generation, session)
    """

    def __init__(
        self,
        retrieval_pipeline: "RetrievalPipeline",
        generator: Generator,
        evaluator: EvaluationPipeline,
        store: "BaseVectorStore",
        embedder: "Embedder",
        max_rounds: int | None = None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator
        self.evaluator = evaluator
        self.store = store
        self.embedder = embedder
        self.max_rounds = (
            max_rounds
            if max_rounds is not None
            else settings.evaluation_cfg.get("max_heal_rounds", 3)
        )

    async def run(
        self,
        initial_retrieval: RetrievalResult,
        initial_generation: GenerationResult,
        session: AsyncSession,
    ) -> HealingResult:
        t0 = time.perf_counter()
        fb = FeedbackLogger(session)

        # ── Step 1: initial evaluation ────────────────────────────────
        eval_result = self.evaluator.run(initial_retrieval, initial_generation)

        best_retrieval = initial_retrieval
        best_generation = initial_generation
        best_eval = eval_result
        actions_taken: list[str] = []
        heal_rounds = 0

        # ── Step 2: early exit if already PASS ───────────────────────
        if eval_result.verdict == Verdict.PASS:
            await fb.log_query(
                initial_retrieval,
                initial_generation,
                eval_result,
                heal_rounds=0,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return HealingResult(
                retrieval=initial_retrieval,
                generation=initial_generation,
                eval_result=eval_result,
                heal_rounds=0,
                healed=True,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── Step 3: healing loop ──────────────────────────────────────
        # Persist the QueryLog row NOW (before healing) so HealEvents can
        # reference it via FK. We'll update it after healing via a second flush.
        query_log_id = await fb.log_query(
            initial_retrieval,
            initial_generation,
            eval_result,
            heal_rounds=0,
            total_latency_ms=0.0,   # updated at the end
        )
        current_eval = eval_result

        for round_num in range(1, self.max_rounds + 1):
            heal_rounds = round_num
            logger.info(
                f"[HealingLoop] Round {round_num}/{self.max_rounds} "
                f"verdict={current_eval.verdict.value} "
                f"failed={current_eval.metadata.get('failed_judges', [])}"
            )

            # Get actions for current eval state
            actions = dispatch(
                current_eval,
                self.retrieval_pipeline,
                self.store,
                self.embedder,
            )

            if not actions:
                logger.warning("[HealingLoop] Dispatcher returned no actions — stopping")
                break

            round_healed = False

            for action_fn in actions:
                action_name = action_fn.func.__name__
                actions_taken.append(f"r{round_num}:{action_name}")

                try:
                    action_result = action_fn()
                except Exception as exc:
                    logger.error(f"[HealingLoop] action {action_name} raised: {exc}")
                    await fb.log_heal_event(
                        query_log_id, current_eval,
                        ActionResult(
                            action=action_name, success=False, details={"error": str(exc)}
                        ),
                        round_number=round_num,
                    )
                    continue

                if not action_result.success:
                    logger.warning(
                        f"[HealingLoop] action {action_name} reported failure: "
                        f"{action_result.details}"
                    )
                    await fb.log_heal_event(
                        query_log_id, current_eval, action_result, round_number=round_num
                    )
                    continue

                # ── If the action produced new retrieval → re-generate + re-eval ─
                if action_result.new_retrieval is not None:
                    new_retrieval = action_result.new_retrieval
                    try:
                        new_generation = self.generator.generate(new_retrieval)
                    except Exception as exc:
                        logger.error(f"[HealingLoop] re-generation failed: {exc}")
                        await fb.log_heal_event(
                            query_log_id, current_eval, action_result, round_number=round_num
                        )
                        continue

                    new_eval = self.evaluator.run(new_retrieval, new_generation)
                    await fb.log_heal_event(
                        query_log_id, current_eval, action_result,
                        round_number=round_num, eval_after=new_eval
                    )

                    # Track the best result seen so far
                    if new_eval.weighted_score > best_eval.weighted_score:
                        best_retrieval = new_retrieval
                        best_generation = new_generation
                        best_eval = new_eval

                    current_eval = new_eval

                    if new_eval.verdict == Verdict.PASS:
                        logger.info(
                            f"[HealingLoop] PASS achieved after round {round_num} "
                            f"action={action_name} weighted_score={new_eval.weighted_score:.3f}"
                        )
                        round_healed = True
                        break  # stop trying more actions in this round

                else:
                    # Chunk-level action (quarantine / re_embed) — log it but no
                    # new retrieval yet; the next action or next round will re-retrieve.
                    await fb.log_heal_event(
                        query_log_id, current_eval, action_result, round_number=round_num
                    )

            if round_healed:
                break  # stop further rounds too

        # ── Step 4: finalise ─────────────────────────────────────────
        total_ms = (time.perf_counter() - t0) * 1000
        healed = best_eval.verdict == Verdict.PASS
        if not healed:
            logger.warning(
                f"[HealingLoop] Exhausted {heal_rounds} round(s) without reaching PASS. "
                f"Best verdict={best_eval.verdict.value} score={best_eval.weighted_score:.3f}"
            )

        return HealingResult(
            retrieval=best_retrieval,
            generation=best_generation,
            eval_result=best_eval,
            heal_rounds=heal_rounds,
            healed=healed,
            actions_taken=actions_taken,
            total_latency_ms=total_ms,
        )