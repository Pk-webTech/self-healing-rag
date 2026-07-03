"""
self-healing-rag/api/routers/query.py
POST /query — full RAG pipeline:
  Phase 1: Retrieve + Generate
  Phase 2: Evaluate (judges)
  Phase 3: Heal (self-healing loop)
  Phase 4: Adapt (quality tracker, tuner, optimizer)
  Phase 5: Observe (metrics, traces, alerts)
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from adaptation import AdaptiveLearner
from api.deps import (
    get_adaptive_learner,
    get_evaluation_pipeline,
    get_generation_pipeline,
    get_healing_loop,
    get_observability_manager,
    get_retrieval_pipeline,
)
from api.schemas import ChunkInfo, JudgeScoreInfo, QueryRequest, QueryResponse
from core.logger import logger
from db.session import get_session
from evaluation import EvaluationPipeline
from generation.generator import Generator
from healing import HealingLoop
from observability import ObservabilityManager
from retrieval import RetrievalPipeline

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    retrieval: RetrievalPipeline = Depends(get_retrieval_pipeline),
    generator: Generator = Depends(get_generation_pipeline),
    evaluator: EvaluationPipeline = Depends(get_evaluation_pipeline),
    healing_loop: HealingLoop = Depends(get_healing_loop),
    learner: AdaptiveLearner = Depends(get_adaptive_learner),
    obs: ObservabilityManager = Depends(get_observability_manager),
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    """
    Full RAG pipeline — Phases 1–5.
    """
    t0 = time.perf_counter()
    eval_result = None
    heal_rounds = 0
    actions_taken: list[str] = []

    try:
        retrieval.retriever.k = body.k
        retrieval_result = retrieval.run(body.query, filters=body.filters)

        if not retrieval_result.chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please ingest documents first.",
            )

        gen_result = generator.generate(retrieval_result)

        verdict = None
        weighted_score = None
        judge_scores = None

        if body.evaluate:
            # Phase 3: healing loop (Phase 2 evaluation is inside it)
            healing_result = await healing_loop.run(retrieval_result, gen_result, session)
            retrieval_result = healing_result.retrieval
            gen_result = healing_result.generation
            eval_result = healing_result.eval_result
            heal_rounds = healing_result.heal_rounds
            actions_taken = healing_result.actions_taken

            verdict = eval_result.verdict.value
            weighted_score = eval_result.weighted_score
            judge_scores = [
                JudgeScoreInfo(
                    judge_name=s.judge_name,
                    score=s.score,
                    passed=s.passed,
                    details=s.details,
                )
                for s in eval_result.scores
            ]

            # Phase 4: adaptive learning (fail-safe)
            try:
                await learner.run(eval_result, session)
            except Exception as adapt_exc:
                logger.error(f"[query] AdaptiveLearner failed (non-fatal): {adapt_exc}")

        total_latency = (time.perf_counter() - t0) * 1000

        # Phase 5: observability (fail-safe — must never affect the response)
        try:
            obs.record(
                retrieval=retrieval_result,
                generation=gen_result,
                eval_result=eval_result,
                total_latency_ms=total_latency,
                heal_rounds=heal_rounds,
                actions_taken=actions_taken,
            )
        except Exception as obs_exc:
            logger.error(f"[query] ObservabilityManager failed (non-fatal): {obs_exc}")

        chunks_info = [
            ChunkInfo(
                chunk_id=rc.chunk.chunk_id,
                source=rc.chunk.source,
                score=round(rc.score, 4),
                rank=rc.rank,
                quality_score=rc.chunk.quality_score,
                token_count=rc.chunk.metadata.get("token_count"),
            )
            for rc in retrieval_result.chunks
        ]

        return QueryResponse(
            query=body.query,
            answer=gen_result.answer,
            sources=gen_result.sources,
            chunks=chunks_info,
            model=gen_result.model,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            latency_ms=round(gen_result.latency_ms, 1),
            total_latency_ms=round(total_latency, 1),
            heal_rounds=heal_rounds,
            verdict=verdict,
            weighted_score=weighted_score,
            judge_scores=judge_scores,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))