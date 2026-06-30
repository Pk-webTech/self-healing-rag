"""
self-healing-rag/api/routers/query.py
POST /query — run the full RAG pipeline for a user query
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_generation_pipeline, get_retrieval_pipeline
from api.schemas import ChunkInfo, QueryRequest, QueryResponse
from core.models import PipelineResponse
from generation.generator import Generator
from retrieval import RetrievalPipeline

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    retrieval: RetrievalPipeline = Depends(get_retrieval_pipeline),
    generator: Generator = Depends(get_generation_pipeline),
) -> QueryResponse:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks (hybrid dense + BM25)
    2. Re-rank with cross-encoder
    3. Build context within token budget
    4. Generate answer with LLM
    """
    t0 = time.perf_counter()
    try:
        # Override pipeline settings from request
        retrieval.retriever.k = body.k
        if hasattr(retrieval.reranker, "model_name"):
            pass  # reranker uses final_k from config; could be extended

        retrieval_result = retrieval.run(body.query, filters=body.filters)

        if not retrieval_result.chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please ingest documents first.",
            )

        gen_result = generator.generate(retrieval_result)
        total_latency = (time.perf_counter() - t0) * 1000

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
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))