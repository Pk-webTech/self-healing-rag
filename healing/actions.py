"""
self-healing-rag/healing/actions.py
Four concrete healing actions that the dispatcher can call.

Action            Targets              When used
─────────────────────────────────────────────────────────────────
expand_query      retrieval            SOFT_FAIL: relevance or grounding fails
                                       — try a broader/rephrased query
re_retrieve       retrieval            SOFT_FAIL: retrieval may have missed
                                       — bump k and re-run the retrieval pipeline
re_embed_chunk    specific chunks      HARD_FAIL: context embeddings are stale/bad
                                       — regenerate embedding for worst-scored chunks
quarantine_chunk  specific chunks      HARD_FAIL: faithfulness fail traced to chunk
                                       — set heal_flag=True, decrement quality_score

Design contract
───────────────
- Every action returns an ActionResult (what happened, new retrieval if any).
- Actions NEVER raise to the caller — they catch internally and mark success=False.
- All vector-store mutations are safe to re-run (idempotent metadata updates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.config import get_settings
from core.logger import logger
from core.models import EvalResult, RetrievalResult, Verdict
from core.llm_client import call_llm

if TYPE_CHECKING:
    from ingestion.embedder import Embedder
    from ingestion.vector_store import BaseVectorStore
    from retrieval import RetrievalPipeline

settings = get_settings()


# ── result container ─────────────────────────────────────────────────

@dataclass
class ActionResult:
    action: str
    success: bool
    new_retrieval: RetrievalResult | None = None  # set when retrieval was re-run
    affected_chunk_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────────────

def _expand_query_text(original_query: str, failed_judges: list[str]) -> str:
    """
    Ask the LLM to produce a broader reformulation that might fix the
    specific failure mode (relevance miss vs grounding noise).
    Falls back to the original query on any error.
    """
    if "relevance" in failed_judges:
        instruction = (
            "The previous answer did not fully address the question. "
            "Rewrite the question to be more specific and retrieval-friendly. "
            "Output ONLY the rewritten question, nothing else."
        )
    else:
        instruction = (
            "The retrieved context was too noisy. "
            "Rewrite the question using more precise, distinctive keywords "
            "so a semantic search returns more targeted passages. "
            "Output ONLY the rewritten question, nothing else."
        )

    try:
        expanded = call_llm(
            system=instruction,
            user=f"Original question: {original_query}",
            temperature=0.3,
            max_tokens=120,
        )
        expanded = expanded.strip().strip('"').strip("'")
        logger.debug(f"Query expanded: {original_query!r} → {expanded!r}")
        return expanded if expanded else original_query
    except Exception as exc:
        logger.warning(f"Query expansion LLM call failed ({exc}), using original")
        return original_query


# ── Action 1: expand_query ───────────────────────────────────────────

def expand_query(
    eval_result: EvalResult,
    retrieval_pipeline: "RetrievalPipeline",
) -> ActionResult:
    """
    Expand/rephrase the query and re-run the full retrieval pipeline.
    Best for SOFT_FAIL caused by relevance or grounding failures.
    """
    failed_judges = eval_result.metadata.get("failed_judges", [])
    original_query = eval_result.retrieval.query

    new_query = _expand_query_text(original_query, failed_judges)
    if new_query == original_query:
        return ActionResult(
            action="expand_query",
            success=False,
            details={"reason": "expansion produced no change"},
        )

    try:
        new_retrieval = retrieval_pipeline.run(new_query)
        logger.info(
            f"[Action:expand_query] re-retrieved {len(new_retrieval.chunks)} chunks "
            f"with expanded query"
        )
        return ActionResult(
            action="expand_query",
            success=True,
            new_retrieval=new_retrieval,
            details={"original_query": original_query, "expanded_query": new_query},
        )
    except Exception as exc:
        logger.error(f"[Action:expand_query] retrieval failed: {exc}")
        return ActionResult(
            action="expand_query",
            success=False,
            details={"error": str(exc)},
        )


# ── Action 2: re_retrieve ────────────────────────────────────────────

def re_retrieve(
    eval_result: EvalResult,
    retrieval_pipeline: "RetrievalPipeline",
    k_multiplier: float = 1.5,
) -> ActionResult:
    """
    Re-run retrieval with a larger k to surface more candidates.
    Best for SOFT_FAIL where the right chunks may just have been below the
    top-k cutoff.
    """
    original_query = eval_result.retrieval.query
    original_k = retrieval_pipeline.retriever.k
    boosted_k = max(original_k + 2, int(original_k * k_multiplier))

    # temporarily bump k — restore after
    retrieval_pipeline.retriever.k = boosted_k
    try:
        new_retrieval = retrieval_pipeline.run(original_query)
        logger.info(
            f"[Action:re_retrieve] k {original_k}→{boosted_k}, "
            f"got {len(new_retrieval.chunks)} chunks"
        )
        return ActionResult(
            action="re_retrieve",
            success=True,
            new_retrieval=new_retrieval,
            details={"original_k": original_k, "boosted_k": boosted_k},
        )
    except Exception as exc:
        logger.error(f"[Action:re_retrieve] failed: {exc}")
        return ActionResult(
            action="re_retrieve",
            success=False,
            details={"error": str(exc)},
        )
    finally:
        # ALWAYS restore original k so the singleton is not permanently mutated
        retrieval_pipeline.retriever.k = original_k


# ── Action 3: re_embed_chunk ─────────────────────────────────────────

def re_embed_chunk(
    eval_result: EvalResult,
    store: "BaseVectorStore",
    embedder: "Embedder",
    max_chunks: int = 2,
) -> ActionResult:
    """
    Re-compute embeddings for the lowest-quality chunks in the last
    retrieval result and upsert them back into the vector store.
    Best for HARD_FAIL where embeddings may have drifted (e.g. after
    an embedding model change or document corruption).
    """
    chunks_to_fix = sorted(
        [rc.chunk for rc in eval_result.retrieval.chunks],
        key=lambda c: c.quality_score,
    )[:max_chunks]

    if not chunks_to_fix:
        return ActionResult(
            action="re_embed_chunk",
            success=False,
            details={"reason": "no chunks in retrieval result to re-embed"},
        )

    affected_ids: list[str] = []
    try:
        re_embedded = embedder.embed_chunks(chunks_to_fix)
        store.add_chunks(re_embedded)  # upsert

        for chunk in re_embedded:
            # Update metadata to mark re-embedding happened
            store.update_chunk_metadata(
                chunk.chunk_id,
                {
                    "last_healed": datetime.now(timezone.utc).isoformat(),
                    "heal_flag": True,
                    "failure_count": chunk.failure_count + 1,
                },
            )
            affected_ids.append(chunk.chunk_id)

        logger.info(
            f"[Action:re_embed_chunk] re-embedded {len(affected_ids)} chunks: {affected_ids}"
        )
        return ActionResult(
            action="re_embed_chunk",
            success=True,
            affected_chunk_ids=affected_ids,
            details={"num_re_embedded": len(affected_ids)},
        )
    except Exception as exc:
        logger.error(f"[Action:re_embed_chunk] failed: {exc}")
        return ActionResult(
            action="re_embed_chunk",
            success=False,
            details={"error": str(exc)},
        )


# ── Action 4: quarantine_chunk ───────────────────────────────────────

def quarantine_chunk(
    eval_result: EvalResult,
    store: "BaseVectorStore",
    max_chunks: int = 2,
    quality_penalty: float = 0.2,
) -> ActionResult:
    """
    Flag the worst chunks as quarantined: set heal_flag=True,
    decrement quality_score, increment failure_count.
    The chunks stay in the store (so queries still work) but Phase 4's
    adaptive learning will later decide whether to re-index or drop them.

    Best for HARD_FAIL caused by faithfulness failure — the answer
    contained a claim not supported by context, which usually means a
    specific chunk fed bad information to the LLM.
    """
    # Sort by quality_score ascending — penalise the worst first
    chunks_to_flag = sorted(
        [rc.chunk for rc in eval_result.retrieval.chunks],
        key=lambda c: c.quality_score,
    )[:max_chunks]

    if not chunks_to_flag:
        return ActionResult(
            action="quarantine_chunk",
            success=False,
            details={"reason": "no chunks available to quarantine"},
        )

    affected_ids: list[str] = []
    try:
        for chunk in chunks_to_flag:
            new_quality = max(0.0, round(chunk.quality_score - quality_penalty, 4))
            new_failure_count = chunk.failure_count + 1
            store.update_chunk_metadata(
                chunk.chunk_id,
                {
                    "heal_flag": True,
                    "quality_score": new_quality,
                    "failure_count": new_failure_count,
                    "last_healed": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.warning(
                f"[Action:quarantine_chunk] chunk {chunk.chunk_id[:8]} "
                f"quality {chunk.quality_score:.3f}→{new_quality:.3f} "
                f"failure_count={new_failure_count}"
            )
            affected_ids.append(chunk.chunk_id)

        return ActionResult(
            action="quarantine_chunk",
            success=True,
            affected_chunk_ids=affected_ids,
            details={
                "num_quarantined": len(affected_ids),
                "quality_penalty": quality_penalty,
            },
        )
    except Exception as exc:
        logger.error(f"[Action:quarantine_chunk] failed: {exc}")
        return ActionResult(
            action="quarantine_chunk",
            success=False,
            details={"error": str(exc)},
        )