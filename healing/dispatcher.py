"""
self-healing-rag/healing/dispatcher.py
Routes an EvalResult to the appropriate ordered list of healing actions.

Routing table
─────────────────────────────────────────────────────────────────────────
Verdict      Failed judges              Action sequence
─────────────────────────────────────────────────────────────────────────
PASS         —                          [] (nothing to do)
SOFT_FAIL    relevance only             [expand_query]
SOFT_FAIL    grounding only             [re_retrieve]
SOFT_FAIL    relevance + grounding*     [expand_query, re_retrieve]   (* shouldn't
             (* if both fail → HARD_FAIL via verdict rules, but belt+suspenders)  happen)
HARD_FAIL    faithfulness               [quarantine_chunk, expand_query]
HARD_FAIL    grounding + relevance      [re_retrieve, re_embed_chunk]
HARD_FAIL    any other combo            [re_embed_chunk, expand_query]
─────────────────────────────────────────────────────────────────────────

The dispatcher returns a list of callables (partial-bound to their
non-pipeline arguments) so the HealingLoop can call them in order,
stopping as soon as one produces a new RetrievalResult that passes.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from core.logger import logger
from core.models import EvalResult, Verdict
from healing.actions import (
    ActionResult,
    expand_query,
    quarantine_chunk,
    re_embed_chunk,
    re_retrieve,
)

if TYPE_CHECKING:
    from ingestion.embedder import Embedder
    from ingestion.vector_store import BaseVectorStore
    from retrieval import RetrievalPipeline


# Type alias for a zero-argument healing callable
HealingAction = Callable[[], ActionResult]


def dispatch(
    eval_result: EvalResult,
    retrieval_pipeline: "RetrievalPipeline",
    store: "BaseVectorStore",
    embedder: "Embedder",
) -> list[HealingAction]:
    """
    Return an ordered list of zero-argument callables.
    The HealingLoop calls them left-to-right, stopping on first success.
    """
    verdict = eval_result.verdict
    failed = set(eval_result.metadata.get("failed_judges", []))

    if verdict == Verdict.PASS:
        return []

    actions: list[HealingAction] = []

    if verdict == Verdict.SOFT_FAIL:
        if "relevance" in failed:
            actions.append(partial(expand_query, eval_result, retrieval_pipeline))
        if "grounding" in failed:
            actions.append(partial(re_retrieve, eval_result, retrieval_pipeline))
        # Fallback — shouldn't happen per verdict rules but be defensive
        if not actions:
            actions.append(partial(re_retrieve, eval_result, retrieval_pipeline))

    elif verdict == Verdict.HARD_FAIL:
        if "faithfulness" in failed:
            # Faithfulness failure = answer hallucinated claims.
            # First quarantine the likely-bad chunk, then try a fresh expanded query.
            actions.append(partial(quarantine_chunk, eval_result, store))
            actions.append(partial(expand_query, eval_result, retrieval_pipeline))
        elif "grounding" in failed and "relevance" in failed:
            # Both retrieval-quality judges failed — retrieval itself is broken.
            actions.append(partial(re_retrieve, eval_result, retrieval_pipeline))
            actions.append(partial(re_embed_chunk, eval_result, store, embedder))
        else:
            # Generic hard fail — re-embed stale chunks + expand query
            actions.append(partial(re_embed_chunk, eval_result, store, embedder))
            actions.append(partial(expand_query, eval_result, retrieval_pipeline))

    logger.debug(
        f"Dispatcher: verdict={verdict.value} failed={failed} "
        f"→ {[a.func.__name__ for a in actions]}"
    )
    return actions