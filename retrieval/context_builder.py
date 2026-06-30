"""
self-healing-rag/retrieval/context_builder.py
Assembles a context string from re-ranked chunks respecting a token budget.
Produces a structured context block with source citations.
"""
from __future__ import annotations

import tiktoken

from core.config import get_settings
from core.logger import logger
from core.models import RetrievalResult, RetrievedChunk

settings = get_settings()
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _format_chunk(rc: RetrievedChunk, idx: int) -> str:
    source = rc.chunk.metadata.get("filename") or rc.chunk.source
    return (
        f"[Source {idx}: {source} | score={rc.score:.3f}]\n"
        f"{rc.chunk.content}"
    )


class ContextBuilder:
    """
    Builds a single context string from a ranked list of RetrievedChunks,
    staying within a token budget.

    Usage:
        builder = ContextBuilder()
        result = builder.build(query, reranked_chunks)
        # result.context_text → string passed to LLM
    """

    def __init__(self, token_budget: int | None = None) -> None:
        self.token_budget = token_budget or settings.generation_cfg.get("token_budget", 3000)

    def build(self, query: str, chunks: list[RetrievedChunk]) -> RetrievalResult:
        used_tokens = 0
        selected: list[RetrievedChunk] = []
        parts: list[str] = []

        for rc in chunks:
            block = _format_chunk(rc, len(selected) + 1)
            block_tokens = _count_tokens(block)
            if used_tokens + block_tokens > self.token_budget:
                logger.debug(
                    f"Token budget {self.token_budget} reached after "
                    f"{len(selected)} chunks ({used_tokens} tokens)"
                )
                break
            selected.append(rc)
            parts.append(block)
            used_tokens += block_tokens

        context_text = "\n\n---\n\n".join(parts)
        logger.debug(
            f"Context built: {len(selected)} chunks, {used_tokens} tokens"
        )

        return RetrievalResult(
            query=query,
            chunks=selected,
            context_text=context_text,
            total_tokens=used_tokens,
            metadata={
                "num_chunks": len(selected),
                "token_budget": self.token_budget,
            },
        )