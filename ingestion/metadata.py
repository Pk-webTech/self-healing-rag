"""
self-healing-rag/ingestion/metadata.py
Metadata enrichment and initial quality scoring for chunks.

Quality heuristics (Phase 1 baseline):
  - token_density: ratio of alphabetic chars (penalises garbled OCR)
  - length_score:  prefer chunks close to ideal chunk_size
  - info_density:  penalise repetition and very short sentences
"""
from __future__ import annotations

import re
import string
from datetime import datetime

import tiktoken

from core.config import get_settings
from core.logger import logger
from core.models import Chunk

settings = get_settings()
_enc = tiktoken.get_encoding("cl100k_base")


def _alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphabetic."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def _length_score(n_tokens: int, ideal: int) -> float:
    """Gaussian-like score peaking at ideal token count."""
    diff = abs(n_tokens - ideal) / ideal
    return max(0.0, 1.0 - diff)


def _repetition_penalty(text: str) -> float:
    """
    Detect copy-paste / OCR artefacts.
    Low unique-word ratio → penalty.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 5:
        return 0.5
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio  # 1.0 = fully unique, 0.0 = all repeats


def _compute_quality_score(chunk: Chunk) -> float:
    """
    Composite quality score in [0, 1].
    Weights:
      alpha_ratio      0.30
      length_score     0.30
      unique_ratio     0.40
    """
    text = chunk.content
    n_tokens = chunk.metadata.get("token_count") or len(_enc.encode(text))
    ideal = settings.ingestion["chunk_size"]

    alpha = _alpha_ratio(text)
    length = _length_score(n_tokens, ideal)
    unique = _repetition_penalty(text)

    score = 0.30 * alpha + 0.30 * length + 0.40 * unique
    return round(min(max(score, 0.0), 1.0), 4)


class MetadataTagger:
    """
    Enriches chunks with:
    - computed quality_score
    - structured metadata fields (timestamp, source_id, etc.)

    Usage:
        tagger = MetadataTagger()
        chunks = tagger.tag(chunks)
    """

    def tag(self, chunks: list[Chunk]) -> list[Chunk]:
        now = datetime.utcnow().isoformat()
        for chunk in chunks:
            qs = _compute_quality_score(chunk)
            chunk.quality_score = qs
            chunk.metadata.update(
                {
                    "indexed_at": now,
                    "quality_score": qs,
                    "alpha_ratio": round(_alpha_ratio(chunk.content), 4),
                    "unique_word_ratio": round(_repetition_penalty(chunk.content), 4),
                    "heal_flag": False,
                    "failure_count": 0,
                    "retrieval_count": 0,
                }
            )
            if qs < 0.4:
                logger.warning(
                    f"Low quality chunk [{chunk.chunk_id[:8]}] "
                    f"score={qs:.3f} source={chunk.source}"
                )
        logger.debug(f"Tagged {len(chunks)} chunks with quality scores")
        return chunks