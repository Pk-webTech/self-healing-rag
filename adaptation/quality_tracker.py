"""
self-healing-rag/adaptation/quality_tracker.py
Exponentially Weighted Moving Average (EWM) quality tracker.

After every evaluation cycle the HealingLoop has an EvalResult that
contains the retrieved chunks and the verdict. The QualityTracker:
  1. Reads each chunk's current quality_score from the store metadata.
  2. Computes a new EWM score:  ewm = α * signal + (1-α) * old_ewm
     where `signal` = 1.0 for PASS, 0.0 for HARD_FAIL, 0.5 for SOFT_FAIL
  3. Writes the updated score back to the vector store via update_chunk_metadata.
  4. Persists a ChunkQualityHistory row for the time-series.

This is intentionally kept dependency-free (no Optuna, no pandas) — just
arithmetic on floats and async DB writes.

Bug-prevention notes
────────────────────
- We never set quality_score < 0 or > 1.
- We only update chunks that actually appeared in THIS retrieval result
  (not all chunks in the store) — targeting is per-retrieved-chunk.
- alpha is read from config every call (not cached) so live config
  changes take effect without restart.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logger import logger
from core.models import EvalResult, Verdict
from db.models import ChunkQualityHistory

if TYPE_CHECKING:
    from ingestion.vector_store import BaseVectorStore

settings = get_settings()

# Signal value per verdict — maps eval outcome to a quality signal
_VERDICT_SIGNAL: dict[str, float] = {
    Verdict.PASS.value: 1.0,
    Verdict.SOFT_FAIL.value: 0.5,
    Verdict.HARD_FAIL.value: 0.0,
}


def _ewm_update(old_score: float, signal: float, alpha: float) -> float:
    """Standard EWM update: new = alpha * signal + (1 - alpha) * old."""
    new = alpha * signal + (1.0 - alpha) * old_score
    return round(min(1.0, max(0.0, new)), 4)


class QualityTracker:
    """
    Tracks per-chunk quality scores using EWM.

    Usage:
        tracker = QualityTracker(store)
        await tracker.update(eval_result, session)
    """

    def __init__(self, store: "BaseVectorStore") -> None:
        self.store = store

    async def update(
        self,
        eval_result: EvalResult,
        session: AsyncSession,
    ) -> dict[str, float]:
        """
        Update quality scores for all chunks in eval_result.retrieval.
        Returns mapping of chunk_id → new EWM score.
        """
        cfg = settings.adaptation_cfg
        alpha = float(cfg["ewm_alpha"])
        signal = _VERDICT_SIGNAL.get(eval_result.verdict.value, 0.5)
        verdict_str = eval_result.verdict.value
        weighted = eval_result.weighted_score

        updated: dict[str, float] = {}

        for rc in eval_result.retrieval.chunks:
            chunk = rc.chunk
            old_score = chunk.quality_score
            new_ewm = _ewm_update(old_score, signal, alpha)
            updated[chunk.chunk_id] = new_ewm

            # Write back to vector store
            try:
                self.store.update_chunk_metadata(
                    chunk.chunk_id,
                    {
                        "quality_score": new_ewm,
                        "retrieval_count": chunk.retrieval_count + 1,
                    },
                )
            except Exception as exc:
                logger.error(
                    f"[QualityTracker] Failed to update store for "
                    f"chunk {chunk.chunk_id[:8]}: {exc}"
                )

            # Persist history row (fail-safe)
            try:
                row = ChunkQualityHistory(
                    chunk_id=chunk.chunk_id,
                    chunk_source=chunk.source,
                    raw_quality_score=old_score,
                    ewm_quality_score=new_ewm,
                    verdict=verdict_str,
                    weighted_score=weighted,
                    failure_count=chunk.failure_count,
                    heal_flag=chunk.heal_flag,
                )
                session.add(row)
                await session.flush()
            except Exception as exc:
                logger.error(
                    f"[QualityTracker] Failed to persist history for "
                    f"chunk {chunk.chunk_id[:8]}: {exc}"
                )

            logger.debug(
                f"[QualityTracker] chunk {chunk.chunk_id[:8]} "
                f"quality {old_score:.3f}→{new_ewm:.3f} "
                f"(signal={signal} α={alpha})"
            )

        logger.info(
            f"[QualityTracker] Updated {len(updated)} chunks "
            f"verdict={verdict_str} signal={signal}"
        )
        return updated