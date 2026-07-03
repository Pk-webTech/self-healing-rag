"""
self-healing-rag/adaptation/chunk_evolver.py
ChunkEvolver: inspects flagged chunks and decides their fate.

Decision tree per chunk
───────────────────────
quality_score < drop_threshold   OR failure_count >= max_failure_count
    → "drop"   : delete from vector store, record in history

drop_threshold <= quality_score < quarantine_threshold
    → "flag"   : keep in store but mark heal_flag=True (already set by Phase 3)
                 a human or future re-index pass can handle these

quality_score >= quarantine_threshold
    → "keep"   : clear heal_flag if set, update store

The evolver only inspects chunks that have heal_flag=True in the store,
fetched via a dedicated scan. It processes at most `evolver_batch_size`
per run to avoid memory spikes.

Bug-prevention notes
────────────────────
- We never query the entire vector store at once — we scan only flagged chunks
  by filtering on heal_flag=True.
- drop_threshold < quarantine_threshold is validated at init time.
- Deleting from FAISS is a soft-delete (chunk removed from dict but FAISS
  index is NOT rebuilt — acceptable for Phase 4, noted as future work).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logger import logger
from db.models import ChunkQualityHistory

if TYPE_CHECKING:
    from ingestion.vector_store import BaseVectorStore

settings = get_settings()


@dataclass
class EvolverReport:
    dropped: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.dropped) + len(self.flagged) + len(self.kept)


class ChunkEvolver:
    """
    Inspect heal-flagged chunks and evolve the store.

    Usage:
        evolver = ChunkEvolver(store)
        report = await evolver.run(session)
    """

    def __init__(self, store: "BaseVectorStore") -> None:
        self.store = store
        cfg = settings.adaptation_cfg
        self.drop_threshold: float = float(cfg["drop_threshold"])
        self.quarantine_threshold: float = float(cfg["quarantine_threshold"])
        self.max_failure_count: int = int(cfg["max_failure_count"])
        self.batch_size: int = int(cfg["evolver_batch_size"])

        # Validate thresholds are ordered correctly — catch config errors early
        if self.drop_threshold >= self.quarantine_threshold:
            raise ValueError(
                f"drop_threshold ({self.drop_threshold}) must be < "
                f"quarantine_threshold ({self.quarantine_threshold})"
            )

    async def run(self, session: AsyncSession) -> EvolverReport:
        """
        Scan flagged chunks and apply drop / flag / keep decisions.
        """
        report = EvolverReport()

        # Fetch flagged chunks from store — filter heal_flag=True
        try:
            flagged_chunks = self.store.query(
                query_embedding=[0.0] * 1,  # dummy — we only use filters
                k=self.batch_size,
                filters={"heal_flag": True},
            )
        except Exception as exc:
            # Some stores (FAISS) don't support metadata filters — fall back
            # to scanning all chunks in the internal dict if available.
            logger.warning(
                f"[ChunkEvolver] Store filter not supported ({exc}), "
                "falling back to full scan"
            )
            flagged_chunks = self._fallback_scan()

        if not flagged_chunks:
            logger.info("[ChunkEvolver] No flagged chunks found — nothing to evolve")
            return report

        logger.info(f"[ChunkEvolver] Inspecting {len(flagged_chunks)} flagged chunk(s)")

        for rc in flagged_chunks:
            chunk = rc.chunk
            action = self._decide(chunk.quality_score, chunk.failure_count)

            try:
                if action == "drop":
                    self.store.delete_chunk(chunk.chunk_id)
                    report.dropped.append(chunk.chunk_id)
                    logger.warning(
                        f"[ChunkEvolver] DROPPED {chunk.chunk_id[:8]} "
                        f"quality={chunk.quality_score:.3f} "
                        f"failures={chunk.failure_count}"
                    )
                elif action == "flag":
                    # Already flagged — ensure metadata is consistent
                    self.store.update_chunk_metadata(
                        chunk.chunk_id, {"heal_flag": True}
                    )
                    report.flagged.append(chunk.chunk_id)
                else:  # keep — quality recovered
                    self.store.update_chunk_metadata(
                        chunk.chunk_id, {"heal_flag": False}
                    )
                    report.kept.append(chunk.chunk_id)

                # Persist decision to history (fail-safe)
                await self._persist_decision(session, chunk, action)

            except Exception as exc:
                logger.error(
                    f"[ChunkEvolver] Error processing chunk {chunk.chunk_id[:8]}: {exc}"
                )

        logger.info(
            f"[ChunkEvolver] Complete — "
            f"dropped={len(report.dropped)} "
            f"flagged={len(report.flagged)} "
            f"kept={len(report.kept)}"
        )
        return report

    def _decide(self, quality_score: float, failure_count: int) -> str:
        """Return 'drop', 'flag', or 'keep'."""
        if quality_score < self.drop_threshold or failure_count >= self.max_failure_count:
            return "drop"
        if quality_score < self.quarantine_threshold:
            return "flag"
        return "keep"

    def _fallback_scan(self):
        """
        FAISS fallback: iterate internal _chunks dict if available.
        Returns list of RetrievedChunk-like objects with .chunk attribute.
        """
        from core.models import RetrievedChunk
        chunks = []
        store_dict = getattr(self.store, "_chunks", {})
        for chunk in store_dict.values():
            if chunk.heal_flag:
                chunks.append(RetrievedChunk(chunk=chunk, score=0.0))
        return chunks[: self.batch_size]

    async def _persist_decision(self, session: AsyncSession, chunk, action: str) -> None:
        try:
            row = ChunkQualityHistory(
                chunk_id=chunk.chunk_id,
                chunk_source=chunk.source,
                raw_quality_score=chunk.quality_score,
                ewm_quality_score=chunk.quality_score,  # evolver reads current score
                failure_count=chunk.failure_count,
                heal_flag=chunk.heal_flag,
                evolver_action=action,
            )
            session.add(row)
            await session.flush()
        except Exception as exc:
            logger.error(
                f"[ChunkEvolver] Failed to persist decision for "
                f"chunk {chunk.chunk_id[:8]}: {exc}"
            )