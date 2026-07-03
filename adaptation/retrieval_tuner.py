"""
self-healing-rag/adaptation/retrieval_tuner.py
Lightweight grid search over retrieval hyperparameters (k, alpha) using
past QueryLog outcomes as a proxy objective.

Why no Optuna?
──────────────
Optuna requires a trial function that calls the actual retrieval + eval
pipeline, which would cost LLM API calls per trial. Instead we use a
*retrospective* approach: read past QueryLog rows (which already have
weighted_score), group them by the k/alpha that was active at query time,
and pick the combo with the highest average weighted_score.

Since we don't currently log k/alpha per query (Phase 1 design), we instead
simulate: for each candidate (k, alpha) pair we score the candidate against
the distribution of weighted_scores in the log, applying a small reward for
higher k (more context) and higher alpha (richer dense signal). This is
a deliberate simplification — a real HPO loop would require infrastructure
(shadow traffic, A/B routing) that is out of scope for Phase 4.

The tuner:
  1. Reads the most recent `tuner_lookback` QueryLog rows.
  2. Computes average weighted_score as the baseline quality signal.
  3. Applies heuristic scoring per (k, alpha) candidate.
  4. Writes the best (k, alpha) back to the live RetrievalPipeline.
  5. Returns a TunerReport so the caller can log the decision.

Bug-prevention notes
────────────────────
- We never write to config.yaml (immutable at runtime) — we mutate the
  in-memory RetrievalPipeline singleton directly, same pattern as the
  existing re_retrieve action does safely.
- min_samples guard: if there are fewer than `tuner_min_samples` rows we
  skip tuning to avoid noisy decisions on sparse data.
- All SQL uses async SQLAlchemy — no blocking calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logger import logger
from db.models import QueryLog

if TYPE_CHECKING:
    from retrieval import RetrievalPipeline

settings = get_settings()


@dataclass
class TunerReport:
    skipped: bool
    reason: str
    best_k: int | None = None
    best_alpha: float | None = None
    baseline_score: float | None = None
    candidate_scores: dict[str, float] | None = None
    applied: bool = False


def _heuristic_score(
    base_avg: float,
    k: int,
    alpha: float,
    current_k: int,
    current_alpha: float,
) -> float:
    """
    Score a (k, alpha) candidate relative to current settings.

    Intuition:
    - Higher k broadens recall; we reward moderate increases but penalise
      very high k (more noise, slower retrieval).
    - alpha closer to 0.7 is the empirically safe dense/BM25 balance;
      extreme values (pure BM25 or pure dense) are penalised.

    Returns a score in roughly the same range as weighted_score (0–1).
    """
    # k factor: reward up to 1.5x current k, penalise beyond
    k_ratio = k / max(current_k, 1)
    k_factor = 1.0 + 0.05 * min(k_ratio - 1.0, 0.5) - 0.03 * max(k_ratio - 1.5, 0.0)

    # alpha factor: gentle bowl centred at 0.7
    alpha_factor = 1.0 - 0.1 * abs(alpha - 0.7)

    return round(base_avg * k_factor * alpha_factor, 4)


class RetrievalTuner:
    """
    Usage:
        tuner = RetrievalTuner(retrieval_pipeline)
        report = await tuner.run(session)
    """

    def __init__(self, pipeline: "RetrievalPipeline") -> None:
        self.pipeline = pipeline
        cfg = settings.adaptation_cfg
        self.k_candidates: list[int] = [int(x) for x in cfg["k_candidates"]]
        self.alpha_candidates: list[float] = [float(x) for x in cfg["alpha_candidates"]]
        self.min_samples: int = int(cfg["tuner_min_samples"])
        self.lookback: int = int(cfg["tuner_lookback"])

    async def run(self, session: AsyncSession) -> TunerReport:
        # ── 1. Fetch recent query logs ────────────────────────────────
        result = await session.execute(
            select(QueryLog.weighted_score)
            .where(QueryLog.weighted_score.isnot(None))
            .order_by(QueryLog.created_at.desc())
            .limit(self.lookback)
        )
        scores = [row[0] for row in result.fetchall()]

        if len(scores) < self.min_samples:
            logger.info(
                f"[RetrievalTuner] Only {len(scores)} samples — "
                f"need {self.min_samples} before tuning"
            )
            return TunerReport(
                skipped=True,
                reason=f"insufficient_samples ({len(scores)}/{self.min_samples})",
            )

        base_avg = round(sum(scores) / len(scores), 4)
        current_k = self.pipeline.retriever.k
        current_alpha = self.pipeline.retriever.alpha

        # ── 2. Score all candidates ───────────────────────────────────
        candidate_scores: dict[str, float] = {}
        best_key = f"k={current_k},α={current_alpha}"
        best_score = base_avg

        for k in self.k_candidates:
            for alpha in self.alpha_candidates:
                key = f"k={k},α={alpha}"
                sc = _heuristic_score(base_avg, k, alpha, current_k, current_alpha)
                candidate_scores[key] = sc
                if sc > best_score:
                    best_score = sc
                    best_key = key

        # ── 3. Extract best k and alpha ───────────────────────────────
        # Parse from "k=10,α=0.7" format
        try:
            parts = best_key.replace("α=", "").split(",")
            best_k = int(parts[0].split("=")[1])
            best_alpha = float(parts[1])
        except Exception as exc:
            logger.error(f"[RetrievalTuner] Failed to parse best key '{best_key}': {exc}")
            return TunerReport(
                skipped=True,
                reason=f"parse_error: {exc}",
                baseline_score=base_avg,
            )

        # ── 4. Apply if different from current ────────────────────────
        applied = False
        if best_k != current_k or best_alpha != current_alpha:
            self.pipeline.retriever.k = best_k
            self.pipeline.retriever.alpha = best_alpha
            applied = True
            logger.info(
                f"[RetrievalTuner] Applied: k {current_k}→{best_k}, "
                f"α {current_alpha}→{best_alpha} "
                f"(baseline={base_avg:.3f} best={best_score:.3f})"
            )
        else:
            logger.info(
                f"[RetrievalTuner] Current settings already optimal "
                f"(k={current_k}, α={current_alpha}, avg_score={base_avg:.3f})"
            )

        return TunerReport(
            skipped=False,
            reason="ok",
            best_k=best_k,
            best_alpha=best_alpha,
            baseline_score=base_avg,
            candidate_scores=candidate_scores,
            applied=applied,
        )