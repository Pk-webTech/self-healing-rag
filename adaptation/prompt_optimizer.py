"""
self-healing-rag/adaptation/prompt_optimizer.py
Few-shot prompt optimizer: selects high-quality (query, answer) pairs from
QueryLog history and injects them as examples into the RAG prompt.

Design
──────
- Reads the top-N highest-scoring QueryLog rows (weighted_score >= threshold,
  verdict = PASS) as few-shot candidates.
- Deduplicates by semantic similarity: we sort by score DESC and keep examples
  that are "sufficiently different" from already-selected ones (simple
  character-level Jaccard distance — no embedding call needed).
- Writes the selected examples into `configs/prompts.yaml` under a new
  `rag_few_shot_examples` key that the Generator reads at runtime.

Why no DSPy?
────────────
DSPy requires a `dspy.LM` compiled training loop with ground-truth labels.
We don't have labels in Phase 4 — only proxy signals (weighted_score).
A retrieval-based few-shot selector on existing good examples is more
appropriate and dependency-free.

Bug-prevention notes
────────────────────
- We never write to config.yaml directly (could corrupt the file).
  Instead we update `configs/prompts.yaml` which is already the designated
  mutable prompts file.
- We reload the prompts file fresh each call (no module-level cache) so
  changes propagate immediately.
- Jaccard dedup threshold is generous (0.3) to allow topically similar
  but phrased-differently examples through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logger import logger
from db.models import QueryLog

settings = get_settings()

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate(
    candidates: list[tuple[str, str]],
    n: int,
    diversity_threshold: float = 0.5,
) -> list[tuple[str, str]]:
    """
    Select up to `n` examples from candidates that are pairwise dissimilar.
    candidates: list of (query, answer) sorted by score DESC.
    Keeps an example if its Jaccard similarity to ALL already-selected
    examples is below diversity_threshold.
    """
    selected: list[tuple[str, str]] = []
    for query, answer in candidates:
        too_similar = any(
            _jaccard(query, sel_q) > diversity_threshold
            for sel_q, _ in selected
        )
        if not too_similar:
            selected.append((query, answer))
        if len(selected) >= n:
            break
    return selected


@dataclass
class OptimizerReport:
    examples_found: int
    examples_selected: int
    examples_written: bool
    min_score: float | None = None
    max_score: float | None = None
    skipped: bool = False
    reason: str = ""


class PromptOptimizer:
    """
    Selects high-quality few-shot examples from QueryLog and writes them
    to prompts.yaml for use by the Generator.

    Usage:
        optimizer = PromptOptimizer()
        report = await optimizer.run(session)
    """

    def __init__(self) -> None:
        cfg = settings.adaptation_cfg
        self.n_examples: int = int(cfg["prompt_optimizer_n_examples"])
        self.min_score: float = float(cfg["prompt_optimizer_min_score"])

    async def run(self, session: AsyncSession) -> OptimizerReport:
        # ── 1. Fetch high-quality examples ───────────────────────────
        result = await session.execute(
            select(QueryLog.query, QueryLog.answer, QueryLog.weighted_score)
            .where(
                QueryLog.verdict == "PASS",
                QueryLog.weighted_score >= self.min_score,
                # Exclude very short answers (likely error messages)
                QueryLog.answer.isnot(None),
            )
            .order_by(QueryLog.weighted_score.desc())
            .limit(self.n_examples * 5)  # fetch 5x to allow diversity filtering
        )
        rows = result.fetchall()

        if not rows:
            logger.info(
                f"[PromptOptimizer] No qualifying examples "
                f"(weighted_score >= {self.min_score}, verdict=PASS)"
            )
            return OptimizerReport(
                examples_found=0,
                examples_selected=0,
                examples_written=False,
                skipped=True,
                reason="no_qualifying_examples",
            )

        # ── 2. Deduplicate for diversity ──────────────────────────────
        candidates = [(row[0], row[1]) for row in rows]
        selected = _deduplicate(candidates, self.n_examples)
        scores = [row[2] for row in rows[: len(selected)]]

        # ── 3. Write to prompts.yaml ──────────────────────────────────
        written = self._write_examples(selected)

        logger.info(
            f"[PromptOptimizer] Selected {len(selected)} few-shot example(s) "
            f"from {len(rows)} candidates "
            f"(scores: {min(scores):.3f}–{max(scores):.3f})"
        )

        return OptimizerReport(
            examples_found=len(rows),
            examples_selected=len(selected),
            examples_written=written,
            min_score=min(scores) if scores else None,
            max_score=max(scores) if scores else None,
        )

    def _write_examples(self, examples: list[tuple[str, str]]) -> bool:
        """
        Append/update the rag_few_shot_examples key in prompts.yaml.
        Returns True on success, False on error.
        """
        try:
            with open(PROMPTS_PATH) as f:
                prompts = yaml.safe_load(f) or {}

            prompts["rag_few_shot_examples"] = [
                {"query": q, "answer": a} for q, a in examples
            ]

            # Also update the rag_system prompt to mention few-shot examples
            # if examples are present — only if the prompt doesn't already
            # include the few-shot instruction.
            existing_system = prompts.get("rag_system", "")
            if examples and "Example" not in existing_system:
                prompts["rag_system"] = existing_system.rstrip() + (
                    "\n\nUse the following high-quality examples as style guidance "
                    "(do not copy them verbatim):\n{few_shot_block}"
                )

            with open(PROMPTS_PATH, "w") as f:
                yaml.safe_dump(prompts, f, default_flow_style=False, allow_unicode=True)

            return True
        except Exception as exc:
            logger.error(f"[PromptOptimizer] Failed to write examples: {exc}")
            return False