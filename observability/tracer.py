"""
self-healing-rag/observability/tracer.py
Structured request tracer. Writes one JSON object per line to
data/logs/traces.jsonl. Rotates the file when it exceeds trace_max_file_mb.

Design choices
──────────────
- Synchronous file I/O only — no async file handles. The GIL makes short
  sequential writes safe, and tracing must never block the event loop.
- File is opened in append mode per write — no persistent file handle that
  can become stale or locked across reloads.
- Rotation renames the current file to traces.jsonl.1 before starting fresh.
  Only one backup is kept (simple; sufficient for dev/staging use).
- The tracer is entirely fail-safe: _write() never raises.

Trace schema per line
─────────────────────
{
  "trace_id":        str   — uuid4 hex (unique per query)
  "timestamp":       str   — ISO 8601 UTC
  "query":           str   — user query (truncated to 500 chars)
  "verdict":         str|null
  "weighted_score":  float|null
  "heal_rounds":     int
  "actions_taken":   list[str]
  "latency_ms":      float   — total pipeline latency
  "generation_ms":   float   — LLM generation latency
  "prompt_tokens":   int
  "completion_tokens": int
  "model":           str
  "num_chunks":      int     — chunks retrieved
  "avg_chunk_score": float   — avg retrieval score
  "error":           str|null — set if pipeline raised
}
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from core.config import get_settings
from core.logger import logger

if TYPE_CHECKING:
    from core.models import EvalResult, GenerationResult, RetrievalResult

settings = get_settings()


class RequestTracer:
    """
    Write structured traces to a JSONL file.

    Usage:
        tracer = RequestTracer()
        tracer.trace(retrieval, generation, eval_result, total_latency_ms=123.4)
    """

    def __init__(self, trace_path: str | Path | None = None) -> None:
        cfg = settings.observability_cfg
        self._enabled: bool = bool(cfg.get("tracer_enabled", True))
        self._path = Path(trace_path or cfg.get("trace_log_path", "data/logs/traces.jsonl"))
        self._max_bytes: int = int(cfg.get("trace_max_file_mb", 50)) * 1024 * 1024

    def trace(
        self,
        retrieval: "RetrievalResult",
        generation: "GenerationResult",
        eval_result: "EvalResult | None" = None,
        total_latency_ms: float = 0.0,
        heal_rounds: int = 0,
        actions_taken: list[str] | None = None,
        error: str | None = None,
    ) -> str:
        """
        Write one trace record. Returns the trace_id string.
        Never raises — all exceptions are caught and logged.
        """
        trace_id = uuid.uuid4().hex

        if not self._enabled:
            return trace_id

        try:
            avg_chunk_score = (
                sum(rc.score for rc in retrieval.chunks) / len(retrieval.chunks)
                if retrieval.chunks else 0.0
            )

            record = {
                "trace_id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": retrieval.query[:500],
                "verdict": eval_result.verdict.value if eval_result else None,
                "weighted_score": (
                    round(eval_result.weighted_score, 4) if eval_result else None
                ),
                "heal_rounds": heal_rounds,
                "actions_taken": actions_taken or [],
                "latency_ms": round(total_latency_ms, 1),
                "generation_ms": round(generation.latency_ms, 1),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "model": generation.model[:64],
                "num_chunks": len(retrieval.chunks),
                "avg_chunk_score": round(avg_chunk_score, 4),
                "error": error,
            }

            self._write(record)

        except Exception as exc:
            logger.error(f"[Tracer] Failed to build trace record: {exc}")

        return trace_id

    def _write(self, record: dict) -> None:
        """Append one JSON record to the trace file. Rotate if needed."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            # Rotate if file exceeds max size
            if self._path.exists() and self._path.stat().st_size >= self._max_bytes:
                backup = self._path.with_suffix(".jsonl.1")
                backup.unlink(missing_ok=True)
                self._path.rename(backup)
                logger.info(f"[Tracer] Rotated trace file → {backup}")

            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as exc:
            logger.error(f"[Tracer] File write failed: {exc}")

    def read_recent(self, n: int = 100) -> list[dict]:
        """
        Read the last `n` trace records from the file.
        Returns an empty list if the file doesn't exist or is unreadable.
        Used by the /traces API endpoint.
        """
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            recent = lines[-n:]
            records = []
            for line in recent:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue   # skip malformed lines
            return list(reversed(records))  # newest first
        except Exception as exc:
            logger.error(f"[Tracer] Failed to read traces: {exc}")
            return []