"""
self-healing-rag/observability/alerting.py
Threshold-based alert engine for the Self-Healing RAG pipeline.

Checks (evaluated after each query when enabled)
────────────────────────────────────────────────
  SCORE_DROP    — rolling avg weighted_score dropped by > threshold vs baseline
  HIGH_HEAL_RATE — >N% of recent queries required healing
  HIGH_LATENCY   — p95 total_latency_ms in rolling window exceeds threshold

Alert backends
──────────────
  "log"      — write to application logger (always available)
  "webhook"  — POST JSON payload to alert_webhook_url
  "slack"    — POST Slack-formatted message to alert_slack_webhook_url

Design contracts
────────────────
- All backends are fire-and-forget: failures are logged, never re-raised.
- Rolling window is in-memory only — no DB reads (keeps the alert path fast).
- Alert deduplication: an alert of the same type fires at most once per
  `cooldown_queries` queries (default 20) to prevent alert storms.
- The AlertEngine is instantiated once as a singleton; its window is shared
  across all requests in the process.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from core.config import get_settings
from core.logger import logger

settings = get_settings()

AlertType = Literal["SCORE_DROP", "HIGH_HEAL_RATE", "HIGH_LATENCY"]


@dataclass
class Alert:
    alert_type: AlertType
    message: str
    value: float
    threshold: float
    fired_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class QueryWindow:
    """One slot in the rolling window."""
    weighted_score: float | None
    heal_rounds: int
    total_latency_ms: float


class AlertEngine:
    """
    Evaluates alert conditions on a rolling window of recent queries.

    Usage:
        engine = AlertEngine()
        engine.record(weighted_score=0.82, heal_rounds=1, total_latency_ms=450.0)
        # alert firing is handled internally
    """

    def __init__(self, cooldown_queries: int = 20) -> None:
        cfg = settings.observability_cfg
        self._enabled: bool = bool(cfg.get("alerting_enabled", True))
        self._window_size: int = int(cfg.get("alert_window_queries", 50))
        self._score_drop_threshold: float = float(cfg.get("alert_score_drop_threshold", 0.15))
        self._heal_rate_threshold: float = float(cfg.get("alert_heal_rate_threshold", 0.5))
        self._latency_p95_ms: float = float(cfg.get("alert_latency_p95_ms", 5000))
        self._backend: str = str(cfg.get("alert_backend", "log"))
        self._webhook_url: str = str(cfg.get("alert_webhook_url", ""))
        self._slack_url: str = str(cfg.get("alert_slack_webhook_url", ""))

        self._window: deque[QueryWindow] = deque(maxlen=self._window_size)
        self._history: list[Alert] = []          # all fired alerts (in-memory)
        self._cooldown: dict[str, int] = {}      # alert_type → queries_since_last_fire
        self._cooldown_queries = cooldown_queries
        self._baseline_score: float | None = None

    # ── public API ────────────────────────────────────────────────────

    def record(
        self,
        weighted_score: float | None,
        heal_rounds: int,
        total_latency_ms: float,
    ) -> list[Alert]:
        """
        Record one completed query. Evaluate alert conditions.
        Returns list of alerts fired this call (may be empty).
        Never raises.
        """
        if not self._enabled:
            return []

        try:
            slot = QueryWindow(
                weighted_score=weighted_score,
                heal_rounds=heal_rounds,
                total_latency_ms=total_latency_ms,
            )
            self._window.append(slot)
            self._tick_cooldowns()

            # Only evaluate alerts when window is sufficiently full
            if len(self._window) < max(5, self._window_size // 5):
                return []

            fired: list[Alert] = []
            fired.extend(self._check_score_drop())
            fired.extend(self._check_heal_rate())
            fired.extend(self._check_latency())

            for alert in fired:
                self._history.append(alert)
                self._dispatch(alert)
                self._cooldown[alert.alert_type] = 0   # reset cooldown

            return fired

        except Exception as exc:
            logger.error(f"[AlertEngine] record() failed: {exc}")
            return []

    def get_history(self, limit: int = 100) -> list[Alert]:
        """Return the most recent fired alerts."""
        return list(reversed(self._history[-limit:]))

    # ── alert checks ──────────────────────────────────────────────────

    def _check_score_drop(self) -> list[Alert]:
        scores = [s.weighted_score for s in self._window if s.weighted_score is not None]
        if not scores:
            return []

        current_avg = sum(scores) / len(scores)

        # Establish baseline from first half of full window
        if self._baseline_score is None and len(self._window) >= self._window_size:
            half = list(self._window)[: self._window_size // 2]
            half_scores = [s.weighted_score for s in half if s.weighted_score is not None]
            if half_scores:
                self._baseline_score = sum(half_scores) / len(half_scores)

        if self._baseline_score is None:
            return []

        drop = self._baseline_score - current_avg
        if drop >= self._score_drop_threshold and self._can_fire("SCORE_DROP"):
            return [Alert(
                alert_type="SCORE_DROP",
                message=(
                    f"Weighted score dropped by {drop:.3f} "
                    f"(baseline={self._baseline_score:.3f} → current={current_avg:.3f})"
                ),
                value=current_avg,
                threshold=self._baseline_score - self._score_drop_threshold,
            )]
        return []

    def _check_heal_rate(self) -> list[Alert]:
        if not self._window:
            return []
        healed = sum(1 for s in self._window if s.heal_rounds > 0)
        rate = healed / len(self._window)
        if rate >= self._heal_rate_threshold and self._can_fire("HIGH_HEAL_RATE"):
            return [Alert(
                alert_type="HIGH_HEAL_RATE",
                message=(
                    f"{healed}/{len(self._window)} recent queries required healing "
                    f"({rate*100:.1f}% ≥ threshold {self._heal_rate_threshold*100:.0f}%)"
                ),
                value=rate,
                threshold=self._heal_rate_threshold,
            )]
        return []

    def _check_latency(self) -> list[Alert]:
        if not self._window:
            return []
        latencies = sorted(s.total_latency_ms for s in self._window)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95 = latencies[p95_idx]
        if p95 >= self._latency_p95_ms and self._can_fire("HIGH_LATENCY"):
            return [Alert(
                alert_type="HIGH_LATENCY",
                message=(
                    f"p95 latency {p95:.0f}ms ≥ threshold {self._latency_p95_ms:.0f}ms "
                    f"(over last {len(self._window)} queries)"
                ),
                value=p95,
                threshold=self._latency_p95_ms,
            )]
        return []

    # ── dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, alert: Alert) -> None:
        """Send alert to the configured backend. Never raises."""
        try:
            if self._backend == "log":
                logger.warning(
                    f"[ALERT:{alert.alert_type}] {alert.message} "
                    f"value={alert.value:.4f} threshold={alert.threshold:.4f}"
                )
            elif self._backend == "webhook" and self._webhook_url:
                self._send_webhook(alert)
            elif self._backend == "slack" and self._slack_url:
                self._send_slack(alert)
            else:
                # Fallback to log if backend is misconfigured
                logger.warning(
                    f"[ALERT:{alert.alert_type}] {alert.message} "
                    f"(backend={self._backend!r} not configured, falling back to log)"
                )
        except Exception as exc:
            logger.error(f"[AlertEngine] dispatch failed: {exc}")

    def _send_webhook(self, alert: Alert) -> None:
        import urllib.request
        payload = json.dumps({
            "alert_type": alert.alert_type,
            "message": alert.message,
            "value": alert.value,
            "threshold": alert.threshold,
            "fired_at": alert.fired_at,
        }).encode()
        req = urllib.request.Request(
            self._webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.debug(f"[AlertEngine] Webhook delivered: HTTP {resp.status}")

    def _send_slack(self, alert: Alert) -> None:
        import urllib.request
        emoji = {"SCORE_DROP": "📉", "HIGH_HEAL_RATE": "🔧", "HIGH_LATENCY": "🐢"}.get(
            alert.alert_type, "⚠️"
        )
        payload = json.dumps({
            "text": f"{emoji} *[Self-Healing RAG Alert: {alert.alert_type}]*\n{alert.message}"
        }).encode()
        req = urllib.request.Request(
            self._slack_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.debug(f"[AlertEngine] Slack alert delivered: HTTP {resp.status}")

    # ── helpers ───────────────────────────────────────────────────────

    def _can_fire(self, alert_type: str) -> bool:
        """Return True if the cooldown for this alert type has expired."""
        since = self._cooldown.get(alert_type, self._cooldown_queries)
        return since >= self._cooldown_queries

    def _tick_cooldowns(self) -> None:
        """Increment query-count since last fire for every alert type."""
        for k in list(self._cooldown.keys()):
            self._cooldown[k] = self._cooldown[k] + 1