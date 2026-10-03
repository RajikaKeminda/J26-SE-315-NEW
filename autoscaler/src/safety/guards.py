"""
guards.py

The layer between "what the agent wants" and "what actually happens to the
cluster". A frozen policy is not a perfect policy, so the job here is to
CONTAIN bad actions rather than pretend they never occur.

Four protections:

  1. Bounds          - never below MIN_REPLICAS or above MAX_REPLICAS.
  2. Rate limiting   - a service can't be rescaled every single window.
  3. Asymmetry       - scale UP fast, scale DOWN slow. Under-provisioning
                       hurts users immediately; over-provisioning only costs
                       money, so the two errors deserve different caution.
  4. Circuit breaker - if the SLO stays breached for N consecutive windows
                       despite the agent acting, stop trusting the agent and
                       fall back to plain HPA rules until things recover.

Plus staleness detection: if metrics are older than METRIC_STALENESS_SECONDS
we refuse to act at all, rather than acting confidently on stale or
zero-filled data.
"""

import time
from dataclasses import dataclass, field

from src import config


@dataclass
class SafetyGuard:
    services: list
    min_replicas: int = config.MIN_REPLICAS
    max_replicas: int = config.MAX_REPLICAS

    _last_action_step: dict = field(default_factory=dict)
    _last_scale_down_step: dict = field(default_factory=dict)
    _consecutive_violations: int = 0
    _breaker_open: bool = False
    _breaker_opened_at_step: int = -1

    # ------------------------------------------------------------------
    @property
    def breaker_open(self) -> bool:
        return self._breaker_open

    def observe_slo(self, crit_latency_ms: float, step: int):
        """Feed the SLO outcome each window so the breaker can track health."""
        if crit_latency_ms > config.SLO_LATENCY_MS:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0
            if self._breaker_open:
                self._breaker_open = False   # recovered - hand control back

        if (not self._breaker_open
                and self._consecutive_violations >= config.CIRCUIT_BREAKER_VIOLATIONS):
            self._breaker_open = True
            self._breaker_opened_at_step = step

    def is_stale(self, metrics_timestamp: float | None) -> bool:
        if metrics_timestamp is None:
            return False
        return (time.time() - metrics_timestamp) > config.METRIC_STALENESS_SECONDS

    # ------------------------------------------------------------------
    def filter(self, proposed: dict[str, int], current_replicas: dict[str, int],
               step: int) -> tuple[dict[str, int], dict[str, str]]:
        """
        Returns (approved_deltas, reasons) where `reasons` explains every
        action that was blocked or modified - surfaced in the dashboard so
        the operator can see why the agent didn't get its way.
        """
        approved, reasons = {}, {}

        for svc, delta in proposed.items():
            if delta == 0:
                approved[svc] = 0
                continue

            current = current_replicas.get(svc, self.min_replicas)

            # 1. bounds
            target = current + delta
            if target < self.min_replicas:
                reasons[svc] = f"blocked: would go below min ({self.min_replicas})"
                approved[svc] = 0
                continue
            if target > self.max_replicas:
                reasons[svc] = f"blocked: would exceed max ({self.max_replicas})"
                approved[svc] = 0
                continue

            # 2. rate limit
            since_last = step - self._last_action_step.get(svc, -999)
            if since_last < config.RATE_LIMIT_STEPS:
                reasons[svc] = f"blocked: rate limited ({since_last} of {config.RATE_LIMIT_STEPS} steps)"
                approved[svc] = 0
                continue

            # 3. asymmetric caution on scale-down
            if delta < 0:
                since_down = step - self._last_scale_down_step.get(svc, -999)
                if since_down < config.SCALE_DOWN_COOLDOWN_STEPS:
                    reasons[svc] = (f"blocked: scale-down cooldown "
                                    f"({since_down} of {config.SCALE_DOWN_COOLDOWN_STEPS} steps)")
                    approved[svc] = 0
                    continue
                self._last_scale_down_step[svc] = step

            approved[svc] = delta
            self._last_action_step[svc] = step

        return approved, reasons

    def reset(self):
        self._last_action_step.clear()
        self._last_scale_down_step.clear()
        self._consecutive_violations = 0
        self._breaker_open = False
