"""
guards.py

The layer between "what the agent wants" and "what actually happens to the
cluster". A learned policy is not a perfect policy, so the job here is to CONTAIN
bad actions rather than pretend they never occur. The SAME guard runs inside the
training environment and in the live loop - a policy that is trained against
actions the deployed system would block learns the wrong thing.

An action is {service: {"dr": replica_delta, "cpu": target_cores, "mem": target_MiB}}
(absolute resource targets, so "no resize" is simply cpu == current cpu).

Protections, in the order they are applied:

  1. Bounds        replicas in [MIN, MAX], CPU/memory in their ranges, quantised to
                   the allocation quantum (out-of-range requests are CLAMPED, not
                   discarded - a +3 with room for +1 becomes +1).
  2. Memory floor  never shrink memory below 1.25x the observed working set
                   (the agent must not OOM-kill its own pods).
  3. Asymmetry     scale UP immediately, scale DOWN only after a cooldown:
                   under-provisioning hurts users now, over-provisioning only
                   costs money.
  4. Resize        a resize is a rolling restart, so: ignore changes inside the
                   deadband, and at most one resize per RESIZE_COOLDOWN_STEPS.
  5. Budget        the cluster can only schedule so much. Cost-increasing actions
                   are admitted in order of need until the budget is spent.
  6. Breaker       if the SLO stays breached for N consecutive windows despite the
                   agent acting, stop trusting it and fall back to HPA rules until
                   things recover.

Plus staleness detection: metrics older than METRIC_STALENESS_SECONDS are refused
rather than acted on.

`physical_only=True` applies just (1) and (5): the constraints of the machine, with
none of OUR policy. It is what the HPA/VPA baselines get - Kubernetes has no
equivalent of our cooldowns, but it cannot schedule pods beyond the cluster's size.
"""

import time
from dataclasses import dataclass, field

from src import config


def quantize(value: float, quantum: float, lo: float, hi: float) -> float:
    q = round(value / quantum) * quantum
    return float(min(hi, max(lo, round(q, 6))))


def pod_cost(cpu: float, mem_mb: float) -> float:
    """Cost of one pod in core-equivalents."""
    return cpu + config.MEM_COST_PER_GIB * mem_mb / 1024.0


@dataclass
class SafetyGuard:
    services: list
    budget: float = float("inf")
    min_replicas: int = config.MIN_REPLICAS
    max_replicas: int = config.MAX_REPLICAS

    _last_scale_down_step: dict = field(default_factory=dict)
    _last_resize_step: dict = field(default_factory=dict)
    _consecutive_violations: int = 0
    _breaker_open: bool = False
    _breaker_opened_at_step: int = -1

    # ------------------------------------------------------------------
    @property
    def breaker_open(self) -> bool:
        return self._breaker_open

    def observe_slo(self, p95_ms: float, slo_ms: float, step: int):
        """Feed the SLO outcome each window so the breaker can track health."""
        if p95_ms > slo_ms:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0
            if self._breaker_open:
                self._breaker_open = False       # recovered - hand control back
        if (not self._breaker_open
                and self._consecutive_violations >= config.CIRCUIT_BREAKER_VIOLATIONS):
            self._breaker_open = True
            self._breaker_opened_at_step = step

    def is_stale(self, metrics_timestamp: float | None) -> bool:
        if metrics_timestamp is None:
            return False
        return (time.time() - metrics_timestamp) > config.METRIC_STALENESS_SECONDS

    # ------------------------------------------------------------------
    def filter(self, proposed: dict, current: dict, step: int, physical_only: bool = False):
        """
        proposed: {svc: {"dr": int, "cpu": float, "mem": float}}
        current:  {svc: {"replicas": int, "cpu": float, "mem": float, "mem_utilization": float}}
        Returns (approved, reasons) - `approved` has the same shape as `proposed`
        with every field already safe to apply; `reasons` explains each action that
        was blocked or modified (surfaced in the dashboard).
        """
        approved, reasons = {}, {}
        for svc, act in proposed.items():
            cur = current[svc]
            a = {"dr": int(act.get("dr", 0)),
                 "cpu": float(act.get("cpu", cur["cpu"])),
                 "mem": float(act.get("mem", cur["mem"]))}
            notes = []

            # 1. bounds -------------------------------------------------------
            target = cur["replicas"] + a["dr"]
            clamped = min(self.max_replicas, max(self.min_replicas, target))
            if clamped != target:
                notes.append(f"clamped replicas to [{self.min_replicas},{self.max_replicas}]")
                a["dr"] = clamped - cur["replicas"]
            a["cpu"] = quantize(a["cpu"], config.CPU_QUANTUM, config.CPU_MIN, config.CPU_MAX)
            a["mem"] = quantize(a["mem"], config.MEM_QUANTUM_MB, config.MEM_MIN_MB, config.MEM_MAX_MB)

            if not physical_only:
                # 2. memory floor ------------------------------------------------
                working_set = cur.get("mem_utilization", 0.0) * cur["mem"]
                floor = quantize(1.25 * working_set, config.MEM_QUANTUM_MB,
                                 config.MEM_MIN_MB, config.MEM_MAX_MB)
                if a["mem"] < floor and a["mem"] < cur["mem"]:
                    notes.append(f"memory floor {floor:.0f} MiB (working set {working_set:.0f})")
                    a["mem"] = max(a["mem"], min(floor, cur["mem"]))

                # 3. asymmetric scale-down --------------------------------------
                if a["dr"] < 0:
                    since = step - self._last_scale_down_step.get(svc, -999)
                    if since < config.SCALE_DOWN_COOLDOWN_STEPS:
                        notes.append(f"scale-down cooldown ({since} of {config.SCALE_DOWN_COOLDOWN_STEPS})")
                        a["dr"] = 0

                # 4. resize deadband + cooldown ----------------------------------
                d_cpu = abs(a["cpu"] - cur["cpu"]) / max(cur["cpu"], 1e-9)
                d_mem = abs(a["mem"] - cur["mem"]) / max(cur["mem"], 1e-9)
                if (a["cpu"] != cur["cpu"] or a["mem"] != cur["mem"]):
                    if max(d_cpu, d_mem) < config.RESIZE_DEADBAND:
                        a["cpu"], a["mem"] = cur["cpu"], cur["mem"]
                    else:
                        since = step - self._last_resize_step.get(svc, -999)
                        if since < config.RESIZE_COOLDOWN_STEPS:
                            notes.append(f"resize cooldown ({since} of {config.RESIZE_COOLDOWN_STEPS})")
                            a["cpu"], a["mem"] = cur["cpu"], cur["mem"]
            approved[svc] = a
            if notes:
                reasons[svc] = "; ".join(notes)

        # 5. cluster budget -------------------------------------------------------
        self._enforce_budget(approved, current, reasons)

        # bookkeeping only for what actually goes through
        for svc, a in approved.items():
            cur = current[svc]
            if a["dr"] < 0:
                self._last_scale_down_step[svc] = step
            if a["cpu"] != cur["cpu"] or a["mem"] != cur["mem"]:
                self._last_resize_step[svc] = step
        return approved, reasons

    # ------------------------------------------------------------------
    def readiness(self, step: int) -> dict:
        """Per service, how far each cooldown still has to run, in [0, 1] (0 = ready).
        Fed to the agent as features so it does not waste actions the guard would block."""
        out = {}
        for svc in self.services:
            since_r = step - self._last_resize_step.get(svc, -999)
            since_d = step - self._last_scale_down_step.get(svc, -999)
            out[svc] = (max(0.0, (config.RESIZE_COOLDOWN_STEPS - since_r) / config.RESIZE_COOLDOWN_STEPS),
                        max(0.0, (config.SCALE_DOWN_COOLDOWN_STEPS - since_d) / config.SCALE_DOWN_COOLDOWN_STEPS))
        return out

    def total_cost(self, state: dict) -> float:
        return sum(s["replicas"] * pod_cost(s["cpu"], s["mem"]) for s in state.values())

    def _enforce_budget(self, approved, current, reasons):
        if self.budget == float("inf"):
            return
        after = {svc: {"replicas": current[svc]["replicas"] + a["dr"], "cpu": a["cpu"], "mem": a["mem"]}
                 for svc, a in approved.items()}
        total = self.total_cost(after)
        if total <= self.budget + 1e-9:
            return
        # roll back cost-increasing actions, least needy first, until within budget
        def need(svc):
            return current[svc].get("offered_load", current[svc].get("utilization", 0.0))
        order = sorted(approved, key=need)
        for svc in order:
            if total <= self.budget + 1e-9:
                break
            a, cur = approved[svc], current[svc]
            before_cost = cur["replicas"] * pod_cost(cur["cpu"], cur["mem"])
            now_cost = (cur["replicas"] + a["dr"]) * pod_cost(a["cpu"], a["mem"])
            if now_cost > before_cost + 1e-9:
                total -= now_cost - before_cost
                approved[svc] = {"dr": 0, "cpu": cur["cpu"], "mem": cur["mem"]}
                reasons[svc] = (reasons.get(svc, "") + "; " if svc in reasons else "") + \
                    f"blocked: would exceed cluster budget ({self.budget:.1f} cores-eq)"

    def reset(self):
        self._last_scale_down_step.clear()
        self._last_resize_step.clear()
        self._consecutive_violations = 0
        self._breaker_open = False
