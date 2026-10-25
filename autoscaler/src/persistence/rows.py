"""
rows.py

Turns one control-step payload (inference/engines.py BaseEngine._payload) into the rows to store - a pure function of the payload and a
small amount of state, so it is tested without a database and runs on the control loop's thread in microseconds.

Rows carry no cluster or run id: the writer adds them, because the identity is resolved (and may be refused) in the background.

Which table gets what:
  steps               every step (a stale/hold step is stored, flagged)
  service_metrics     every non-stale step, every service
  actions             a service where something was proposed, or a controlled service whose action was blocked
  critical_paths      every CRITICAL_PATH_SNAPSHOT_EVERY steps and whenever the chain changes
  topology_snapshots  whenever the set of edges changes
  violations          one upserted row per SLO-breach episode (opens on the first breach, closes after two clean steps)
  events              breaker open/close, entering and leaving a hold, scaling errors
  frames              the whole payload, only when asked for
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone

from src.persistence.categorize import categorize

CLEAN_STEPS_TO_CLOSE = 2

COLUMNS = {
    "steps": ["step", "ts", "rps", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "error_rate", "slo_ms", "slo_breached",
              "breaker_open", "total_replicas", "cost", "budget", "stale", "stale_reason", "shadow", "controller"],
    "service_metrics": ["step", "ts", "service", "replicas", "ready", "pending", "cpu_limit", "mem_limit_mb", "cpu_util",
                        "mem_util", "latency_ms", "p95_ms", "rps", "on_critical_path", "controlled"],
    "actions": ["step", "ts", "service", "shadow", "outcome", "proposed_dr", "guard_dr", "applied_dr", "proposed_cpu",
                "applied_cpu", "proposed_mem_mb", "applied_mem_mb", "replicas_before", "replicas_after", "reason",
                "reason_codes", "cpu_util", "p95_ms"],
    "critical_paths": ["step", "ts", "source", "chain", "chain_latency_ms", "changed", "evidence", "profile", "slowest_trace"],
    "topology_snapshots": ["step", "ts", "edges_hash", "n_services", "edges"],
    "violations": ["start_step", "started_at", "ended_at", "end_step", "steps_breached", "peak_p95_ms", "slo_ms",
                   "top_services", "actions", "breaker_opened", "slowest_trace"],
    "events": ["step", "ts", "type", "severity", "detail"],
    "frames": ["step", "ts", "payload"],
}
JSON_COLUMNS = {"evidence", "profile", "edges", "top_services", "detail", "payload"}
# on conflict: rows are idempotent; a violation episode is updated while it is open
KEYS = {"steps": "(run_id, step)", "service_metrics": "(run_id, step, service)", "actions": "(run_id, step, service)",
        "critical_paths": "(run_id, step)", "topology_snapshots": "(run_id, edges_hash)", "violations": "(run_id, start_step)",
        "frames": "(run_id, step)"}


def _ts(payload):
    return datetime.fromtimestamp(float(payload["timestamp"]), tz=timezone.utc)


def _flag(x):
    return bool(x)


class StepRecorder:
    """Builds the rows of each step; remembers what it needs between steps (the last stored chain and edge set, the open violation,
    whether the breaker / a hold was already reported)."""

    def __init__(self, critical_source="dfs", snapshot_every=10, store_frames=False):
        self.critical_source, self.snapshot_every, self.store_frames = critical_source, max(1, snapshot_every), store_frames
        self._last_chain = None
        self._last_edges = None
        self._breaker = False
        self._holding = None
        self._episode = None
        self._clean = 0

    # ------------------------------------------------------------------
    def rows(self, payload, extras=None):
        """-> {table: [row tuple in COLUMNS order]}. Never raises on a well-formed payload; the caller still guards it."""
        extras = extras or {}
        out = defaultdict(list)
        ts, step = _ts(payload), int(payload["step"])
        e2e = payload.get("e2e", {})
        stale = _flag(payload.get("stale"))
        shadow = _flag(payload.get("shadow"))
        breaker = _flag(payload.get("breaker_open"))

        out["steps"].append((
            step, ts, payload.get("external_rps"), e2e.get("p50_ms"), e2e.get("p95_ms"), e2e.get("p99_ms"), e2e.get("mean_ms"),
            e2e.get("error_rate"), payload.get("slo_ms"), _flag(payload.get("slo_breached")) and not stale, breaker,
            payload.get("total_replicas"), payload.get("cost"), payload.get("budget"), stale, payload.get("stale_reason"),
            shadow, payload.get("controller")))
        if self.store_frames:
            out["frames"].append((step, ts, payload))

        self._events(out, payload, ts, step, stale, breaker)
        if stale:                          # a hold step carries no measurement: it neither extends nor closes a violation episode
            return dict(out)

        nodes = payload.get("nodes", [])
        for n in nodes:
            out["service_metrics"].append((
                step, ts, n["id"], n.get("replicas"), n.get("ready"), n.get("pending"), n.get("cpu"), n.get("mem"),
                n.get("utilization"), n.get("mem_utilization"), n.get("latency_ms"), n.get("latency_p95_ms"),
                n.get("requests"), _flag(n.get("on_critical_path")), _flag(n.get("controlled"))))
        applied = 0
        guard = extras.get("guard_approved") or {}
        for n in nodes:
            row = self._action(n, step, ts, shadow, guard.get(n["id"]))
            if row:
                out["actions"].append(row)
                if row[4] in ("applied", "modified") and not shadow and (
                        row[7] or row[9] != n.get("cpu") or row[11] != n.get("mem")):
                    applied += 1
                if row[4] == "k8s_error":
                    out["events"].append((step, ts, "k8s_error", "warning", {"service": n["id"], "reason": row[14]}))

        self._critical(out, payload, ts, step)
        self._topology(out, payload, ts, step, len(nodes))
        self._violation(out, payload, ts, step, breaker, applied)
        return dict(out)

    # ------------------------------------------------------------------
    @staticmethod
    def _action(n, step, ts, shadow, guard_dr):
        p_dr, a_dr = int(n.get("proposed_delta", 0)), int(n.get("applied_delta", 0))
        cpu, mem = n.get("cpu"), n.get("mem")
        p_cpu, a_cpu = n.get("proposed_cpu", cpu), n.get("applied_cpu", cpu)
        p_mem, a_mem = n.get("proposed_mem", mem), n.get("applied_mem", mem)
        proposal = bool(p_dr or p_cpu != cpu or p_mem != mem)
        reason = n.get("blocked_reason")
        if str(reason or "").startswith("HOLD"):
            return None
        if not proposal and not (n.get("controlled") and reason):
            return None
        codes = categorize(reason)
        if "k8s_error" in codes:
            outcome = "k8s_error"
        elif not proposal:
            outcome = "vetoed"
        elif (a_dr, a_cpu, a_mem) == (p_dr, p_cpu, p_mem):
            outcome = "applied"
        elif (a_dr, a_cpu, a_mem) == (0, cpu, mem):
            outcome = "vetoed"
        else:
            outcome = "modified"
        before = n.get("replicas")
        after = before + a_dr if (before is not None and outcome in ("applied", "modified") and not shadow) else before
        return (step, ts, n["id"], shadow, outcome, p_dr, guard_dr, a_dr, p_cpu, a_cpu, p_mem, a_mem, before, after,
                reason, codes, n.get("utilization"), n.get("latency_p95_ms"))

    def _events(self, out, payload, ts, step, stale, breaker):
        if breaker != self._breaker:
            out["events"].append((step, ts, "breaker_open" if breaker else "breaker_closed",
                                  "warning" if breaker else "info", {"p95_ms": payload.get("e2e", {}).get("p95_ms"),
                                                                    "slo_ms": payload.get("slo_ms")}))
            self._breaker = breaker
        if stale and payload.get("stale_reason") != self._holding:
            out["events"].append((step, ts, "hold", "warning", {"reason": payload.get("stale_reason"),
                                                                "metrics_age_s": payload.get("metrics_age_s")}))
            self._holding = payload.get("stale_reason")
        elif not stale and self._holding is not None:
            out["events"].append((step, ts, "recovered", "info", {"after": self._holding}))
            self._holding = None

    def _critical(self, out, payload, ts, step):
        chain = list(payload.get("critical_path") or [])
        if not chain:
            return
        changed = chain != self._last_chain
        if not changed and step % self.snapshot_every:
            return
        ev = payload.get("critical_path_evidence")
        self._last_chain = chain
        out["critical_paths"].append((step, ts, self.critical_source, chain, payload.get("critical_path_latency_ms"), changed,
                                      ev, payload.get("critical_profile"), (ev or {}).get("slowest_trace")))

    def _topology(self, out, payload, ts, step, n_services):
        edges = sorted((e["source"], e["target"]) for e in payload.get("edges", []))
        if edges == self._last_edges:
            return
        self._last_edges = edges
        digest = hashlib.sha1(json.dumps(edges).encode()).hexdigest()[:16]
        out["topology_snapshots"].append((step, ts, digest, n_services, [
            {"source": e["source"], "target": e["target"], "share": e.get("traffic_share")} for e in payload.get("edges", [])]))

    def finish(self):
        """At the end of a run: an SLO-violation episode still open is closed at its last breached step (otherwise it would be drawn as lasting for ever)."""
        out = {}
        if self._episode is not None:
            ep = self._episode
            out["violations"] = [self._episode_row(ep, ep["peak"] and ep.get("slo", 0.0), closed=True)]
            self._episode, self._clean = None, 0
        return out

    # -- SLO violation episodes ---------------------------------------------
    def _violation(self, out, payload, ts, step, breaker, applied):
        breached = _flag(payload.get("slo_breached"))
        p95, slo = payload.get("e2e", {}).get("p95_ms", 0.0), payload.get("slo_ms", 0.0)
        ep = self._episode
        if breached:
            self._clean = 0
            if ep is None:
                ep = self._episode = {"start_step": step, "started_at": ts, "steps": 0, "peak": 0.0, "actions": 0,
                                      "breaker": False, "trace": None, "shares": defaultdict(float), "last_breach": ts,
                                      "last_step": step}
            ep["steps"] += 1
            ep["slo"] = slo
            ep["last_breach"], ep["last_step"] = ts, step
            ep["actions"] += applied
            ep["breaker"] = ep["breaker"] or breaker
            if p95 >= ep["peak"]:
                ep["peak"], ep["trace"] = p95, (payload.get("critical_path_evidence") or {}).get("slowest_trace") or ep["trace"]
            share = (payload.get("critical_path_evidence") or {}).get("share") or {}
            if share:
                for s, v in share.items():
                    ep["shares"][s] += v
            else:
                for n in payload.get("nodes", []):
                    if n.get("on_critical_path") and not n.get("is_entry"):
                        ep["shares"][n["id"]] += 1.0
            out["violations"].append(self._episode_row(ep, slo, closed=False))
        elif ep is not None:
            self._clean += 1
            if self._clean >= CLEAN_STEPS_TO_CLOSE:
                out["violations"].append(self._episode_row(ep, slo, closed=True))
                self._episode, self._clean = None, 0

    @staticmethod
    def _episode_row(ep, slo, closed):
        total = sum(ep["shares"].values()) or 1.0
        top = [{"service": s, "share": round(v / total, 3)} for s, v in sorted(ep["shares"].items(), key=lambda kv: -kv[1])[:5]]
        return (ep["start_step"], ep["started_at"], ep["last_breach"] if closed else None, ep["last_step"] if closed else None,
                ep["steps"], ep["peak"], slo, top, ep["actions"], ep["breaker"], ep["trace"])
