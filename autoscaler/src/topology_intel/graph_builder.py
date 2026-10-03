"""
graph_builder.py

Turns one window of monitoring data into the graph-structured state the agent reads.
Both graphs of the research design are built here:

  SDG (Service Dependency Graph)
      Nodes = services, edges = call frequency (the share of the caller's requests
      routed to each callee). Static while the topology is static.

  EHG (Execution History Graph)
      Nodes = services, edges = how much latency the call actually costs RIGHT NOW:
      the callee's inclusive p95 latency (relative to the SLO) scaled by the traffic
      share. Rebuilt every window from live latency percentiles.

plus a per-node feature matrix X, a small vector of cluster-level scalars, and the
critical path. Everything is topology-size-agnostic: pass 6 services or 60 and only
the tensor shapes change. The functions return NumPy arrays (the replay buffer stores
them as-is); rl_agent/batching.py pads and converts them to torch tensors.

The input `metrics` dict is the schema produced by BOTH the simulator and the live
engine (inference/live_state.py), so the policy sees the same features either way.
"""

import numpy as np

from src import config
from src.simulation import service_model as sm
from src.topology_intel.critical_path import critical_path_dfs

LAT_NORM_MS = 300.0
RPS_NORM = 50.0
WORK_NORM_MS = 50.0
INFLIGHT_NORM = 8.0


def symmetric_normalize(A: np.ndarray) -> np.ndarray:
    """A_hat = D^-1/2 (A + I) D^-1/2  - the standard GCN normalisation."""
    n = A.shape[0]
    A_self = A + np.eye(n, dtype=np.float32)
    deg = A_self.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    return (d_inv_sqrt[:, None] * A_self * d_inv_sqrt[None, :]).astype(np.float32)


def offered_load(m: dict) -> float:
    """Offered load / capacity from request rate, ready pods and CPU allocation.
    Unlike CPU utilisation (which saturates at 1.0 once a pod is throttled) this keeps
    growing with overload, so it tells the agent HOW overloaded a service is."""
    n = max(m.get("ready_replicas", m["replicas"]), 1)
    cap = n * sm.capacity_rps(m["cpu"], m.get("work_ms", config.WORK_MS))
    return m["requests"] / max(cap, 1e-6)


def node_features(services, metrics, entry, crit_path, crit_total) -> np.ndarray:
    on_path = set(crit_path)
    rows = []
    for s in services:
        m = metrics[s]
        rps_prev = m.get("rps_prev", m["requests"])
        trend = (m["requests"] - rps_prev) / max(rps_prev, 1.0)
        share = (m["latency_ms"] / crit_total) if (s in on_path and crit_total > 1e-9) else 0.0
        rows.append([
            min(m["latency_ms"] / LAT_NORM_MS, 5.0),
            min(m.get("latency_p95_ms", m["latency_ms"]) / LAT_NORM_MS, 5.0),
            min(max(m.get("utilization", 0.0), 0.0), 1.0),
            min(offered_load(m), 3.0) / 3.0,
            min(m["requests"] / RPS_NORM, 4.0),
            float(np.clip(trend, -1.0, 1.0)),
            m["replicas"] / config.MAX_REPLICAS,
            m.get("pending_replicas", 0) / config.MAX_REPLICAS,
            m["cpu"] / config.CPU_MAX,
            m["mem"] / config.MEM_MAX_MB,
            min(m.get("mem_utilization", 0.0), 1.5),
            min(m.get("inflight", 0.0) / INFLIGHT_NORM, 4.0),
            min(m.get("work_ms", config.WORK_MS) / WORK_NORM_MS, 2.0),
            1.0 if s == entry else 0.0,
            1.0 if s in on_path else 0.0,
            share,
            m.get("resize_cooldown", 0.0),
            m.get("scaledown_cooldown", 0.0),
        ])
    return np.asarray(rows, dtype=np.float32)


def build_sdg(services, call_graph) -> np.ndarray:
    """Static structural graph; edge weight = traffic share. Symmetrised for message
    passing: congestion is relevant context in both directions even though requests
    flow one way."""
    n = len(services)
    idx = {s: i for i, s in enumerate(services)}
    A = np.zeros((n, n), dtype=np.float32)
    for caller, targets in call_graph.items():
        if caller not in idx:
            continue
        for callee, share in targets:
            if callee in idx:
                A[idx[caller], idx[callee]] = share
                A[idx[callee], idx[caller]] = share
    return symmetric_normalize(A)


def build_ehg(services, call_graph, metrics, slo_ms) -> np.ndarray:
    """Dynamic latency graph: edge (caller, callee) = share x callee inclusive p95 / SLO.

    If catalog sends half its traffic to pricing and pricing's p95 is high, that edge
    lights up; if pricing is quick it goes quiet even though the SDG edge persists -
    the signal a purely structural graph cannot express."""
    n = len(services)
    idx = {s: i for i, s in enumerate(services)}
    A = np.zeros((n, n), dtype=np.float32)
    for caller, targets in call_graph.items():
        if caller not in idx:
            continue
        for callee, share in targets:
            if callee not in idx:
                continue
            # edge-specific percentile from traces when available, else the callee's inclusive p95
            p95 = metrics[caller].get("edge_p95", {}).get(
                callee, metrics[callee].get("latency_incl_p95_ms", metrics[callee]["latency_ms"]))
            w = share * min(p95 / max(slo_ms, 1e-6), 2.0)
            A[idx[caller], idx[callee]] = w
            A[idx[callee], idx[caller]] = w
    return symmetric_normalize(A)


def global_features(metrics, services, e2e, cost, budget, slo_ms, crit_total, entry) -> np.ndarray:
    utils = np.array([metrics[s].get("utilization", 0.0) for s in services])
    m_entry = metrics[entry]
    rps_prev = m_entry.get("rps_prev", m_entry["requests"])
    return np.array([
        min(e2e["p95_ms"] / slo_ms, 4.0) / 2.0,
        cost / budget,
        min(float(np.var(utils)), 1.0),
        min(e2e.get("error_rate", 0.0), 1.0),
        float(np.clip((m_entry["requests"] - rps_prev) / max(rps_prev, 1.0), -1.0, 1.0)),
        min(crit_total / slo_ms, 4.0) / 2.0,
    ], dtype=np.float32)


def build_observation(services, call_graph, entry, metrics, e2e, cost, budget, slo_ms):
    """-> dict(X, A_sdg, A_ehg, g, critical_path, critical_total, metrics, e2e, cost).
    The graph tensors feed the agent; `metrics`/`e2e` feed the rule-based baselines, so every
    controller reads one observation object."""
    crit_total, crit_path = critical_path_dfs(
        call_graph, entry, {s: metrics[s]["latency_ms"] for s in services})
    return {
        "X": node_features(services, metrics, entry, crit_path, crit_total),
        "A_sdg": build_sdg(services, call_graph),
        "A_ehg": build_ehg(services, call_graph, metrics, slo_ms),
        "g": global_features(metrics, services, e2e, cost, budget, slo_ms, crit_total, entry),
        "critical_path": crit_path,
        "critical_total": crit_total,
        "metrics": metrics, "e2e": e2e, "cost": cost,
    }
