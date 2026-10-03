"""
graph_builder.py

Builds BOTH graphs from the original research design:

  SDG (Service Dependency Graph)
      Structure: who calls whom. Edge weight = traffic share.
      Changes only when the topology itself changes.

  EHG (Execution History Graph)
      Runtime behaviour: how much latency pressure actually flows along
      each edge RIGHT NOW. Edge weight is recomputed every window from
      observed per-service latency.

v1 collapsed these into one graph (latency lived only in node features).
That threw away the *edge-specific* signal - "how much does A's latency
currently depend on B" - which is exactly what the EHG was for. Both are
now kept separate and fed into separate GCN branches (see dual_gcn.py).

Everything here is topology-size-agnostic: pass 6 services or 60, the
returned tensors just change shape accordingly.
"""

import numpy as np
import torch

# normalization constants
LATENCY_NORM = 500.0
REQUEST_NORM = 200.0


def _symmetric_normalize(A: np.ndarray) -> torch.Tensor:
    """A_hat = D^-1/2 (A + I) D^-1/2  — the standard GCN normalization."""
    n = A.shape[0]
    A_self = A + np.eye(n, dtype=np.float32)
    deg = A_self.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A_self @ D, dtype=torch.float32)


def build_node_features(services: list[str], metrics: dict[str, dict],
                         entry_service: str, max_replicas: int) -> torch.Tensor:
    """
    (N, 5) tensor — one row per service, in `services` order.
    Features are all per-node and scale-free, so N can be anything.
      [latency_norm, utilization, rps_norm, replicas_norm, is_entry]
    """
    rows = []
    for svc in services:
        m = metrics[svc]
        replicas = max(1, m.get("replicas", 1))
        rows.append([
            m["latency_ms"] / LATENCY_NORM,
            min(m.get("utilization", 0.0), 2.0) / 2.0,
            m["requests"] / REQUEST_NORM,
            replicas / max_replicas,
            1.0 if svc == entry_service else 0.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def build_sdg(services: list[str], call_graph: dict) -> torch.Tensor:
    """
    Static structural graph. Edge weight = traffic share (how much of the
    caller's load goes to that callee). Symmetrized for message passing:
    congestion is relevant context in both directions even though traffic
    flows one way.
    """
    n = len(services)
    idx = {s: i for i, s in enumerate(services)}
    A = np.zeros((n, n), dtype=np.float32)
    for caller, targets in call_graph.items():
        if caller not in idx:
            continue
        for callee, weight in targets:
            if callee not in idx:
                continue
            A[idx[caller], idx[callee]] = weight
            A[idx[callee], idx[caller]] = weight
    return _symmetric_normalize(A)


def build_ehg(services: list[str], call_graph: dict,
               metrics: dict[str, dict]) -> torch.Tensor:
    """
    DYNAMIC latency-pressure graph, rebuilt every window.

    Edge (caller -> callee) weight = how much of the caller's observed
    latency is plausibly attributable to that callee: the callee's own
    latency, scaled by the fraction of traffic routed to it, normalized
    across the caller's outgoing edges.

    Intuition: if catalog sends half its traffic to pricing and pricing is
    slow, the catalog->pricing edge lights up. If pricing is fast, that
    edge goes quiet even though the SDG edge still exists. This is the
    signal a purely structural graph cannot express.
    """
    n = len(services)
    idx = {s: i for i, s in enumerate(services)}
    A = np.zeros((n, n), dtype=np.float32)

    for caller, targets in call_graph.items():
        if caller not in idx or not targets:
            continue
        contributions = []
        for callee, traffic_share in targets:
            if callee not in idx:
                contributions.append(0.0)
                continue
            callee_latency = metrics[callee]["latency_ms"]
            contributions.append(traffic_share * callee_latency)

        total = sum(contributions)
        for (callee, _), contrib in zip(targets, contributions):
            if callee not in idx:
                continue
            # normalized share of this caller's downstream latency pressure
            w = (contrib / total) if total > 1e-9 else 0.0
            A[idx[caller], idx[callee]] = w
            A[idx[callee], idx[caller]] = w

    return _symmetric_normalize(A)


def build_all(services: list[str], call_graph: dict, metrics: dict,
               entry_service: str, max_replicas: int):
    """Convenience: returns (X, A_sdg, A_ehg) ready for the dual GCN."""
    X = build_node_features(services, metrics, entry_service, max_replicas)
    return X, build_sdg(services, call_graph), build_ehg(services, call_graph, metrics)
