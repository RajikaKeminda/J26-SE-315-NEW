"""
simulator.py

Fast cluster simulator, calibrated against real measurements taken from the
actual deployed services (see README: 1 request ~135ms, 12 concurrent
against MAX_CONCURRENCY=4 -> ~1.1s).

v2 addition: RANDOM TOPOLOGY GENERATION. v1 only ever simulated the one
fixed 6-service graph, which meant a shared policy trained on it would
quietly overfit to that exact shape. Training across many random topologies
(different sizes, different fan-out) is what actually teaches the shared
policy a general rule rather than six memorized ones.
"""

from dataclasses import dataclass, field

import numpy as np

from src import config


# ---------------------------------------------------------------------------
# Topology generation
# ---------------------------------------------------------------------------
def random_topology(rng, min_services=4, max_services=9):
    """Layered random DAG: 1 entry -> mid tier -> leaf tier. Always acyclic."""
    n = int(rng.integers(min_services, max_services + 1))
    services = [f"svc-{i}" for i in range(n)]
    entry = services[0]

    n_mid = max(1, int(n * 0.4))
    mid = services[1:1 + n_mid]
    leaves = services[1 + n_mid:]
    if not leaves:
        leaves = [mid.pop()]

    graph = {s: [] for s in services}
    for s in mid:
        graph[entry].append([s, 0.0])
    for s in mid:
        k = int(rng.integers(1, min(2, len(leaves)) + 1))
        for t in rng.choice(leaves, size=k, replace=False):
            graph[s].append([str(t), 0.0])
    # any orphan gets attached to the entry point
    called = {c for ts in graph.values() for c, _ in ts}
    for s in services[1:]:
        if s not in called:
            graph[entry].append([s, 0.0])

    # normalize outgoing traffic shares to sum to 1.0 per node
    for s, targets in graph.items():
        if not targets:
            continue
        w = rng.uniform(0.5, 1.0, size=len(targets))
        w /= w.sum()
        for t, val in zip(targets, w):
            t[1] = float(val)

    return services, entry, {k: [(c, w) for c, w in v] for k, v in graph.items()}


def fixed_topology():
    """The real deployed 6-service topology from config."""
    return config.SERVICES, config.ENTRY_SERVICE, config.CALL_GRAPH


# ---------------------------------------------------------------------------
# Traffic patterns
# ---------------------------------------------------------------------------
def generate_traffic(pattern, length, base_rps, rng):
    t = np.arange(length)
    if pattern == "random":
        pattern = rng.choice(["normal", "burst", "diurnal", "rampup"])

    if pattern == "normal":
        traffic = base_rps + rng.normal(0, base_rps * 0.05, length)
    elif pattern == "burst":
        traffic = np.full(length, base_rps, dtype=float)
        s = length // 3
        e = s + max(2, length // 4)
        peak = rng.uniform(4.0, 8.0)
        traffic[s:e] = base_rps * peak
        ramp = max(1, (e - s) // 3)
        traffic[max(0, s - ramp):s] = np.linspace(base_rps, base_rps * peak, min(ramp, s))
        traffic[e:e + ramp] = np.linspace(base_rps * peak, base_rps, ramp)[:max(0, len(traffic) - e)]
        traffic += rng.normal(0, base_rps * 0.03, length)
    elif pattern == "diurnal":
        traffic = base_rps * (1.5 + 1.2 * np.sin(2 * np.pi * t / length - np.pi / 2))
        traffic += rng.normal(0, base_rps * 0.04, length)
    elif pattern == "rampup":
        traffic = np.linspace(base_rps * 0.5, base_rps * 5.0, length)
        traffic += rng.normal(0, base_rps * 0.04, length)
    else:
        raise ValueError(pattern)
    return np.clip(traffic, 1.0, None)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
@dataclass
class Simulator:
    services: list
    entry_service: str
    call_graph: dict
    replicas: dict = field(default_factory=dict)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.replicas = {s: 2 for s in self.services}

    def _propagate(self, external_rps):
        incoming = {s: 0.0 for s in self.services}
        incoming[self.entry_service] = external_rps
        frontier, seen = [self.entry_service], {self.entry_service}
        while frontier:
            nxt = []
            for node in frontier:
                for callee, weight in self.call_graph.get(node, []):
                    incoming[callee] += incoming[node] * weight
                    if callee not in seen:
                        seen.add(callee)
                        nxt.append(callee)
            frontier = nxt
        return incoming

    def step(self, external_rps):
        incoming = self._propagate(external_rps)
        out = {}
        for s in self.services:
            cap = self.replicas[s] * config.MAX_CONCURRENCY_PER_REPLICA * (1000.0 / config.WORK_MS)
            util = incoming[s] / max(cap, 1e-6)
            base = config.WORK_MS
            if util <= 0.7:
                latency = base * (1.0 + util * 0.5)
            else:
                latency = base * (1.35 + (util - 0.7) * 12.0)
            out[s] = {
                "requests": incoming[s],
                "latency_ms": latency,
                "replicas": self.replicas[s],
                "utilization": util,
            }
        return out

    def apply(self, service, delta):
        self.replicas[service] = int(np.clip(
            self.replicas[service] + delta, config.MIN_REPLICAS, config.MAX_REPLICAS))


# ---------------------------------------------------------------------------
# Critical path (exact DP over the DAG)
# ---------------------------------------------------------------------------
def critical_path(call_graph, entry, latencies):
    order, seen, frontier = [entry], {entry}, [entry]
    while frontier:
        nxt = []
        for node in frontier:
            for c, _ in call_graph.get(node, []):
                if c not in seen:
                    seen.add(c); order.append(c); nxt.append(c)
        frontier = nxt

    preds = {s: [] for s in order}
    for node in order:
        for c, _ in call_graph.get(node, []):
            if c in preds:
                preds[c].append(node)

    best, prev = {}, {}
    for node in order:
        own = latencies.get(node, 0.0)
        if not preds[node]:
            best[node], prev[node] = own, None
        else:
            p = max(preds[node], key=lambda x: best[x])
            best[node], prev[node] = own + best[p], p

    end = max(best, key=best.get)
    path, node = [], end
    while node is not None:
        path.append(node); node = prev[node]
    return best[end], list(reversed(path))
