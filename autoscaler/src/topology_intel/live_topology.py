"""
live_topology.py

The service dependency graph (SDG) learned ONLY from live distributed traces - nothing is declared.
It is what Jaeger's "System Architecture" view does, plus the properties a controller needs:

  * edges are accumulated from every trace window (caller -> callee = a server span whose nearest
    ancestor in another service is the caller; see otel_traces.analyse_trace)
  * observed call counts DECAY (half-life) and an edge that has not been seen for `ttl_s` is dropped,
    so the graph shrinks when a dependency disappears as well as growing when one appears
  * a window with NO traces at all (idle system) freezes the graph instead of erasing it
  * traffic share of an edge = its decayed count / the caller's total decayed outgoing count
  * the result is always a DAG rooted at the entry service (the critical-path DP needs one): if observed
    calls form a cycle the weakest edge of the cycle is ignored

Only services reachable from the entry service are part of the graph.
"""

import time
from collections import defaultdict

from src.data_collection import otel_traces


class LiveTopology:
    def __init__(self, entry, half_life_s=120.0, ttl_s=300.0, min_calls=2.0):
        self.entry = entry
        self.half_life_s, self.ttl_s, self.min_calls = half_life_s, ttl_s, min_calls
        self.counts: dict[tuple[str, str], float] = {}
        self.last_seen: dict[tuple[str, str], float] = {}
        self.updated_at: float | None = None
        self.last_window_traces = 0
        self._t = None

    def update(self, traces, now=None):
        """Fold one window of Jaeger traces into the graph. Returns True if the graph's structure changed."""
        now = time.time() if now is None else now
        before = self.edges_set()
        self.last_window_traces = len(traces)
        if traces:                                              # idle window: keep what we know
            if self._t is not None:
                decay = 0.5 ** (max(0.0, now - self._t) / self.half_life_s)
                for e in self.counts:
                    self.counts[e] *= decay
            self._t = now
            new = defaultdict(int)
            for t in traces:
                for r in otel_traces.analyse_trace(t):
                    if r["parent_service"] and r["parent_service"] != r["service"]:
                        new[(r["parent_service"], r["service"])] += 1
            for e, c in new.items():
                self.counts[e] = self.counts.get(e, 0.0) + c
                self.last_seen[e] = now
            for e in [e for e, ts in self.last_seen.items() if now - ts > self.ttl_s]:
                self.counts.pop(e, None)
                self.last_seen.pop(e, None)
            self.updated_at = now
        return self.edges_set() != before

    # -- views -----------------------------------------------------------
    def _live_edges(self):
        return {e: c for e, c in self.counts.items() if c >= self.min_calls}

    def edges_set(self):
        return frozenset(self._reachable_dag(self._live_edges()))

    def _reachable_dag(self, edges):
        edges = dict(edges)
        # 1) keep only what the entry service can reach
        adj = defaultdict(list)
        for (a, b) in edges:
            adj[a].append(b)
        seen, stack = {self.entry}, [self.entry]
        while stack:
            for b in adj[stack.pop()]:
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        edges = {e: c for e, c in edges.items() if e[0] in seen and e[1] in seen}
        # 2) break cycles by ignoring the weakest edge among the nodes Kahn's algorithm cannot order
        while True:
            nodes = {n for e in edges for n in e}
            indeg = {n: 0 for n in nodes}
            for (_, b) in edges:
                indeg[b] += 1
            queue, order = [n for n, d in indeg.items() if d == 0], []
            out = defaultdict(list)
            for (a, b) in edges:
                out[a].append(b)
            while queue:
                n = queue.pop()
                order.append(n)
                for b in out[n]:
                    indeg[b] -= 1
                    if indeg[b] == 0:
                        queue.append(b)
            if len(order) == len(nodes):
                return edges
            stuck = {n for n, d in indeg.items() if d > 0}
            cyc = [e for e in edges if e[0] in stuck and e[1] in stuck]
            del edges[min(cyc, key=lambda e: edges[e])]

    def services(self):
        edges = self._reachable_dag(self._live_edges())
        return [self.entry] + sorted({n for e in edges for n in e} - {self.entry})

    def call_graph(self, services=None):
        """{caller: [(callee, share)]} over `services` (default: everything discovered); shares sum to 1."""
        edges = self._reachable_dag(self._live_edges())
        keep = set(services) if services is not None else set(self.services())
        edges = {e: c for e, c in edges.items() if e[0] in keep and e[1] in keep}
        total = defaultdict(float)
        for (a, _), c in edges.items():
            total[a] += c
        graph = {s: [] for s in (services if services is not None else self.services())}
        for (a, b), c in sorted(edges.items()):
            graph[a].append((b, round(c / total[a], 6)))
        for a, targets in graph.items():                        # make shares sum to exactly 1
            if targets:
                s = sum(w for _, w in targets)
                graph[a] = [(b, w / s) for b, w in targets]
        return graph

    def observed_calls(self):
        return {e: c for e, c in self._reachable_dag(self._live_edges()).items()}
