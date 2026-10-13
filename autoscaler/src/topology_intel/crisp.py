"""
crisp.py

Critical-path ANALYSIS in the style of Uber's CRISP (https://www.uber.com/us/en/blog/crisp-critical-path-analysis-for-microservice-architectures/):
every sampled request gets its own critical path, and the time is aggregated per service - not one chain for the whole graph.

  per request   the end-to-end time is attributed to services by walking the span tree backwards from the end: the
                last-finishing call is blamed for the time it ran, then whatever ran before it, recursively; calls that
                overlap a later call (parallel) are off the path    (data_collection.otel_traces.critical_time_by_service)
  aggregation   a service's CRITICAL SHARE = the mean, over requests, of its blamed time / the request's downstream time.
                Three views: all requests, the slow tail (where SLO violations live) and per request type (root operation)
  output        services ranked by share, the smallest set that covers a given fraction of the time, and a chain for display

The entry service is left out of the shares: it is on every path by definition, and the time spent in services that emit no
traces is blamed on their traced caller - both would swamp the signal.

What this adds over the graph-level depth-first search of the proposal (topology_intel/critical_path.py):
parallel calls are handled exactly, the weights are time actually spent on the critical path (not mean latency), the tail can be
weighted, and different request types can have different critical services. See evaluation/critical_path_eval.py for how much.
"""

import statistics as st
from collections import defaultdict

from src.data_collection import otel_traces

MIN_DOWNSTREAM_MS = 5.0        # a request that never left the entry service carries no signal


class Request:
    __slots__ = ("op", "total_ms", "blame", "downstream_ms")

    def __init__(self, op, total_ms, blame, downstream_ms):
        self.op, self.total_ms, self.blame, self.downstream_ms = op, total_ms, blame, downstream_ms


def _root_operation(trace):
    spans = trace.get("spans", [])
    roots = [s for s in spans if not s.get("references")]
    root = min(roots or spans, key=lambda s: s["startTime"], default=None)
    return root["operationName"] if root else "?"


def requests_from_traces(traces, entry):
    """[Request] with the per-service blame (entry excluded) of every trace that left the entry service."""
    out = []
    for t in traces:
        blame, total = otel_traces.critical_time_by_service(t)
        blame = {s: v for s, v in blame.items() if s != entry}
        down = sum(blame.values())
        if down >= MIN_DOWNSTREAM_MS:
            out.append(Request(_root_operation(t), total, blame, down))
    return out


def shares(requests):
    """{service: mean over the requests of (blamed time / downstream time)}; sums to 1 over the services."""
    acc = defaultdict(float)
    for r in requests:
        for s, v in r.blame.items():
            acc[s] += v / r.downstream_ms
    n = max(len(requests), 1)
    return {s: v / n for s, v in sorted(acc.items(), key=lambda x: -x[1])}


def tail(requests, quantile=0.8):
    """The slowest (1 - quantile) of the requests by end-to-end time."""
    ranked = sorted(requests, key=lambda r: r.total_ms)
    return ranked[int(quantile * len(ranked)):] or ranked


class CriticalProfile:
    """Critical shares of one window of requests: overall, slow tail, per request type."""

    def __init__(self, requests, tail_quantile=0.8, min_type_requests=5):
        self.n = len(requests)
        self.overall = shares(requests)
        self.tail = shares(tail(requests, tail_quantile)) if requests else {}
        by_type = defaultdict(list)
        for r in requests:
            by_type[r.op].append(r)
        self.by_type = {op: shares(rs) for op, rs in by_type.items() if len(rs) >= min_type_requests}
        self.type_counts = {op: len(rs) for op, rs in by_type.items()}

    def critical_set(self, coverage=0.9, view="overall", op=None):
        """The smallest set of services, taken in order of share, that explains at least `coverage` of the downstream time."""
        table = (self.by_type.get(op) if op else None) or getattr(self, view)
        out, acc = [], 0.0
        for s, v in table.items():
            out.append(s)
            acc += v
            if acc >= coverage:
                break
        return out

    def top(self, k, view="overall", op=None):
        table = (self.by_type.get(op) if op else None) or getattr(self, view)
        return list(table)[:k]


def chain_from_shares(call_graph, entry, weights):
    """A single chain for display: the depth-first search of the proposal, weighting each service by its critical share."""
    from src.topology_intel.critical_path import critical_path_dfs
    return critical_path_dfs(call_graph, entry, weights)[1]


def coverage(requests, services):
    """Mean over requests of the share of downstream time spent in `services` (higher is better)."""
    on = set(services)
    return st.mean(sum(v for s, v in r.blame.items() if s in on) / r.downstream_ms for r in requests) if requests else float("nan")


def top1(requests, services):
    """Fraction of requests whose single most-blamed service is in `services`."""
    on = set(services)
    return sum(1 for r in requests if r.blame and max(r.blame, key=r.blame.get) in on) / max(len(requests), 1)
