"""
critical_path_eval.py

How good is the agent's critical path? It is checked against what real requests actually waited for.

  ground truth   per request, the end-to-end time is attributed to services by walking the span tree backwards from
                 the end (otel_traces.critical_time_by_service); the values sum to the request's duration
  agent          the call graph learned from the same traces (LiveTopology) + the depth-first search of the proposal
                 (critical_path_dfs) with each service's measured own latency as its weight
  variant        the same search weighting each service by its EXPECTED contribution per entry request
                 (own latency x calls per entry request): a service called by 5 % of requests matters 5 % as much
  baselines      a random root-to-leaf chain (expected value over all chains, weighted by traffic) and the deepest chain

Metrics, over all traces, counted on DOWNSTREAM time only (the entry service is on every path by definition, and time
spent in services without traces is blamed on their traced caller, which would inflate both numbers):
  coverage   mean share of a request's downstream time that is spent in the services of the path (higher is better)
  top-1 hit  fraction of requests whose single most-blamed downstream service lies on the path

    python -m src.evaluation.critical_path_eval tests/fixtures/trainticket_traces.json
"""

import json
import statistics as st
import sys
from collections import defaultdict

from src.data_collection import otel_traces
from src.topology_intel.critical_path import critical_path_dfs, path_distribution
from src.topology_intel.live_topology import LiveTopology


def _short(name):
    name = name[3:] if name.startswith("ts-") else name
    return name[:-8] if name.endswith("-service") else name


def _blames(traces, entry):
    """[(downstream blame {service: ms}, downstream total ms)] - the entry service is excluded (see module docstring)."""
    out = []
    for t in traces:
        b, _ = otel_traces.critical_time_by_service(t)
        b = {k: v for k, v in b.items() if k != entry}
        total = sum(b.values())
        if total >= 5.0:                                   # ms; requests that never left the entry carry no signal
            out.append((b, total))
    return out


def _coverage(blames, path):
    on = set(path)
    return st.mean(sum(v for s, v in b.items() if s in on) / tot for b, tot in blames)


def _top1(blames, path):
    on = set(path)
    return sum(1 for b, _ in blames if max(b, key=b.get) in on) / len(blames)


def _shares(blames):
    share = defaultdict(float)
    for b, tot in blames:
        for s, v in b.items():
            share[s] += v / tot / len(blames)
    return dict(sorted(share.items(), key=lambda x: -x[1]))


def learn(traces, entry, min_calls=2.0):
    lt = LiveTopology(entry, min_calls=min_calls)
    lt.update(traces, now=1.0)
    return lt.services(), lt.call_graph()


def service_weights(traces, entry):
    stats = otel_traces.window_stats(traces, window_s=1.0)
    n_entry = max(stats[entry]["samples"], 1)
    own = {s: v["own_latency_ms"] for s, v in stats.items()}
    expected = {s: v["own_latency_ms"] * v["samples"] / n_entry for s, v in stats.items()}
    return own, expected


def deepest_path(graph, entry):
    best = (-1, [entry])
    for path, _ in path_distribution(graph, entry):
        if len(path) > best[0]:
            best = (len(path), path)
    return best[1]


def random_chain_baseline(blames, graph, entry):
    """Expected coverage / top-1 of a chain drawn at random with the traffic shares (what 'any chain' would score)."""
    dist = path_distribution(graph, entry)
    return {"coverage": sum(p * _coverage(blames, path) for path, p in dist),
            "top1": sum(p * _top1(blames, path) for path, p in dist)}


def evidence(traces, path, entry, graph=None):
    """Live proof for the dashboard: how well does `path` (the agent's current critical path) explain where real requests
    spent their downstream time in this window? None when no request left the entry service."""
    blames = _blames(traces, entry)
    if not blames:
        return None
    rnd = None
    if graph:
        try:
            rnd = {k: round(v, 3) for k, v in random_chain_baseline(blames, graph, entry).items()}
        except ValueError:
            rnd = None
    slowest = max(traces, key=lambda t: max((s["duration"] for s in t.get("spans", [])), default=0))
    return {"requests": len(blames), "entry_only_requests": len(traces) - len(blames),
            "coverage": round(_coverage(blames, path), 3), "top1": round(_top1(blames, path), 3), "random": rnd,
            "share": {s: round(v, 3) for s, v in _shares(blames).items()}, "slowest_trace": slowest.get("traceID")}


def evaluate(traces, entry="ts-gateway-service"):
    services, graph = learn(traces, entry)
    own, expected = service_weights(traces, entry)
    paths = {
        "agent: DFS on own latency": critical_path_dfs(graph, entry, own)[1],
        "variant: DFS on expected contribution": critical_path_dfs(graph, entry, expected)[1],
        "baseline: deepest chain": deepest_path(graph, entry),
    }
    blames = _blames(traces, entry)
    rows = [{"method": name, "path": path, "coverage": _coverage(blames, path), "top1": _top1(blames, path)}
            for name, path in paths.items()]
    rows.append({"method": "baseline: random chain (expected)", "path": None,
                 **random_chain_baseline(blames, graph, entry)})
    return {"traces": len(blames), "traces_without_downstream_time": len(traces) - len(blames),
            "services": len(services), "edges": sum(len(v) for v in graph.values()),
            "rows": rows, "blame_share": _shares(blames)}


def main(path):
    r = evaluate(json.load(open(path)))
    print(f"{r['traces']} real requests with downstream time ({r['traces_without_downstream_time']} more never left the "
          f"entry service); graph learned from them: {r['services']} services, {r['edges']} edges\n")
    print(f"{'method':40} {'coverage':>9} {'top-1 hit':>10}   path")
    for x in r["rows"]:
        print(f"{x['method']:40} {x['coverage'] * 100:8.1f}% {x['top1'] * 100:9.1f}%   "
              f"{' -> '.join(_short(s) for s in x['path']) if x['path'] else '(mean over all chains)'}")
    print("\nshare of DOWNSTREAM request time by service (ground truth):",
          ", ".join(f"{_short(s)} {v * 100:.0f}%" for s, v in list(r["blame_share"].items())[:6]))
    return r


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/trainticket_traces.json")
