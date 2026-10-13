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


def cross_validate(traces, entry="ts-gateway-service", folds=5, repeats=20, seed=0):
    """Held-out comparison. In each fold the graph and every weight are learned from the training traces only, and the
    result is scored on requests it never saw. Every method names the SAME number of downstream services (the length of the
    current path), so no method wins by naming more of them. Returns {method: {coverage, top1, tail_coverage, ...}} with the mean
    and the standard deviation over the repeats."""
    import random
    from src.topology_intel import crisp
    rng = random.Random(seed)
    names = ["current: DFS on mean own latency", "DFS weighted by critical share", "DFS weighted by tail critical share",
             "top-k by mean own latency", "top-k critical share", "top-k tail critical share",
             "top-k critical share per request type", "random chain (expected)"]
    per_repeat = {n: {"coverage": [], "top1": [], "tail_coverage": []} for n in names}
    for _ in range(repeats):
        idx = list(range(len(traces)))
        rng.shuffle(idx)
        acc = {n: {"coverage": [], "top1": [], "tail_coverage": [], "w": []} for n in names}
        for f in range(folds):
            test_idx = set(idx[f::folds])
            train = [traces[i] for i in idx if i not in test_idx]
            test = crisp.requests_from_traces([traces[i] for i in test_idx], entry)
            if len(test) < 3:
                continue
            services, graph = learn(train, entry)
            own, _ = service_weights(train, entry)
            prof = crisp.CriticalProfile(crisp.requests_from_traces(train, entry))
            if not prof.overall:
                continue
            current = [x for x in critical_path_dfs(graph, entry, own)[1] if x != entry]
            k = max(len(current), 1)
            by_own = [x for x, _ in sorted(own.items(), key=lambda t: -t[1]) if x != entry][:k]
            def chain(w):
                """the chain found by the depth-first search with weights w, reduced to its k highest-weight services (never by position:
                cutting a chain after k nodes could drop the service that matters, which sits at its end)"""
                c = [x for x in crisp.chain_from_shares(graph, entry, w) if x != entry]
                return sorted(c, key=lambda x: -w.get(x, 0.0))[:k]
            sets = {
                names[0]: lambda r: current,
                names[1]: lambda r: chain(prof.overall),
                names[2]: lambda r: chain(prof.tail),
                names[3]: lambda r: by_own,
                names[4]: lambda r: prof.top(k),
                names[5]: lambda r: prof.top(k, "tail"),
                names[6]: lambda r: prof.top(k, op=r.op),
            }
            test_tail = crisp.tail(test)
            for n, pick in sets.items():
                acc[n]["coverage"].append(st.mean(crisp.coverage([r], pick(r)) for r in test))
                acc[n]["top1"].append(st.mean(crisp.top1([r], pick(r)) for r in test))
                acc[n]["tail_coverage"].append(st.mean(crisp.coverage([r], pick(r)) for r in test_tail))
                acc[n]["w"].append(len(test))
            dist = path_distribution(graph, entry)
            rn = names[7]
            acc[rn]["coverage"].append(sum(p * crisp.coverage(test, [x for x in path if x != entry]) for path, p in dist))
            acc[rn]["top1"].append(sum(p * crisp.top1(test, [x for x in path if x != entry]) for path, p in dist))
            acc[rn]["tail_coverage"].append(sum(p * crisp.coverage(test_tail, [x for x in path if x != entry]) for path, p in dist))
            acc[rn]["w"].append(len(test))
        for n in names:
            w = acc[n]["w"]
            if w:
                for key in ("coverage", "top1", "tail_coverage"):
                    per_repeat[n][key].append(sum(a_ * b_ for a_, b_ in zip(acc[n][key], w)) / sum(w))
    return {n: {key: (st.mean(v), st.pstdev(v)) for key, v in d.items() if v} for n, d in per_repeat.items()}


def print_cross_validation(res):
    print(f"{'method (same number of services for all)':44} {'coverage':>14} {'top-1 hit':>14} {'tail coverage':>14}")
    for n, d in res.items():
        print(f"{n:44} " + " ".join(f"{d[k][0] * 100:6.1f}% +-{d[k][1] * 100:4.1f}" for k in ("coverage", "top1", "tail_coverage")))


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
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/trainticket_traces.json"
    main(path)
    print("\nheld-out (5-fold x 20 repeats; graph and weights learned on the training traces only):")
    print_cross_validation(cross_validate(json.load(open(path))))
