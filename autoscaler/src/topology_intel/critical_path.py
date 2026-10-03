"""
critical_path.py

Critical-path analysis over the service call graph.

The "critical path" is the chain of service calls, from the entry service down to
a leaf, whose summed per-service latency is largest - the chain that dominates
end-to-end latency, and therefore the one a latency-aware autoscaler must
protect. The proposal specifies a depth-first search over the weighted graph;
`critical_path_dfs` is exactly that.

`path_distribution` additionally enumerates every root-to-leaf chain together
with the probability a request takes it (the product of the traffic shares along
the chain). The generic service routes each request to ONE downstream chosen by
those shares, so the latency a client actually experiences is a probability
mixture over chains, not the single worst chain. The simulator uses this mixture
to compute realistic end-to-end percentiles.

Both functions require an acyclic graph (the benchmark is a DAG); a cycle is
reported rather than looped on forever.
"""

import sys

sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))


def critical_path_dfs(call_graph: dict, entry: str, latencies: dict):
    """Depth-first search for the longest latency chain.

    Returns (total_latency, [entry, ..., leaf]). `latencies[s]` is service s's own
    contribution (missing -> 0). Memoised per node, so shared callees in a DAG are
    evaluated once.
    """
    best: dict[str, tuple[float, list]] = {}
    on_stack: set[str] = set()

    def dfs(node):
        if node in best:
            return best[node]
        if node in on_stack:
            raise ValueError(f"cycle through {node!r}: critical path needs a DAG")
        on_stack.add(node)
        own = float(latencies.get(node, 0.0))
        tail_total, tail_path = 0.0, []
        for callee, _share in call_graph.get(node, []):
            total, path = dfs(callee)
            if total > tail_total or not tail_path:
                tail_total, tail_path = total, path
        on_stack.discard(node)
        best[node] = (own + tail_total, [node] + tail_path)
        return best[node]

    total, path = dfs(entry)
    return total, list(path)


def path_distribution(call_graph: dict, entry: str, max_paths: int = 512):
    """All root-to-leaf chains with their probabilities.

    Returns [(path, probability)] with probabilities summing to 1 (renormalised if
    `max_paths` truncated a very wide graph). Chains of equal route are merged.
    """
    out: list[tuple[list, float]] = []

    def walk(node, path, prob):
        if len(out) >= max_paths:
            return
        targets = call_graph.get(node, [])
        if not targets:
            out.append((path, prob))
            return
        for callee, share in targets:
            if callee in path:
                raise ValueError(f"cycle through {callee!r}")
            walk(callee, path + [callee], prob * share)

    walk(entry, [entry], 1.0)
    total = sum(p for _, p in out) or 1.0
    return [(p, pr / total) for p, pr in out]


def reachable(call_graph: dict, entry: str) -> list:
    """Services reachable from entry, in BFS order (entry first)."""
    order, seen, frontier = [entry], {entry}, [entry]
    while frontier:
        nxt = []
        for node in frontier:
            for callee, _ in call_graph.get(node, []):
                if callee not in seen:
                    seen.add(callee)
                    order.append(callee)
                    nxt.append(callee)
        frontier = nxt
    return order
