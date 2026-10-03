"""
jaeger_client.py

A real client for the Jaeger Query HTTP API (the same API Jaeger's own UI
uses). Fetches genuine distributed traces and computes the critical path
from REAL span data - this is the production version of the POC's
per-trace critical path idea, now using actual request traces instead of
synthetic ones.

HOW THE CRITICAL PATH IS COMPUTED:
Every service in app.py creates a span literally named "cpu_work" around
its own local processing (see services/generic-service/app.py). That gives
us a clean, unambiguous measurement of each service's OWN latency
contribution per real request, with no need to subtract child-span time.
We then run the exact same DAG dynamic-programming approach as the POC
(critical_path.py) over the KNOWN, fixed call graph (config.CALL_GRAPH),
using these real per-service latencies instead of simulated ones.

Assumes Jaeger Query is reachable at config.JAEGER_QUERY_URL (by default,
`kubectl port-forward svc/jaeger 16686:16686 -n autoscaler-poc` - see README).
"""

import time

import requests

from src import config
from src.simulation.simulator import critical_path as find_critical_path


def fetch_recent_traces(service: str, lookback_seconds: int, limit: int) -> list[dict]:
    """Pulls the most recent traces for `service` from the real Jaeger backend."""
    end_us = int(time.time() * 1_000_000)
    start_us = end_us - lookback_seconds * 1_000_000
    try:
        resp = requests.get(
            f"{config.JAEGER_QUERY_URL}/api/traces",
            params={"service": service, "start": start_us, "end": end_us, "limit": limit},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.RequestException as e:
        print(f"[jaeger_client] failed to fetch traces: {e}")
        return []


def _extract_cpu_work_latencies(trace: dict) -> dict[str, float]:
    """
    Given one raw Jaeger trace, returns {service_name: cpu_work_duration_ms}
    by finding every span named "cpu_work" and resolving its owning service
    via the trace's processes map.
    """
    processes = trace.get("processes", {})
    latencies = {}
    for span in trace.get("spans", []):
        if span.get("operationName") != "cpu_work":
            continue
        process = processes.get(span.get("processID"), {})
        service_name = process.get("serviceName")
        duration_ms = span.get("duration", 0) / 1000.0  # Jaeger durations are in microseconds
        if service_name:
            latencies[service_name] = duration_ms
    return latencies


def get_real_per_service_latency(services: list[str], entry_service: str,
                                   lookback_seconds: int, limit: int,
                                   fallback: dict[str, float] | None = None) -> dict[str, float]:
    """
    Aggregates real per-service latency from recent actual traces, averaged
    across however many traces were sampled. Any service NOT seen in the
    sampled traces (e.g. a low-probability branch that wasn't hit) falls
    back to the provided `fallback` dict (normally Prometheus's window
    average for that service) so the critical-path DP always has a number
    to work with for every node.
    """
    traces = fetch_recent_traces(entry_service, lookback_seconds, limit)
    fallback = fallback or {}

    samples: dict[str, list[float]] = {s: [] for s in services}
    for trace in traces:
        for svc, latency_ms in _extract_cpu_work_latencies(trace).items():
            if svc in samples:
                samples[svc].append(latency_ms)

    result = {}
    for svc in services:
        if samples[svc]:
            result[svc] = sum(samples[svc]) / len(samples[svc])
        else:
            result[svc] = fallback.get(svc, 0.0)
    return result


def build_call_graph_from_traces(traces: list[dict], min_edge_count: int = 1):
    """
    Reconstructs the service call graph from raw Jaeger traces.

    An edge A -> B exists when a span owned by service B has, as its parent
    span, a span owned by a DIFFERENT service A (that is exactly what
    OpenTelemetry context propagation over HTTP produces). The edge weight is
    A's traffic share to B: calls(A->B) / total calls A made downstream.

    Returns (services, call_graph) with call_graph = {svc: [(callee, share)]},
    or None if the traces contain no cross-service calls or the discovered
    graph has a cycle (the critical-path DP requires a DAG).
    """
    edge_counts: dict[tuple[str, str], int] = {}
    seen_services: set[str] = set()

    for trace in traces:
        processes = trace.get("processes", {})
        spans = trace.get("spans", [])
        span_service = {}
        for sp in spans:
            svc = processes.get(sp.get("processID"), {}).get("serviceName")
            span_service[sp.get("spanID")] = svc
            if svc:
                seen_services.add(svc)
        for sp in spans:
            child_svc = span_service.get(sp.get("spanID"))
            if not child_svc:
                continue
            for ref in sp.get("references", []):
                if ref.get("refType") != "CHILD_OF":
                    continue
                parent_svc = span_service.get(ref.get("spanID"))
                if parent_svc and parent_svc != child_svc:
                    edge_counts[(parent_svc, child_svc)] = edge_counts.get((parent_svc, child_svc), 0) + 1

    edge_counts = {e: c for e, c in edge_counts.items() if c >= min_edge_count}
    if not edge_counts:
        return None

    out_total: dict[str, int] = {}
    for (a, _), c in edge_counts.items():
        out_total[a] = out_total.get(a, 0) + c

    services = sorted(seen_services | {x for e in edge_counts for x in e})
    graph: dict[str, list[tuple[str, float]]] = {s: [] for s in services}
    for (a, b), c in sorted(edge_counts.items()):
        graph[a].append((b, round(c / out_total[a], 4)))

    # DAG check (Kahn)
    indeg = {s: 0 for s in services}
    for targets in graph.values():
        for b, _ in targets:
            indeg[b] += 1
    queue = [s for s, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        n = queue.pop()
        visited += 1
        for b, _ in graph[n]:
            indeg[b] -= 1
            if indeg[b] == 0:
                queue.append(b)
    if visited != len(services):
        return None
    return services, graph


def discover_topology(entry_service: str, lookback_seconds: int, limit: int):
    """
    Zero-touch topology: reconstruct (services, entry, call_graph) from live
    Jaeger traces rooted at `entry_service`. Returns None when Jaeger has no
    usable data, so callers fall back to the declared config.CALL_GRAPH.
    """
    traces = fetch_recent_traces(entry_service, lookback_seconds, limit)
    built = build_call_graph_from_traces(traces)
    if built is None:
        return None
    services, graph = built
    return services, entry_service, graph


if __name__ == "__main__":
    from src.data_collection.prometheus_client import get_all_service_metrics

    prom_metrics = get_all_service_metrics(config.SERVICES)
    prom_fallback = {svc: m["latency_ms"] for svc, m in prom_metrics.items()}

    real_latencies = get_real_per_service_latency(
        config.SERVICES, config.ENTRY_SERVICE,
        config.JAEGER_LOOKBACK_SECONDS, config.JAEGER_TRACE_LIMIT,
        fallback=prom_fallback,
    )
    print("Per-service latency (from real traces, Prometheus fallback where missing):")
    for svc, lat in real_latencies.items():
        print(f"  {svc:20s} {lat:7.2f} ms")

    total, path = find_critical_path(config.CALL_GRAPH, config.ENTRY_SERVICE, real_latencies)
    print(f"\nCritical path: {' -> '.join(path)}  (total {total:.1f} ms)")

    found = discover_topology(config.ENTRY_SERVICE, config.JAEGER_LOOKBACK_SECONDS, 200)
    if found:
        print("\nTopology discovered from traces:")
        for caller, targets in found[2].items():
            for callee, share in targets:
                print(f"  {caller} -> {callee}  share={share}")
    else:
        print("\n(no topology discoverable from current traces)")
