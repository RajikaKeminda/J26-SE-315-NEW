"""
trainticket_metrics.py

The metrics source for applications whose services export NO request metrics (Train-Ticket).
It returns exactly the schema of prometheus_client.collect(), so LiveStateBuilder and the rest of
the pipeline are unchanged:

    resources (Prometheus)   cAdvisor pod-level CPU / working-set memory, kube-state-metrics
                             pod counts, aggregated per Deployment
    request stats (Jaeger)   per-service request rate, own and inclusive latency percentiles,
                             error rate - from the OpenTelemetry server spans (otel_traces.py)

The service of a pod is its Deployment name, taken from the pod name (<deployment>-<rs-hash>-<id>).
Failure semantics are the same as prometheus_client: a failed query raises QueryFailed, so the
control loop holds instead of reading missing data as zero load.
"""

from src import config
from src.data_collection import jaeger_client, otel_traces
from src.data_collection.prometheus_client import QueryFailed, _by_service, _instant_query

POD_TO_SERVICE = r'(.+)-[a-z0-9]+-[a-z0-9]+'          # ts-order-service-848589b9d8-7rwfs -> ts-order-service

last_traces: list = []                                 # the trace window of the latest collect() (reused for edges)


def _per_service(expr: str, w: str) -> str:
    sel = f'namespace="{config.NAMESPACE}",pod!="",container=""'
    return (f'label_replace({expr.format(sel=sel, w=w)}, "service", "$1", "pod", "{POD_TO_SERVICE}")')


def resource_queries(w: str) -> dict:
    cpu = _per_service("rate(container_cpu_usage_seconds_total{{{sel}}}[{w}])", w)
    mem = _per_service("container_memory_working_set_bytes{{{sel}}}", w)
    return {
        "cpu_cores": f"sum by (service) ({cpu})",
        "mem_bytes": f"avg by (service) ({mem})",
        "pods": f"count by (service) ({mem})",
    }


def collect(window: str | None = None) -> dict[str, dict]:
    """One window of per-service measurements; raises QueryFailed if Prometheus or Jaeger is unusable."""
    global last_traces
    w = window or config.METRIC_WINDOW
    res = {k: _by_service(q) for k, q in resource_queries(w).items()}

    traces = jaeger_client.fetch_recent_traces(
        config.ENTRY_SERVICE, config.JAEGER_LOOKBACK_SECONDS, config.JAEGER_TRACE_LIMIT, strict=True)
    last_traces = traces
    stats = otel_traces.window_stats(traces, limit=config.JAEGER_TRACE_LIMIT,
                                     window_s=config.JAEGER_LOOKBACK_SECONDS, sample_rate=config.OTEL_SAMPLE_RATE)

    out = {}
    for s in (set(res["cpu_cores"]) | set(res["mem_bytes"]) | set(stats)) - {""}:
        t = stats.get(s, {})
        out[s] = {
            "requests": t.get("requests", 0.0),
            "error_rate": t.get("error_rate", 0.0),
            "own_latency_ms": t.get("own_latency_ms", 0.0),
            "own_p95_ms": t.get("own_p95_ms", 0.0),
            "incl_latency_ms": t.get("incl_latency_ms", 0.0),
            "incl_p50_ms": t.get("incl_p50_ms", 0.0),
            "incl_p95_ms": t.get("incl_p95_ms", 0.0),
            "incl_p99_ms": t.get("incl_p99_ms", 0.0),
            "cpu_cores": res["cpu_cores"].get(s, 0.0),
            "mem_mb": res["mem_bytes"].get(s, 0.0) / 2**20,
            "inflight": 0.0,                           # not exported by the Java services
            "pods_scraped": int(res["pods"].get(s, 0)),
            "trace_samples": t.get("samples", 0),
            "ok": True,
        }
    with_pods = [s for s, d in out.items() if d["pods_scraped"]]
    no_cpu = [s for s in with_pods if s not in res["cpu_cores"]]
    if with_pods and len(no_cpu) > 0.25 * len(with_pods):
        # A running pod always has a CPU series; if many are absent the rate window is shorter than the scrape interval allows
        # (or cAdvisor is failing). Reading "no data" as "idle" could scale a saturated service DOWN, so hold instead.
        raise QueryFailed(f"CPU series missing for {len(no_cpu)} of {len(with_pods)} services with pods "
                          f"(e.g. {', '.join(sorted(no_cpu)[:3])}): METRIC_WINDOW {w} is too short for the scrape interval")
    return out


def node_memory_headroom_mb() -> float:
    """Allocatable node memory minus what the node's cgroup currently uses (working set), in MiB."""
    alloc = _instant_query('sum(kube_node_status_allocatable{resource="memory"})')
    used = _instant_query('sum(container_memory_working_set_bytes{id="/"})')
    if alloc is None or used is None:
        raise QueryFailed("node memory metrics missing")
    return (alloc - used) / 2**20


def get_all_service_metrics(services, window=None):
    try:
        bulk = collect(window)
    except QueryFailed as e:
        print(f"[trainticket_metrics] failed: {e}")
        return {s: {"requests": 0.0, "latency_ms": 0.0, "ok": False} for s in services}
    return {s: {"requests": bulk.get(s, {}).get("requests", 0.0),
                "latency_ms": bulk.get(s, {}).get("incl_latency_ms", 0.0), "ok": True} for s in services}


if __name__ == "__main__":
    print("Prometheus", config.PROMETHEUS_URL, "| Jaeger", config.JAEGER_QUERY_URL, "| sample rate", config.OTEL_SAMPLE_RATE)
    for svc, m in sorted(collect().items()):
        print(f"  {svc:26s} rps={m['requests']:6.2f} own={m['own_latency_ms']:7.1f}ms incl_p95={m['incl_p95_ms']:8.1f}ms "
              f"cpu={m['cpu_cores']:.3f} rss={m['mem_mb']:.0f}MiB pods={m['pods_scraped']} traces={m['trace_samples']}")
