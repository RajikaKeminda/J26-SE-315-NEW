"""
prometheus_client.py

A real client for Prometheus's HTTP API (/api/v1/query) - the same API Grafana or any
real monitoring tool uses. No mocking: it sends genuine PromQL queries and parses
genuine responses.

Assumes Prometheus is reachable at config.PROMETHEUS_URL (by default
`kubectl port-forward svc/prometheus 9090:9090 -n autoscaler-poc`; in-cluster it is
the Service DNS name).

What is measured per service (one query per metric for ALL services at once, so a
control tick costs ~10 HTTP requests however many services there are):

    requests        rate(http_requests_total)
    own latency     service_own_duration_seconds  mean + p95 (queueing + CPU, excl. downstream)
    inclusive       http_request_duration_seconds mean + p95 (incl. downstream - at the entry
                    service this IS the end-to-end latency a client sees)
    cpu cores       rate(process_cpu_seconds_total)   - REAL CPU consumption of the pods
    memory          process_resident_memory_bytes     - REAL resident set
    inflight        http_requests_inflight
    error rate      rate(http_request_errors_total) / requests

FAILURE SEMANTICS (important for safety): a query that FAILS (network error, HTTP
error, malformed body) is NOT the same as one that legitimately returns "no traffic".
Failures raise QueryFailed and the service is reported ok=False so the control loop can
refuse to act on missing data instead of reading it as zero load and scaling the whole
application down.
"""

import requests

from src import config


class QueryFailed(Exception):
    """Prometheus could not be reached or returned an unusable response."""


def _get(promql: str) -> list:
    try:
        resp = requests.get(f"{config.PROMETHEUS_URL}/api/v1/query",
                            params={"query": promql}, timeout=5)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise QueryFailed(f"prometheus status={body.get('status')!r}")
        return body["data"]["result"]
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as e:
        raise QueryFailed(f"{promql!r}: {e}") from e


def _instant_query(promql: str) -> float | None:
    """First scalar result; None if the query matched no series; raises QueryFailed on error."""
    result = _get(promql)
    if not result:
        return None
    return float(result[0]["value"][1])


def _by_service(promql: str) -> dict[str, float]:
    """Runs a query that aggregates `by (service)` -> {service: value}; NaN/absent dropped."""
    out = {}
    for item in _get(promql):
        v = float(item["value"][1])
        if v == v:                                   # drop NaN (e.g. quantile with no traffic)
            out[item["metric"].get("service", "")] = v
    return out


# --- single-service helpers (kept: used by tools, tests and the CLI below) -------------
def get_request_rate(service: str, window: str | None = None) -> float:
    window = window or config.METRIC_WINDOW
    value = _instant_query(f'sum(rate(http_requests_total{{service="{service}"}}[{window}]))')
    return value if value is not None else 0.0


def get_avg_latency_ms(service: str, window: str | None = None) -> float:
    """Average INCLUSIVE latency in ms (rate(sum)/rate(count) of the latency histogram)."""
    window = window or config.METRIC_WINDOW
    promql = (
        f'sum(rate(http_request_duration_seconds_sum{{service="{service}"}}[{window}])) '
        f'/ clamp_min(sum(rate(http_request_duration_seconds_count{{service="{service}"}}[{window}])), 1e-9)')
    value = _instant_query(promql)
    return 0.0 if (value is None or value != value) else value * 1000.0


# --- bulk collection --------------------------------------------------------------------
def _quantile(q, metric, window):
    return (f'histogram_quantile({q}, sum by (service, le) (rate({metric}_bucket[{window}])))')


def _mean(metric, window):
    return (f'sum by (service) (rate({metric}_sum[{window}])) / '
            f'clamp_min(sum by (service) (rate({metric}_count[{window}])), 1e-9)')


def collect(window: str | None = None) -> dict[str, dict]:
    """One window of per-service measurements for EVERY service Prometheus knows about.

    Raises QueryFailed if Prometheus is unreachable. Services with no series simply do
    not appear (the caller treats an expected-but-missing service as 'no traffic yet')."""
    w = window or config.METRIC_WINDOW
    q = {
        "requests":        f"sum by (service) (rate(http_requests_total[{w}]))",
        "errors":          f"sum by (service) (rate(http_request_errors_total[{w}]))",
        "own_mean":        _mean("service_own_duration_seconds", w),
        "own_p95":         _quantile(0.95, "service_own_duration_seconds", w),
        "incl_mean":       _mean("http_request_duration_seconds", w),
        "incl_p50":        _quantile(0.50, "http_request_duration_seconds", w),
        "incl_p95":        _quantile(0.95, "http_request_duration_seconds", w),
        "incl_p99":        _quantile(0.99, "http_request_duration_seconds", w),
        "cpu_cores":       f"sum by (service) (rate(process_cpu_seconds_total[{w}]))",
        "mem_bytes":       "avg by (service) (process_resident_memory_bytes)",
        "inflight":        "avg by (service) (http_requests_inflight)",
        "pods":            "count by (service) (process_resident_memory_bytes)",
    }
    data = {k: _by_service(v) for k, v in q.items()}
    services = set().union(*[set(d) for d in data.values()]) - {""}
    out = {}
    for s in services:
        out[s] = {
            "requests": data["requests"].get(s, 0.0),
            "error_rate": min(1.0, data["errors"].get(s, 0.0) / max(data["requests"].get(s, 0.0), 1e-9))
            if data["requests"].get(s, 0.0) > 0 else 0.0,
            "own_latency_ms": data["own_mean"].get(s, 0.0) * 1000.0,
            "own_p95_ms": data["own_p95"].get(s, 0.0) * 1000.0,
            "incl_latency_ms": data["incl_mean"].get(s, 0.0) * 1000.0,
            "incl_p50_ms": data["incl_p50"].get(s, 0.0) * 1000.0,
            "incl_p95_ms": data["incl_p95"].get(s, 0.0) * 1000.0,
            "incl_p99_ms": data["incl_p99"].get(s, 0.0) * 1000.0,
            "cpu_cores": data["cpu_cores"].get(s, 0.0),
            "mem_mb": data["mem_bytes"].get(s, 0.0) / 2**20,
            "inflight": data["inflight"].get(s, 0.0),
            "pods_scraped": int(data["pods"].get(s, 0)),
            "ok": True,
        }
    return out


def get_all_service_metrics(services: list[str], window: str | None = None) -> dict[str, dict]:
    """Backwards-compatible bulk call returning {service: {requests, latency_ms, ok}}.

    A failed Prometheus gives ok=False for every service (zeroed numbers the caller MUST
    NOT act on)."""
    try:
        bulk = collect(window)
    except QueryFailed as e:
        print(f"[prometheus_client] query failed: {e}")
        return {s: {"requests": 0.0, "latency_ms": 0.0, "ok": False} for s in services}
    return {s: {"requests": bulk.get(s, {}).get("requests", 0.0),
                "latency_ms": bulk.get(s, {}).get("incl_latency_ms", 0.0), "ok": True}
            for s in services}


if __name__ == "__main__":
    print("Querying live Prometheus at", config.PROMETHEUS_URL)
    for svc, m in sorted(collect().items()):
        print(f"  {svc:20s} rps={m['requests']:6.2f}  own={m['own_latency_ms']:6.1f}ms  "
              f"incl_p95={m['incl_p95_ms']:7.1f}ms  cpu={m['cpu_cores']:.3f} cores  "
              f"rss={m['mem_mb']:.0f}MiB  pods={m['pods_scraped']}")
