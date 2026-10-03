"""
prometheus_client.py

A real client for Prometheus's HTTP API (/api/v1/query) - the same API
Grafana or any real monitoring tool uses. No mocking: this sends genuine
PromQL queries and parses genuine responses.

Assumes Prometheus is reachable at config.PROMETHEUS_URL (by default this
means you have `kubectl port-forward svc/prometheus 9090:9090 -n autoscaler-poc`
running - see the project README; in-cluster it is the Service DNS name).

FAILURE SEMANTICS (important for safety): a query that FAILS (network error,
HTTP error, malformed body) is NOT the same as a query that legitimately
returns "no traffic". Failures are reported as `ok=False` so the control loop
can refuse to act on missing data instead of reading it as zero load and
scaling the whole application down.
"""

import requests

from src import config


class QueryFailed(Exception):
    """Prometheus could not be reached or returned an unusable response."""


def _instant_query(promql: str) -> float | None:
    """
    Runs one PromQL instant query.
      - returns the first scalar result,
      - returns None if the query succeeded but matched no series,
      - raises QueryFailed if Prometheus was unreachable / errored.
    """
    try:
        resp = requests.get(
            f"{config.PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=5,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise QueryFailed(f"prometheus status={body.get('status')!r}")
        result = body["data"]["result"]
        if not result:
            return None
        return float(result[0]["value"][1])
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as e:
        raise QueryFailed(f"{promql!r}: {e}") from e


def get_request_rate(service: str, window: str = "30s") -> float:
    """Requests/sec for `service`, averaged over `window`. 0.0 if no traffic.
    Raises QueryFailed if Prometheus is unavailable."""
    promql = f'sum(rate(http_requests_total{{service="{service}"}}[{window}]))'
    value = _instant_query(promql)
    return value if value is not None else 0.0


def get_avg_latency_ms(service: str, window: str = "30s") -> float:
    """
    Average per-request latency in ms for `service` over `window`, computed
    the standard PromQL way: rate(sum)/rate(count) of the latency histogram.
    NOTE: this is INCLUSIVE latency (it contains time spent waiting on
    downstream services). Use Jaeger cpu_work spans for a service's OWN time.
    Returns 0.0 if there's no traffic in the window.
    Raises QueryFailed if Prometheus is unavailable.
    """
    promql = (
        f'sum(rate(http_request_duration_seconds_sum{{service="{service}"}}[{window}])) '
        f'/ '
        f'clamp_min(sum(rate(http_request_duration_seconds_count{{service="{service}"}}[{window}])), 1e-9)'
    )
    value = _instant_query(promql)
    if value is None or value != value:  # None or NaN
        return 0.0
    return value * 1000.0


def get_all_service_metrics(services: list[str], window: str = "30s") -> dict[str, dict]:
    """
    Returns {service: {"requests": rps, "latency_ms": inclusive_avg, "ok": bool}}.
    Does NOT include replicas/cpu - that comes from the Kubernetes API
    (see k8s_action/k8s_client.py), merged in by the caller.

    A service whose queries failed gets ok=False (and zeroed numbers that the
    caller MUST NOT act on).
    """
    metrics = {}
    for svc in services:
        try:
            metrics[svc] = {
                "requests": get_request_rate(svc, window),
                "latency_ms": get_avg_latency_ms(svc, window),
                "ok": True,
            }
        except QueryFailed as e:
            print(f"[prometheus_client] query failed: {e}")
            metrics[svc] = {"requests": 0.0, "latency_ms": 0.0, "ok": False}
    return metrics


if __name__ == "__main__":
    print("Querying live Prometheus at", config.PROMETHEUS_URL)
    metrics = get_all_service_metrics(config.SERVICES)
    for svc, m in metrics.items():
        flag = "" if m["ok"] else "  (QUERY FAILED)"
        print(f"  {svc:20s} rps={m['requests']:6.2f}  latency={m['latency_ms']:7.2f}ms{flag}")
