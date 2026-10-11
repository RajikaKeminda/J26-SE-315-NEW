"""Tests for the Train-Ticket (OpenTelemetry-span) metrics path. The traces are synthetic but copy the
shape of the real ones: server span -> internal -> client span -> callee server span, plus DB client spans."""

import re

import pytest

from src import config
from src.data_collection import otel_traces, trainticket_metrics
from src.topologies import TRAINTICKET_LIVE, validate


def _span(sid, svc, kind, start_ms, dur_ms, parent=None, op="op", tags=None):
    return {"spanID": sid, "processID": svc, "operationName": op, "startTime": int(start_ms * 1000),
            "duration": int(dur_ms * 1000),
            "references": [{"refType": "CHILD_OF", "spanID": parent}] if parent else [],
            "tags": [{"key": "span.kind", "value": kind}] + (tags or [])}


def _trace(spans):
    return {"traceID": "t", "spans": spans, "processes": {s: {"serviceName": s} for s in {x["processID"] for x in spans}}}


def booking():
    """gateway(100ms) -> preserve(80ms) -> [seat 30ms, basic 20ms in PARALLEL @10..40]; preserve also has a DB call."""
    return _trace([
        _span("g", "gw", "server", 0, 100),
        _span("gc", "gw", "client", 2, 96, parent="g"),
        _span("p", "pre", "server", 3, 94, parent="gc"),
        _span("pc1", "pre", "client", 10, 30, parent="p"),
        _span("s", "seat", "server", 11, 28, parent="pc1"),
        _span("pc2", "pre", "client", 10, 20, parent="p"),
        _span("b", "basic", "server", 11, 18, parent="pc2"),
        _span("db", "pre", "client", 50, 10, parent="p", op="SELECT ts.x"),
    ])


def test_own_latency_subtracts_waiting_on_other_services_once():
    rows = {r["service"]: r for r in otel_traces.analyse_trace(booking())}
    assert rows["gw"]["incl_ms"] == 100 and rows["gw"]["parent_service"] is None
    assert rows["pre"]["parent_service"] == "gw"
    # preserve waits on the UNION of its two parallel calls (10..40) = 30 ms; its DB call stays its own time
    assert rows["pre"]["incl_ms"] == 94 and rows["pre"]["own_ms"] == pytest.approx(94 - 30)
    # gateway waits for the whole preserve call (its client span 2..98 = 96 ms)
    assert rows["gw"]["own_ms"] == pytest.approx(100 - 96)
    assert rows["seat"]["own_ms"] == rows["seat"]["incl_ms"] == 28          # leaf


def test_window_stats_rate_and_percentiles():
    st = otel_traces.window_stats([booking()] * 20, window_s=10.0, sample_rate=1.0)
    assert st["gw"]["requests"] == pytest.approx(2.0)                         # 20 requests / 10 s
    assert st["gw"]["incl_p95_ms"] == 100 and st["seat"]["incl_latency_ms"] == 28
    st = otel_traces.window_stats([booking()] * 20, window_s=10.0, sample_rate=0.25)
    assert st["gw"]["requests"] == pytest.approx(8.0)                         # sampling-corrected


def test_truncated_window_uses_the_time_the_traces_cover():
    traces = []
    for i in range(10):                                  # 10 traces over 1 s although 30 s were requested
        t = booking()
        for s in t["spans"]:
            s["startTime"] += i * 100_000
        traces.append(t)
    st = otel_traces.window_stats(traces, limit=10, window_s=30.0)
    assert st["gw"]["requests"] > 5.0                    # NOT 10/30 s


def test_errors_counted_from_status_code():
    t = booking()
    t["spans"][0]["tags"].append({"key": "http.status_code", "value": 503})
    assert otel_traces.window_stats([t, booking()], window_s=1.0)["gw"]["error_rate"] == 0.5


def test_pod_name_to_deployment_regex_is_anchored_like_prometheus():
    rx = re.compile(trainticket_metrics.POD_TO_SERVICE)
    assert rx.fullmatch("ts-order-service-848589b9d8-7rwfs").group(1) == "ts-order-service"
    assert rx.fullmatch("ts-order-other-service-dffdcd5f-56xbb").group(1) == "ts-order-other-service"


def test_collect_merges_cadvisor_and_trace_stats(monkeypatch):
    monkeypatch.setattr(trainticket_metrics, "_by_service", lambda q: (
        {"gw": 0.4} if "cpu_usage" in q and "count by" not in q else
        {"gw": 700 * 2**20} if "count by" not in q else {"gw": 2}))
    monkeypatch.setattr(trainticket_metrics.jaeger_client, "fetch_recent_traces", lambda *a, **k: [booking()] * 5)
    monkeypatch.setattr(config, "ENTRY_SERVICE", "gw")
    out = trainticket_metrics.collect()
    assert out["gw"]["cpu_cores"] == 0.4 and out["gw"]["pods_scraped"] == 2 and out["gw"]["mem_mb"] == 700
    assert out["gw"]["incl_p95_ms"] == 100 and out["seat"]["cpu_cores"] == 0.0     # trace-only service


def test_topology_is_a_valid_dag_and_services_are_a_whitelist():
    validate(TRAINTICKET_LIVE)
    assert TRAINTICKET_LIVE.entry == "ts-gateway-service" and len(TRAINTICKET_LIVE.services) == 11


def test_default_bounds_unchanged_for_the_demo_app():
    assert (config.CPU_MIN, config.CPU_MAX, config.MEM_MIN_MB, config.MEM_MAX_MB, config.MAX_REPLICAS) == (0.1, 1.0, 96, 512, 8)
