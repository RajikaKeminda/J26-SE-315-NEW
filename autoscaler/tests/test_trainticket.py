"""Tests for the Train-Ticket (OpenTelemetry-span) metrics path. The traces are synthetic but copy the
shape of the real ones: server span -> internal -> client span -> callee server span, plus DB client spans."""

import re

import pytest

from src import config
from src.data_collection import otel_traces, trainticket_metrics
from src.topologies import TRAINTICKET_LIVE, validate
from src.topology_intel.live_topology import LiveTopology


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


def test_seed_is_only_the_entry_point():
    validate(TRAINTICKET_LIVE)
    assert TRAINTICKET_LIVE.services == [TRAINTICKET_LIVE.entry] == ["ts-gateway-service"]


def test_live_topology_is_learned_from_traces_only():
    lt = LiveTopology("gw", half_life_s=60, ttl_s=120, min_calls=1)
    assert lt.services() == ["gw"]                                   # nothing declared
    assert lt.update([booking()] * 3, now=1000.0) is True
    assert lt.services() == ["gw", "basic", "pre", "seat"]
    g = lt.call_graph()
    assert g["gw"] == [("pre", 1.0)]
    assert dict(g["pre"]) == pytest.approx({"seat": 0.5, "basic": 0.5})
    assert lt.update([booking()] * 3, now=1010.0) is False           # same structure


def test_live_topology_drops_edges_not_seen_for_ttl_but_freezes_when_idle():
    lt = LiveTopology("gw", half_life_s=60, ttl_s=120, min_calls=1)
    lt.update([booking()] * 3, now=1000.0)
    other = _trace([_span("g", "gw", "server", 0, 50), _span("c", "gw", "client", 1, 40, parent="g"),
                    _span("x", "extra", "server", 2, 38, parent="c")])
    lt.update([], now=5000.0)                                        # idle for a long time: graph kept
    assert "pre" in lt.services()
    assert lt.update([other] * 3, now=5001.0) is True                # busy again: old edges are past the TTL
    assert lt.services() == ["gw", "extra"]


def test_live_topology_breaks_cycles_and_ignores_noise():
    lt = LiveTopology("a", min_calls=2)
    mk = lambda p, c: _trace([_span("p", p, "server", 0, 20), _span("pc", p, "client", 1, 10, parent="p"),
                              _span("c", c, "server", 2, 8, parent="pc")])
    lt.update([mk("a", "b")] * 5 + [mk("b", "c")] * 5 + [mk("c", "b")] * 2 + [mk("a", "z")], now=1.0)
    g = lt.call_graph()
    assert "z" not in lt.services()                                  # one call < min_calls
    assert sum(1 for t in g.values() for _ in t) == 2                # the weaker c->b edge was ignored
    from src.topologies import Topology, validate as v
    v(Topology("t", lt.services(), "a", g))


def test_default_bounds_unchanged_for_the_demo_app():
    assert (config.CPU_MIN, config.CPU_MAX, config.MEM_MIN_MB, config.MEM_MAX_MB, config.MAX_REPLICAS) == (0.1, 1.0, 96, 512, 8)


def test_trace_window_follows_the_jaeger_servers_clock(monkeypatch):
    import time as _t
    from email.utils import formatdate
    from src.data_collection import jaeger_client as jc

    class R:   # a server whose clock is 28 s ahead of ours
        headers = {"Date": formatdate(_t.time() + 28, usegmt=True)}
    monkeypatch.setattr(jc.requests, "get", lambda *a, **k: R())
    jc._SKEW.update(value=0.0, at=0.0)
    assert 26.0 < jc.clock_skew_s() < 30.0
    seen = {}

    class Q:
        def raise_for_status(self): pass
        def json(self): return {"data": []}
    def fake_get(url, params=None, timeout=None):
        seen.update(params or {}); return Q()
    monkeypatch.setattr(jc.requests, "get", fake_get)
    jc.fetch_recent_traces("svc", 60, 10)
    assert seen["end"] / 1e6 - _t.time() > 20          # the window ends in the server's time, not ours
    jc._SKEW.update(value=0.0, at=0.0)


def test_cluster_safety_scale_up_needs_node_memory_and_resize_can_be_disabled(monkeypatch):
    from src.inference.engines import LiveEngine
    monkeypatch.setattr(config, "NODE_MEMORY_RESERVE_MB", 768.0)
    monkeypatch.setattr(config, "ALLOW_RESIZE", False)
    monkeypatch.setattr(LiveEngine, "_controlled", lambda self, s: True)

    def run(free_mb):
        e = LiveEngine.__new__(LiveEngine)
        class P:
            @staticmethod
            def node_memory_headroom_mb(): return free_mb
        e._prom = P
        approved = {"a": {"dr": 1, "cpu": 0.5, "mem": 600}, "b": {"dr": 1, "cpu": 1.0, "mem": 800}}
        metrics = {"a": {"cpu": 1.0, "mem": 800, "utilization": 0.9}, "b": {"cpu": 1.0, "mem": 800, "utilization": 0.1}}
        prom = {"a": {"mem_mb": 200.0}, "b": {"mem_mb": 400.0}}      # a needs max(200, 256) MiB, b needs 400
        reasons = {}
        e._apply_cluster_safety(approved, reasons, metrics, prom)
        return approved, reasons

    approved, reasons = run(768 + 332)                  # 332 MiB of room: the busier service gets its pod, b does not
    assert approved["a"] == {"dr": 1, "cpu": 1.0, "mem": 800}        # and its resize was undone
    assert approved["b"]["dr"] == 0 and "node memory headroom" in reasons["b"]
    approved, _ = run(768 + 1000)                       # plenty of room: both
    assert approved["a"]["dr"] == approved["b"]["dr"] == 1
    approved, reasons = run(100)                        # below the reserve: nothing is added
    assert approved["a"]["dr"] == approved["b"]["dr"] == 0


def test_critical_time_blames_the_last_running_call_and_sums_to_the_request():
    acc, total = otel_traces.critical_time_by_service(booking())
    assert total == pytest.approx(100.0) and sum(acc.values()) == pytest.approx(100.0)
    # preserve ran seat (10..40) and basic (10..30) in parallel: only the later-finishing seat is on the critical path
    assert acc.get("seat", 0) == pytest.approx(28.0) and acc.get("basic", 0) < 10.0
    assert acc["pre"] > acc["gw"]                         # its own work and the database call


def test_agents_critical_path_beats_random_chains_on_real_traces():
    import json, os
    from src.evaluation import critical_path_eval as E
    path = os.path.join(os.path.dirname(__file__), "fixtures", "trainticket_traces.json")
    r = E.evaluate(json.load(open(path)))
    rows = {x["method"]: x for x in r["rows"]}
    agent, rnd = rows["agent: DFS on own latency"], rows["baseline: random chain (expected)"]
    assert agent["coverage"] > rnd["coverage"] + 0.2 and agent["top1"] > rnd["top1"] + 0.2
    assert "ts-order-service" in agent["path"]            # order carries about three quarters of the downstream time
    ev = E.evidence(json.load(open(path)), agent["path"], "ts-gateway-service")
    assert ev["coverage"] == pytest.approx(agent["coverage"], abs=1e-3) and ev["slowest_trace"]


def test_missing_cpu_series_make_the_agent_hold_instead_of_reading_zero_load(monkeypatch):
    from src.data_collection.prometheus_client import QueryFailed
    def fake(q):
        if "count by" in q:
            return {s: 1 for s in ("gw", "a", "b", "c")}
        if "cpu_usage" in q:
            return {"gw": 0.4}                                             # three of four services have no CPU series
        return {s: 300 * 2**20 for s in ("gw", "a", "b", "c")}
    monkeypatch.setattr(trainticket_metrics, "_by_service", fake)
    monkeypatch.setattr(trainticket_metrics.jaeger_client, "fetch_recent_traces", lambda *a, **k: [booking()] * 3)
    monkeypatch.setattr(config, "ENTRY_SERVICE", "gw")
    with pytest.raises(QueryFailed, match="too short for the scrape interval"):
        trainticket_metrics.collect()
