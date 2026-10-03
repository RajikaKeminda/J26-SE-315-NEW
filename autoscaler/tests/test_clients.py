"""Tests for the Prometheus / Jaeger / Kubernetes clients (no network, no cluster)."""

import importlib

import pytest
import requests

from src import config
from src.data_collection import jaeger_client, prometheus_client
from src.k8s_action import k8s_client


# ---------------------------------------------------------------- prometheus
class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._body


def _prom_ok(value):
    return _Resp({"status": "success", "data": {"result": [{"value": [0, str(value)]}]}})


def test_prometheus_success(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _prom_ok(12.5))
    assert prometheus_client.get_request_rate("a") == 12.5
    assert prometheus_client.get_avg_latency_ms("a") == pytest.approx(12500.0)  # seconds -> ms


def test_prometheus_empty_result_is_zero_traffic_not_failure(monkeypatch):
    empty = _Resp({"status": "success", "data": {"result": []}})
    monkeypatch.setattr(requests, "get", lambda *a, **k: empty)
    m = prometheus_client.get_all_service_metrics(["a"])
    assert m["a"] == {"requests": 0.0, "latency_ms": 0.0, "ok": True}


def test_prometheus_nan_latency_is_zero(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _prom_ok("NaN"))
    assert prometheus_client.get_avg_latency_ms("a") == 0.0


@pytest.mark.parametrize("failure", [
    lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
    lambda *a, **k: _Resp({}, status=500),
    lambda *a, **k: _Resp({"status": "error"}),
    lambda *a, **k: _Resp({"status": "success", "data": {}}),
])
def test_prometheus_failure_is_flagged_not_zero(monkeypatch, failure):
    """An outage must be distinguishable from 'no traffic' (else we'd scale to min)."""
    monkeypatch.setattr(requests, "get", failure)
    m = prometheus_client.get_all_service_metrics(["a", "b"])
    assert all(v["ok"] is False for v in m.values())
    with pytest.raises(prometheus_client.QueryFailed):
        prometheus_client.get_request_rate("a")


# -------------------------------------------------------------------- jaeger
def _span(span_id, proc, parent=None, op="op", dur=1000):
    refs = [{"refType": "CHILD_OF", "spanID": parent}] if parent else []
    return {"spanID": span_id, "processID": proc, "operationName": op,
            "references": refs, "duration": dur}


def _trace(spans, procs):
    return {"spans": spans, "processes": {p: {"serviceName": s} for p, s in procs.items()}}


def test_call_graph_reconstruction_and_shares():
    procs = {"p1": "gw", "p2": "cat", "p3": "auth", "p4": "inv"}
    traces = []
    for i in range(6):      # gw -> cat (x4), gw -> auth (x2)
        callee = ("p2", "cat") if i < 4 else ("p3", "auth")
        spans = [_span("g", "p1"), _span("gc", "p1", "g"), _span("c", callee[0], "gc")]
        if callee[1] == "cat":
            spans += [_span("cc", "p2", "c"), _span("i", "p4", "cc")]
        traces.append(_trace(spans, procs))
    services, graph = jaeger_client.build_call_graph_from_traces(traces)
    assert services == ["auth", "cat", "gw", "inv"]
    shares = dict(graph["gw"])
    assert shares["cat"] == pytest.approx(4 / 6, abs=1e-3)
    assert shares["auth"] == pytest.approx(2 / 6, abs=1e-3)
    assert graph["cat"] == [("inv", 1.0)]
    assert graph["inv"] == [] and graph["auth"] == []


def test_call_graph_none_without_cross_service_calls():
    t = _trace([_span("a", "p1"), _span("b", "p1", "a")], {"p1": "only"})
    assert jaeger_client.build_call_graph_from_traces([t]) is None
    assert jaeger_client.build_call_graph_from_traces([]) is None


def test_call_graph_rejects_cycles():
    procs = {"p1": "a", "p2": "b"}
    t = _trace([_span("x", "p1"), _span("y", "p2", "x"), _span("z", "p1", "y")], procs)
    assert jaeger_client.build_call_graph_from_traces([t]) is None   # a->b and b->a


def test_cpu_work_latency_extraction_converts_us_to_ms():
    t = _trace([_span("a", "p1", op="cpu_work", dur=15000), _span("b", "p1", op="handle")],
               {"p1": "svc"})
    assert jaeger_client._extract_cpu_work_latencies(t) == {"svc": 15.0}


def test_per_service_latency_falls_back(monkeypatch):
    t = _trace([_span("a", "p1", op="cpu_work", dur=20000)], {"p1": "x"})
    monkeypatch.setattr(jaeger_client, "fetch_recent_traces", lambda *a, **k: [t, t])
    out = jaeger_client.get_real_per_service_latency(["x", "y"], "x", 30, 10, fallback={"y": 7.0})
    assert out == {"x": 20.0, "y": 7.0}


def test_fetch_traces_swallows_network_errors(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    assert jaeger_client.fetch_recent_traces("x", 30, 5) == []
    assert jaeger_client.discover_topology("x", 30, 5) is None


# ---------------------------------------------------------------- kubernetes
class _Dep:
    def __init__(self, n):
        self.spec = type("S", (), {"replicas": n})()
        self.status = type("St", (), {"replicas": n})()


class _FakeApps:
    def __init__(self, replicas):
        self.replicas, self.patches = dict(replicas), []

    def read_namespaced_deployment(self, name, namespace):
        if name not in self.replicas:
            raise k8s_client.client.exceptions.ApiException(status=404, reason="Not Found")
        return _Dep(self.replicas[name])

    def patch_namespaced_deployment_scale(self, name, namespace, body):
        self.patches.append((name, body))
        self.replicas[name] = body["spec"]["replicas"]


@pytest.fixture
def fake_k8s(monkeypatch):
    fake = _FakeApps({"a": 2, "b": config.MAX_REPLICAS, "c": config.MIN_REPLICAS})
    monkeypatch.setattr(k8s_client, "_apps_v1", fake)
    yield fake
    k8s_client.reset_client()


def test_k8s_scale_up_down_and_clamping(fake_k8s):
    assert k8s_client.scale_replicas("a", +1) == 3
    assert k8s_client.scale_replicas("a", -1) == 2
    assert k8s_client.scale_replicas("b", +1) == config.MAX_REPLICAS      # clamped
    assert k8s_client.scale_replicas("c", -1) == config.MIN_REPLICAS      # clamped
    assert [p[0] for p in fake_k8s.patches] == ["a", "a"]                 # clamped = no patch


def test_k8s_counts_omit_unreadable_deployments(fake_k8s):
    counts = k8s_client.get_replica_counts(["a", "missing"])
    assert counts == {"a": 2}          # NOT {"missing": 0}: caller must notice and hold


def test_k8s_uses_incluster_when_in_pod(monkeypatch):
    k8s_client.reset_client()
    called = []
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setattr(k8s_client.k8s_config, "load_incluster_config", lambda: called.append("in"))
    monkeypatch.setattr(k8s_client.k8s_config, "load_kube_config", lambda: called.append("file"))
    k8s_client._load_client()
    k8s_client._load_client()           # cached
    assert called == ["in"]
    k8s_client.reset_client()


# -------------------------------------------------------------------- config
def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_URL", "http://prom:9090")
    monkeypatch.setenv("MAX_CONCURRENCY_PER_REPLICA", "1.5")
    monkeypatch.setenv("AUTODISCOVER_TOPOLOGY", "1")
    try:
        c = importlib.reload(config)
        assert c.PROMETHEUS_URL == "http://prom:9090"
        assert c.MAX_CONCURRENCY_PER_REPLICA == 1.5
        assert c.AUTODISCOVER_TOPOLOGY is True
    finally:
        monkeypatch.undo()
        importlib.reload(config)
