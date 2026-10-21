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


def _vec(by_service):
    return _Resp({"status": "success", "data": {"result": [
        {"metric": {"service": s}, "value": [0, str(v)]} for s, v in by_service.items()]}})


def test_prometheus_bulk_collect_parses_every_signal(monkeypatch):
    """One query per signal for ALL services; absent series -> 0, NaN dropped."""
    seen = []

    def fake_get(url, params=None, timeout=None):
        q = params["query"]
        seen.append(q)
        if "http_requests_total" in q:
            return _vec({"gw": 20.0, "inv": 10.0})
        if "quantile(0.95, sum by (service, le) (rate(http_request_duration_seconds_bucket" in q:
            return _vec({"gw": 0.25, "inv": "NaN"})
        if "service_own_duration_seconds_sum" in q:
            return _vec({"gw": 0.03})
        if "process_cpu_seconds_total" in q:
            return _vec({"gw": 0.5, "inv": 0.2})
        if "process_resident_memory_bytes" in q and "count" not in q:
            return _vec({"gw": 60 * 2**20})
        return _vec({})
    monkeypatch.setattr(requests, "get", fake_get)
    out = prometheus_client.collect("20s")
    assert set(out) == {"gw", "inv"}
    assert out["gw"]["requests"] == 20.0 and out["gw"]["incl_p95_ms"] == pytest.approx(250.0)
    assert out["gw"]["own_latency_ms"] == pytest.approx(30.0) and out["gw"]["cpu_cores"] == 0.5
    assert out["gw"]["mem_mb"] == pytest.approx(60.0)
    assert out["inv"]["incl_p95_ms"] == 0.0, "NaN (no traffic) must not poison the numbers"
    assert len(seen) == 12 and all("[20s]" in q for q in seen if "rate(" in q)


def test_prometheus_collect_raises_on_outage(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    with pytest.raises(prometheus_client.QueryFailed):
        prometheus_client.collect()


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


def test_window_aggregation_gives_edge_frequency_and_latency_percentiles():
    procs = {"p1": "gw", "p2": "cat"}
    traces = []
    for i in range(10):                      # gw -> cat, callee request span lasts (i+1) ms
        traces.append(_trace([_span("g", "p1"), _span("gc", "p1", "g"),
                              _span("c", "p2", "gc", dur=(i + 1) * 1000)], procs))
    edges = jaeger_client.aggregate_window(traces)
    assert set(edges) == {("gw", "cat")}
    e = edges[("gw", "cat")]
    assert e["count"] == 10 and e["p50_ms"] == pytest.approx(6.0, abs=1.0) and e["p95_ms"] >= 9.0
    assert jaeger_client.aggregate_window([]) == {}


def test_fetch_traces_swallows_network_errors(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    assert jaeger_client.fetch_recent_traces("x", 30, 5) == []
    assert jaeger_client.discover_topology("x", 30, 5) is None


# ---------------------------------------------------------------- kubernetes
class _Container:
    def __init__(self, name, cpu, mem):
        res = {"cpu": cpu, "memory": mem}
        self.name = name
        self.resources = type("R", (), {"limits": dict(res), "requests": dict(res)})()


class _Dep:
    def __init__(self, name, n, ready=None, cpu="250m", mem="128Mi"):
        self.metadata = type("M", (), {"name": name})()
        self.spec = type("S", (), {"replicas": n, "template": type("T", (), {"spec": type("P", (), {
            "containers": [_Container(name, cpu, mem)]})()})()})()
        self.status = type("St", (), {"replicas": n, "ready_replicas": n if ready is None else ready})()


class _FakeApps:
    def __init__(self, replicas):
        self.deps = {k: _Dep(k, v) for k, v in replicas.items()}
        self.patches, self.template_patches = [], []

    list_calls = 0
    allow_list = True

    def list_namespaced_deployment(self, namespace):
        type(self).list_calls += 1
        if not self.allow_list:
            raise k8s_client.client.exceptions.ApiException(status=403, reason="Forbidden")
        return type("L", (), {"items": list(self.deps.values())})()

    def read_namespaced_deployment(self, name, namespace):
        self.reads = getattr(self, "reads", 0) + 1
        if name not in self.deps:
            raise k8s_client.client.exceptions.ApiException(status=404, reason="Not Found")
        return self.deps[name]

    def patch_namespaced_deployment_scale(self, name, namespace, body):
        self.patches.append((name, body))
        self.deps[name].spec.replicas = body["spec"]["replicas"]

    def patch_namespaced_deployment(self, name, namespace, body):
        self.template_patches.append((name, body))
        c = body["spec"]["template"]["spec"]["containers"][0]
        self.deps[name].spec.template.spec.containers[0].resources.limits = dict(c["resources"]["limits"])


@pytest.fixture
def fake_k8s(monkeypatch):
    fake = _FakeApps({"a": 2, "b": config.MAX_REPLICAS, "c": config.MIN_REPLICAS})
    monkeypatch.setattr(k8s_client, "_apps_v1", fake)
    yield fake
    k8s_client.reset_client()


def test_k8s_scale_up_down_and_clamping(fake_k8s):
    assert k8s_client.scale_replicas("a", +1) == 3
    assert k8s_client.scale_replicas("a", -1) == 2
    assert k8s_client.scale_replicas("a", +3) == 5                      # the +-3 action space
    assert k8s_client.scale_replicas("b", +1) == config.MAX_REPLICAS      # clamped
    assert k8s_client.scale_replicas("c", -1) == config.MIN_REPLICAS      # clamped
    assert [p[0] for p in fake_k8s.patches] == ["a", "a", "a"]            # clamped = no patch


def test_k8s_state_reads_replicas_readiness_and_limits(fake_k8s):
    fake_k8s.deps["a"] = _Dep("a", 3, ready=1, cpu="500m", mem="256Mi")
    st = k8s_client.get_state(["a", "missing"])
    assert st == {"a": {"replicas": 3, "ready": 1, "cpu": 0.5, "mem": 256.0}}   # missing is omitted


def test_k8s_counts_omit_unreadable_deployments(fake_k8s):
    counts = k8s_client.get_replica_counts(["a", "missing"])
    assert counts == {"a": 2}          # NOT {"missing": 0}: caller must notice and hold


def test_k8s_set_resources_patches_template_with_requests_equal_limits(fake_k8s):
    assert k8s_client.set_resources("a", 0.75, 192) is True
    name, body = fake_k8s.template_patches[-1]
    c = body["spec"]["template"]["spec"]["containers"][0]
    assert name == "a" and c["name"] == "a"
    assert c["resources"]["requests"] == c["resources"]["limits"] == {"cpu": "750m", "memory": "192Mi"}
    assert k8s_client.set_resources("a", 0.75, 192) is False             # no-op: no second rollout
    assert len(fake_k8s.template_patches) == 1


def test_k8s_apply_action_combines_scale_and_resize(fake_k8s):
    done = k8s_client.apply_action("a", {"dr": +2, "cpu": 1.0, "mem": 128.0})
    assert done == {"scaled": True, "resized": True}
    assert fake_k8s.deps["a"].spec.replicas == 4
    assert k8s_client.apply_action("a", {"dr": 0, "cpu": 1.0, "mem": 128.0}) == {"scaled": False, "resized": False}


@pytest.mark.parametrize("q,expected", [("250m", 0.25), ("1", 1.0), ("2", 2.0), (0.5, 0.5), (None, 0.0)])
def test_parse_cpu(q, expected):
    assert k8s_client.parse_cpu(q) == expected


@pytest.mark.parametrize("q,expected", [("128Mi", 128), ("1Gi", 1024), ("134217728", 128), ("512Ki", 0.5), (None, 0)])
def test_parse_mem(q, expected):
    assert k8s_client.parse_mem_mb(q) == pytest.approx(expected)


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
    monkeypatch.setenv("STEP_SECONDS", "5")
    monkeypatch.setenv("AUTODISCOVER_TOPOLOGY", "1")
    monkeypatch.setenv("SLO_LATENCY_MS", "400")
    try:
        c = importlib.reload(config)
        assert c.PROMETHEUS_URL == "http://prom:9090"
        assert c.STEP_SECONDS == 5.0
        assert c.AUTODISCOVER_TOPOLOGY is True
        assert c.slo_for_depth(3) == 400.0
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_k8s_state_uses_one_list_call_for_many_services_and_falls_back_when_listing_is_forbidden(fake_k8s):
    fake_k8s.deps.update({f"s{i}": _Dep(f"s{i}", 1) for i in range(50)})
    type(fake_k8s).list_calls, fake_k8s.reads = 0, 0
    st = k8s_client.get_state([f"s{i}" for i in range(50)] + ["missing"])
    assert len(st) == 50 and type(fake_k8s).list_calls == 1 and fake_k8s.reads == 0      # one round trip, not 51
    fake_k8s.allow_list = False                                                            # RBAC without the list verb
    st = k8s_client.get_state(["a", "s1", "missing"])
    assert set(st) == {"a", "s1"} and fake_k8s.reads == 3                                  # per-service reads still work


def test_k8s_pod_ages_reports_the_youngest_pod_per_deployment(monkeypatch):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    mk = lambda name, age, deleting=False: type("P", (), {"metadata": type("M", (), {
        "name": name, "creation_timestamp": now - datetime.timedelta(seconds=age),
        "deletion_timestamp": now if deleting else None})()})()
    pods = [mk("svc-a-5d8f7c9b4-x2k9p", 600), mk("svc-a-5d8f7c9b4-q7w1z", 40), mk("svc-b-77c6d-zzzzz", 900), mk("svc-b-77c6d-dying", 5, True)]
    monkeypatch.setattr(k8s_client, "_load_client", lambda: None)
    monkeypatch.setattr(k8s_client.client, "CoreV1Api",
                        lambda: type("C", (), {"list_namespaced_pod": lambda self, namespace: type("L", (), {"items": pods})()})())
    ages = k8s_client.pod_ages("ns")
    assert 39 < ages["svc-a"] < 45 and 890 < ages["svc-b"] < 910                          # deleting pods do not count
