"""LiveEngine behaviour with fake infrastructure, the dashboard HTTP/WS API, Pareto maths."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from src import config
from src.dashboard import server
from src.data_collection.prometheus_client import QueryFailed
from src.evaluation import pareto


# ------------------------------------------------------------- fake infra
class FakeProm:
    def __init__(self, rps=100.0, lat=40.0, ok=True):
        self.rps, self.lat, self.ok = rps, lat, ok

    def get_all_service_metrics(self, services, window="30s"):
        return {s: {"requests": self.rps if self.ok else 0.0,
                    "latency_ms": self.lat if self.ok else 0.0, "ok": self.ok} for s in services}


class FakeJaeger:
    def __init__(self, own=15.0):
        self.own = own

    def get_real_per_service_latency(self, services, entry, lookback, limit, fallback=None):
        return {s: self.own for s in services}

    def discover_topology(self, *a, **k):
        return None


class FakeK8s:
    def __init__(self, services, replicas=2):
        self.replicas = {s: replicas for s in services}
        self.scaled = []
        self.fail_scale = False
        self.missing = set()

    def get_replica_counts(self, services, namespace=None):
        return {s: n for s, n in self.replicas.items() if s in services and s not in self.missing}

    def scale_replicas(self, svc, delta, namespace=None):
        if self.fail_scale:
            raise RuntimeError("api down")
        self.scaled.append((svc, delta))
        self.replicas[svc] += delta


@pytest.fixture
def live():
    engine = server.LiveEngine(policy=None)          # HPA rules: deterministic
    engine._prom, engine._jaeger = FakeProm(), FakeJaeger()
    engine._k8s = FakeK8s(engine.services)
    return engine


def test_live_tick_scales_overloaded_services_up(live):
    live._prom.rps = 2000.0                           # way past 2 replicas of capacity
    p = live.tick()
    assert not p["stale"]
    assert any(d > 0 for _, d in live._k8s.scaled)
    assert {n["id"] for n in p["nodes"]} == set(live.services)


def test_live_prometheus_outage_holds_instead_of_scaling_down(live):
    live.tick()                                       # one healthy tick
    live._k8s.scaled.clear()
    live._prom.ok = False
    for _ in range(10):
        p = live.tick()
        assert p["stale"] is True and "prometheus" in p["stale_reason"]
    assert live._k8s.scaled == [], "must not act on missing data"
    assert p["nodes"][0]["blocked_reason"].startswith("HOLD")


def test_live_hold_on_first_tick_has_full_payload(live):
    live._prom.ok = False
    p = live.tick()
    assert p["stale"] and len(p["nodes"]) == len(live.services) and p["edges"]


def test_live_missing_deployment_holds(live):
    live._k8s.missing = {"user-service"}
    p = live.tick()
    assert p["stale"] and "user-service" in p["stale_reason"]
    assert live._k8s.scaled == []


def test_live_cluster_read_exception_holds(live):
    def boom(*a, **k):
        raise RuntimeError("no kubeconfig")
    live._k8s.get_replica_counts = boom
    assert live.tick()["stale"]


def test_live_shadow_mode_never_scales(live):
    live.shadow = True
    live._prom.rps = 3000.0
    p = live.tick()
    assert live._k8s.scaled == [] and p["shadow"] is True
    assert any(n["applied_delta"] > 0 for n in p["nodes"]), "decisions are still computed"


def test_live_scale_failure_does_not_kill_loop(live):
    live._prom.rps = 3000.0
    live._k8s.fail_scale = True
    p = live.tick()
    assert any("k8s error" in (n["blocked_reason"] or "") for n in p["nodes"])
    assert all(n["applied_delta"] == 0 for n in p["nodes"] if n["blocked_reason"])


def test_live_uses_own_latency_not_inclusive(live):
    live._prom.lat = 500.0                            # inclusive numbers are huge...
    live._jaeger.own = 15.0                           # ...but own time is small
    p = live.tick()
    assert p["critical_path_latency_ms"] < 100


def test_exclusive_latency_estimate():
    graph = {"a": [("b", 0.5), ("c", 0.5)], "b": [], "c": []}
    own = server.exclusive_latency_estimate(["a", "b", "c"], graph, {"a": 100.0, "b": 40.0, "c": 20.0})
    assert own == {"a": 70.0, "b": 40.0, "c": 20.0}
    own = server.exclusive_latency_estimate(["a", "b"], {"a": [("b", 1.0)], "b": []}, {"a": 10.0, "b": 50.0})
    assert own["a"] == 0.0                            # never negative


def test_live_topology_discovery_merges_conservatively(live):
    """New services are added; observed edges override; unseen services are NOT dropped."""
    original = list(live.services)
    graph = {"gateway-service": [("catalog-service", 0.7), ("extra-service", 0.3)],
             "extra-service": []}
    live._jaeger.discover_topology = lambda *a, **k: (
        ["catalog-service", "extra-service", "gateway-service"], "gateway-service", graph)
    live._rediscover()
    assert live.services[:6] == original and live.services[-1] == "extra-service"
    assert live.call_graph["gateway-service"] == [("catalog-service", 0.7), ("extra-service", 0.3)]
    assert live.call_graph["auth-service"] == [("user-service", 1.0)]      # untouched
    assert live.topology_source == "jaeger"
    live._prom, live._k8s = FakeProm(), FakeK8s(live.services)
    p = live.tick()
    assert "extra-service" in [n["id"] for n in p["nodes"]] and not p["stale"]


def test_live_topology_discovery_ignores_cycles(live):
    before = dict(live.call_graph)
    live._jaeger.discover_topology = lambda *a, **k: (
        ["user-service", "gateway-service"], "gateway-service",
        {"user-service": [("gateway-service", 1.0)]})            # user -> gateway closes a loop
    live._rediscover()
    assert live.call_graph == before


def test_breaker_opens_then_hands_control_to_hpa():
    """A policy that never scales lets the SLO breach; after N breaches the HPA takes over."""
    import torch

    class Frozen:                                   # does nothing, counts calls
        calls = 0

        def act(self, X, *a, **k):
            Frozen.calls += 1
            return torch.ones(X.shape[0], dtype=torch.long), None, None   # index 1 == delta 0

    eng = server.DemoEngine(policy=Frozen(), scenario="normal", seed=0)
    eng.traffic[:] = 5000.0                          # sustained overload
    opened_at = None
    for i in range(config.CIRCUIT_BREAKER_VIOLATIONS + 6):
        p = eng.tick()
        if p["breaker_open"] and opened_at is None:
            opened_at, calls_at_open = i, Frozen.calls
    assert opened_at == config.CIRCUIT_BREAKER_VIOLATIONS - 1
    assert Frozen.calls == calls_at_open, "agent must not be consulted while the breaker is open"
    assert p["total_replicas"] > 12, "HPA fallback should have scaled the overloaded services up"


# ------------------------------------------------------------------- API
@pytest.fixture
def client():
    server.configure("demo", interval=0.02, seed=1)
    with TestClient(server.app) as c:
        yield c


def test_api_state_and_controls(client):
    s = client.get("/api/state").json()
    assert s["mode"] == "demo" and "burst" in s["scenarios"] and s["policy"] == "rl_full"
    assert client.post("/api/pause").json() == {"paused": True}
    assert client.get("/api/state").json()["paused"] is True
    assert client.post("/api/resume").json() == {"paused": False}
    assert client.post("/api/scenario/diurnal").json() == {"scenario": "diurnal"}
    assert client.post("/api/scenario/nope").status_code == 404
    assert client.post("/api/reset").json() == {"reset": True}
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


def test_api_results_endpoints(client):
    r = client.get("/api/results").json()
    assert r["available"] and "rl_full" in r["results"] and "hpa" in r["results"]
    assert "available" in client.get("/api/pareto").json()


def test_websocket_streams_payloads(client):
    with client.websocket_connect("/ws") as ws:
        first, second = ws.receive_json(), ws.receive_json()
    assert second["step"] > first["step"]
    for key in ("nodes", "edges", "critical_path", "slo_ms", "history", "policy", "stale"):
        assert key in first
    assert len(first["nodes"]) == 6 and first["policy"] == "rl_full"


def test_two_clients_share_one_control_loop(client):
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.receive_json(); b.receive_json()
        sa = [a.receive_json()["step"] for _ in range(3)]
        sb = [b.receive_json()["step"] for _ in range(3)]
    # one ticker: steps are consecutive for each client, not double-speed
    assert sa == sorted(sa) and all(y - x <= 2 for x, y in zip(sa, sa[1:]))
    assert sb == sorted(sb)


def test_scenario_rejected_in_live_mode():
    server.configure("demo", interval=10)
    server.STATE["mode"] = "live"
    try:
        with TestClient(server.app) as c:
            assert c.post("/api/scenario/burst").status_code == 400
    finally:
        server.STATE["mode"] = "demo"


# ------------------------------------------------------------------ pareto
def test_pareto_dominance_and_front():
    a = dict(p95_latency=50, mean_cost=2.0, slo_violation_rate=0.0)
    b = dict(p95_latency=80, mean_cost=2.5, slo_violation_rate=0.1)      # dominated by a
    c = dict(p95_latency=120, mean_cost=1.0, slo_violation_rate=0.05)    # cheaper: non-dominated
    assert pareto.dominates(a, b) and not pareto.dominates(b, a)
    assert not pareto.dominates(a, c) and not pareto.dominates(c, a)
    assert pareto.pareto_front([a, b, c]) == [True, False, True]
    assert pareto.pareto_front([a, dict(a)]) == [True, True]             # ties both stay
