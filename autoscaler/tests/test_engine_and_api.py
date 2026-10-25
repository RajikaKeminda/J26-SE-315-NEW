"""Control engines with fake infrastructure, the dashboard HTTP/WS API, and the live-state mapping."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config
from src.dashboard import server
from src.data_collection.prometheus_client import QueryFailed
from src.inference.engines import DemoEngine, LiveEngine
from src.inference.live_state import LiveStateBuilder
from src.rl_agent.mo_td3 import MOTD3Agent
from src.simulation import service_model as sm
from src.topologies import DEMO6


# ------------------------------------------------------------- fake infra
def prom_row(rps=12.0, own=25.0, cpu=None, incl=None, p95=None, mem=60.0, pods=2):
    return {"requests": rps, "error_rate": 0.0, "own_latency_ms": own, "own_p95_ms": own * 2.2,
            "incl_latency_ms": incl or own, "incl_p50_ms": incl or own, "incl_p95_ms": p95 or (incl or own) * 2.2,
            "incl_p99_ms": (p95 or (incl or own) * 2.2) * 1.3,
            "cpu_cores": cpu if cpu is not None else rps * 0.0158, "mem_mb": mem, "inflight": 0.5,
            "pods_scraped": pods, "ok": True}


class FakeProm:
    def __init__(self, services):
        self.services, self.rows, self.fail, self.empty = services, None, False, False
        self.set(rps=12.0)

    def set(self, rps, own=25.0, incl_gateway=None, cpu=None):
        self.rows = {s: prom_row(rps * 0.5, own, cpu=cpu) for s in self.services}
        self.rows[config.ENTRY_SERVICE] = prom_row(rps, own, cpu=cpu, incl=incl_gateway)

    def collect(self, window=None):
        if self.fail:
            raise QueryFailed("down")
        return {} if self.empty else self.rows


class FakeJaeger:
    def fetch_recent_traces(self, *a, **k):
        return []

    def aggregate_window(self, traces):
        return {}

    def discover_topology(self, *a, **k):
        return None


class FakeK8s:
    def __init__(self, services, replicas=2, cpu=0.25, mem=128.0):
        self.state = {s: {"replicas": replicas, "ready": replicas, "cpu": cpu, "mem": mem} for s in services}
        self.applied, self.fail, self.missing = [], False, set()

    def get_state(self, services, namespace=None):
        return {s: dict(v) for s, v in self.state.items() if s in services and s not in self.missing}

    def apply_action(self, svc, action, namespace=None):
        if self.fail:
            raise RuntimeError("api down")
        self.applied.append((svc, dict(action)))
        self.state[svc]["replicas"] += action["dr"]
        self.state[svc]["ready"] = self.state[svc]["replicas"]
        self.state[svc]["cpu"], self.state[svc]["mem"] = action["cpu"], action["mem"]
        return {"scaled": bool(action["dr"]), "resized": True}


@pytest.fixture
def live():
    engine = LiveEngine(None, None, controller="hpa")           # HPA rules: deterministic
    engine._prom, engine._jaeger = FakeProm(engine.services), FakeJaeger()
    engine._k8s = FakeK8s(engine.services)
    return engine


# ------------------------------------------------------------------ live engine
def test_live_tick_scales_overloaded_services_up(live):
    live._prom.set(rps=60.0, own=400.0, cpu=0.25 * 2)            # CPU pegged at the limit of 2 x 0.25
    p = live.tick()
    assert not p["stale"]
    assert any(a["dr"] > 0 for _, a in live._k8s.applied)
    assert {n["id"] for n in p["nodes"]} == set(live.services)
    assert p["e2e"]["p95_ms"] > 0 and p["controller"] == "hpa"


def test_live_prometheus_outage_holds_instead_of_scaling_down(live):
    live.tick()
    live._k8s.applied.clear()
    live._prom.fail = True
    for _ in range(10):
        p = live.tick()
        assert p["stale"] is True and "cluster read failed" in p["stale_reason"]
    assert live._k8s.applied == [], "must not act on missing data"
    assert p["nodes"][0]["blocked_reason"].startswith("HOLD")


def test_live_no_series_yet_holds(live):
    live._prom.empty = True
    p = live.tick()
    assert p["stale"] and "no series" in p["stale_reason"] and live._k8s.applied == []


def test_live_hold_on_first_tick_has_full_payload(live):
    live._prom.fail = True
    p = live.tick()
    assert p["stale"] and len(p["nodes"]) == len(live.services) and p["edges"]


def test_live_missing_deployment_holds(live):
    live._k8s.missing = {"user-service"}
    p = live.tick()
    assert p["stale"] and "user-service" in p["stale_reason"]
    assert live._k8s.applied == []


def test_live_cluster_read_exception_holds(live):
    def boom(*a, **k):
        raise RuntimeError("no kubeconfig")
    live._k8s.get_state = boom
    assert live.tick()["stale"]


def test_live_shadow_mode_never_acts(live):
    live.shadow = True
    live._prom.set(rps=60.0, own=400.0, cpu=0.5)
    p = live.tick()
    assert live._k8s.applied == [] and p["shadow"] is True
    assert any(n["applied_delta"] > 0 for n in p["nodes"]), "decisions are still computed"


def test_live_scale_failure_does_not_kill_loop(live):
    live._prom.set(rps=60.0, own=400.0, cpu=0.5)
    live._k8s.fail = True
    p = live.tick()
    assert any("k8s error" in (n["blocked_reason"] or "") for n in p["nodes"])
    assert all(n["applied_delta"] == 0 for n in p["nodes"] if n["blocked_reason"])


def test_live_applies_resizes_through_the_guard():
    agent = MOTD3Agent(seed=0)
    for p in agent.actor.parameters():
        p.data += 0.2 * __import__("torch").randn_like(p)
    eng = LiveEngine(agent, "test", controller="mo_td3")
    eng._prom, eng._jaeger, eng._k8s = FakeProm(eng.services), FakeJaeger(), FakeK8s(eng.services)
    eng.tick()
    resized = [a for _, a in eng._k8s.applied if a["cpu"] != 0.25 or a["mem"] != 128.0]
    assert resized, "an untrained-but-perturbed policy proposes a different size and the guard lets some through"
    assert all(config.CPU_MIN <= a["cpu"] <= config.CPU_MAX for _, a in eng._k8s.applied)
    cost = sum(v["replicas"] * (v["cpu"] + config.MEM_COST_PER_GIB * v["mem"] / 1024) for v in eng._k8s.state.values())
    assert cost <= eng.guard.budget + 1e-6


def test_live_topology_discovery_merges_conservatively(live):
    original = list(live.services)
    graph = {"gateway-service": [("catalog-service", 0.7), ("extra-service", 0.3)], "extra-service": []}
    live._jaeger.discover_topology = lambda *a, **k: (
        ["catalog-service", "extra-service", "gateway-service"], "gateway-service", graph)
    live._rediscover()
    assert live.services[:6] == original and live.services[-1] == "extra-service"
    assert live.call_graph["gateway-service"] == [("catalog-service", 0.7), ("extra-service", 0.3)]
    assert live.call_graph["auth-service"] == [("user-service", 1.0)]
    assert live.topology_source == "jaeger"
    live._prom, live._k8s = FakeProm(live.services), FakeK8s(live.services)
    p = live.tick()
    assert "extra-service" in [n["id"] for n in p["nodes"]] and not p["stale"]
    assert live.guard.budget == config.BUDGET_PER_SERVICE * 7


def test_live_topology_discovery_ignores_cycles(live):
    before = dict(live.call_graph)
    live._jaeger.discover_topology = lambda *a, **k: (
        ["user-service", "gateway-service"], "gateway-service", {"user-service": [("gateway-service", 1.0)]})
    live._rediscover()
    assert live.call_graph == before


def test_live_controller_and_preference_can_change_at_runtime(live):
    live.set_controller("vpa")
    assert live.tick()["controller"] == "vpa"
    with pytest.raises(ValueError):
        live.set_controller("nonsense")
    with pytest.raises(ValueError, match="no trained policy"):
        live.set_controller("mo_td3")
    live.set_preference([2, 1, 1])
    assert live.w == pytest.approx([0.5, 0.25, 0.25])


# ------------------------------------------------------- live metrics -> features
def test_live_state_builder_maps_real_signals_to_the_simulator_schema():
    b = LiveStateBuilder(DEMO6.services, DEMO6.entry, DEMO6.call_graph)
    prom = {s: prom_row(rps=10.0, own=30.0, cpu=0.1, mem=64.0) for s in DEMO6.services}
    k8s = {s: {"replicas": 3, "ready": 2, "cpu": 0.5, "mem": 128.0} for s in DEMO6.services}
    metrics, e2e, cost = b.build(prom, k8s)
    m = metrics["gateway-service"]
    assert m["utilization"] == pytest.approx(0.1 / (2 * 0.5))                # cores / (ready x limit)
    assert m["pending_replicas"] == 1 and m["ready_replicas"] == 2 and m["replicas"] == 3
    assert m["mem_utilization"] == pytest.approx(0.5)
    assert m["latency_ms"] == 30.0 and m["latency_p95_ms"] >= 30.0
    assert cost == pytest.approx(6 * 3 * (0.5 + config.MEM_COST_PER_GIB * 128 / 1024))
    sim_keys = set(__import__("src.simulation.simulator", fromlist=["x"]).Simulator(DEMO6).step(10.0)["gateway-service"])
    assert set(m) >= sim_keys, "live metrics must carry every key the simulator provides"
    assert e2e["p95_ms"] == m["latency_incl_p95_ms"]


def test_live_state_utilisation_saturates_at_one_like_a_throttled_pod():
    b = LiveStateBuilder(DEMO6.services, DEMO6.entry, DEMO6.call_graph)
    prom = {s: prom_row(rps=500.0, cpu=9.0) for s in DEMO6.services}
    k8s = {s: {"replicas": 2, "ready": 2, "cpu": 0.25, "mem": 128.0} for s in DEMO6.services}
    assert b.build(prom, k8s)[0]["gateway-service"]["utilization"] == 1.0


def test_live_state_idle_service_uses_model_latency_and_work_estimate_converges():
    b = LiveStateBuilder(DEMO6.services, DEMO6.entry, DEMO6.call_graph)
    k8s = {s: {"replicas": 1, "ready": 1, "cpu": 0.5, "mem": 128.0} for s in DEMO6.services}
    idle = b.build({s: prom_row(rps=0.0, own=0.0, cpu=0.0) for s in DEMO6.services}, k8s)[0]["gateway-service"]
    assert idle["latency_ms"] == pytest.approx(sm.base_latency_ms(0.5, config.WORK_MS))
    for _ in range(40):                                  # 20 rps costing 0.34 cores => ~16 ms/request
        m = b.build({s: prom_row(rps=20.0, cpu=0.34) for s in DEMO6.services}, k8s)[0]["gateway-service"]
    assert 12.0 < m["work_ms"] < 18.0


def test_work_estimate_ignores_saturated_and_transient_windows_and_stays_near_the_prior():
    b = LiveStateBuilder(DEMO6.services, DEMO6.entry, DEMO6.call_graph, priors={s: 15.0 for s in DEMO6.services})
    ready = {s: {"replicas": 2, "ready": 2, "cpu": 0.25, "mem": 128.0} for s in DEMO6.services}
    starting = {s: {"replicas": 3, "ready": 2, "cpu": 0.25, "mem": 128.0} for s in DEMO6.services}
    for _ in range(30):                                  # saturated pods + start-up CPU: 20 rps "costing" 0.5 cores
        b.build({s: prom_row(rps=20.0, cpu=0.5) for s in DEMO6.services}, ready)
        b.build({s: prom_row(rps=20.0, cpu=0.3) for s in DEMO6.services}, starting)
    assert b.work_est["gateway-service"] == pytest.approx(15.0), "must not learn from saturated/starting windows"
    for _ in range(30):                                  # a wildly implausible steady reading is rejected too
        b.build({s: prom_row(rps=20.0, cpu=0.2) for s in DEMO6.services}, {s: dict(v, cpu=2.0) for s, v in ready.items()})
    assert b.work_est["gateway-service"] == pytest.approx(15.0, abs=0.5)
    b.work_est["gateway-service"] = 40.0                  # a corrupted estimate decays back to the declared cost
    for _ in range(60):
        b.build({s: prom_row(rps=0.0, own=0.0, cpu=0.0) for s in DEMO6.services}, ready)
    assert b.work_est["gateway-service"] == pytest.approx(15.0, abs=1.5)


def test_live_state_uses_trace_edge_percentiles_when_there_are_enough_samples():
    b = LiveStateBuilder(DEMO6.services, DEMO6.entry, DEMO6.call_graph)
    prom = {s: prom_row() for s in DEMO6.services}
    k8s = {s: {"replicas": 2, "ready": 2, "cpu": 0.25, "mem": 128.0} for s in DEMO6.services}
    edges = {("gateway-service", "catalog-service"): {"count": 10, "p50_ms": 1, "p95_ms": 777.0},
             ("gateway-service", "auth-service"): {"count": 1, "p50_ms": 1, "p95_ms": 999.0}}
    m = b.build(prom, k8s, edges)[0]["gateway-service"]
    assert m["edge_p95"] == {"catalog-service": 777.0}, "too few samples -> fall back to Prometheus"


# ------------------------------------------------------------------ demo engine
def test_demo_engine_payload_and_controller_switching():
    eng = DemoEngine(None, None, scenario="burst", seed=0, controller="hpa")
    p = eng.tick()
    for key in ("nodes", "edges", "critical_path", "slo_ms", "history", "e2e", "cost", "budget", "controller",
                "preference", "penalties", "breaker_open", "stale", "policy"):
        assert key in p
    n = p["nodes"][0]
    for key in ("cpu", "mem", "ready", "pending", "proposed_cpu", "applied_cpu", "applied_delta"):
        assert key in n
    import json
    json.dumps(p)                                           # must be JSON-serialisable (numpy scrubbed)
    for name in ("vpa", "threshold", "static", "hpa"):
        eng.set_controller(name)
        eng.reset()
        assert eng.tick()["controller"] == name


def test_demo_engine_replays_scenario_when_finished():
    eng = DemoEngine(None, None, scenario="normal", seed=0, controller="static")
    eng.env.t = eng.env.episode_length
    assert eng.tick()["step"] == 0


def test_breaker_opens_then_hands_control_to_hpa():
    """An agent that never scales lets the SLO breach; after N breaches the HPA takes over."""
    agent = MOTD3Agent(seed=0)
    eng = DemoEngine(agent, "frozen", scenario="burst", seed=0, controller="mo_td3")
    called = {"n": 0}
    real_act = agent.act

    def counting_act(*a, **k):
        called["n"] += 1
        idx, cont = real_act(*a, **k)
        return np.full_like(idx, config.NO_OP_INDEX), cont * 0 + cont      # never change replicas
    agent.act = counting_act
    eng.env.traffic[:] = 400.0                                              # sustained overload
    eng.env.guard.budget = 1e6                                              # isolate the breaker from the budget guard
    opened_at = None
    for i in range(config.CIRCUIT_BREAKER_VIOLATIONS + 6):
        p = eng.tick()
        if p["breaker_open"] and opened_at is None:
            opened_at, calls_at_open = i, called["n"]
    assert opened_at is not None and opened_at <= config.CIRCUIT_BREAKER_VIOLATIONS
    assert called["n"] == calls_at_open, "agent must not be consulted while the breaker is open"
    assert p["total_replicas"] > 12, "HPA fallback should have scaled the overloaded services up"


# ------------------------------------------------------------------- API
@pytest.fixture
def client():
    server.configure("demo", interval=0.02, seed=1, controller="hpa")
    with TestClient(server.app) as c:
        yield c


def test_api_state_and_controls(client):
    s = client.get("/api/state").json()
    assert s["mode"] == "demo" and "burst" in s["scenarios"] and s["controller"] == "hpa"
    assert s["objectives"] == list(config.OBJECTIVES)
    assert client.post("/api/pause").json() == {"paused": True}
    assert client.get("/api/state").json()["paused"] is True
    assert client.post("/api/resume").json() == {"paused": False}
    assert client.post("/api/scenario/diurnal").json() == {"scenario": "diurnal"}
    assert client.post("/api/scenario/nope").status_code == 404
    assert client.post("/api/reset").json() == {"reset": True}
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


def test_api_controller_and_preference(client):
    assert client.post("/api/controller/vpa").json() == {"controller": "vpa"}
    assert client.get("/api/state").json()["controller"] == "vpa"
    assert client.post("/api/controller/nope").status_code == 404
    assert client.post("/api/preference", json={"w": [1, 1, 2]}).json() == {"preference": [0.25, 0.25, 0.5]}
    assert client.post("/api/preference", json={"w": [1, 1]}).status_code == 400
    assert client.post("/api/preference", json={"w": [0, 0, 0]}).status_code == 400
    assert client.post("/api/preference", json={"w": "x"}).status_code == 400


def test_api_results_endpoints(client):
    for path in ("/api/results", "/api/pareto", "/api/calibration", "/api/training"):
        body = client.get(path).json()
        assert "available" in body
    cal = client.get("/api/calibration").json()
    assert cal["available"] and "params" in cal["results"] and "fit_report" in cal["results"]


def test_websocket_streams_payloads(client):
    with client.websocket_connect("/ws") as ws:
        first, second = ws.receive_json(), ws.receive_json()
    assert second["step"] > first["step"]
    for key in ("nodes", "edges", "critical_path", "slo_ms", "history", "policy", "stale", "e2e"):
        assert key in first
    assert len(first["nodes"]) == 6


def test_two_clients_share_one_control_loop(client):
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.receive_json(); b.receive_json()
        sa = [a.receive_json()["step"] for _ in range(3)]
        sb = [b.receive_json()["step"] for _ in range(3)]
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


def test_server_falls_back_to_hpa_when_checkpoint_is_missing_or_old(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "POLICY_PATH", str(tmp_path / "shared_policy.pt"))
    server.configure("demo", interval=10, seed=0)
    assert server.STATE["agent"] is None and server.STATE["engine"].controller_name == "hpa"
    import torch
    torch.save({"state_dict": {}}, str(tmp_path / f"{config.DEFAULT_ARM}.pt"))   # old-format checkpoint
    server.configure("demo", interval=10, seed=0)
    assert server.STATE["agent"] is None
    with TestClient(server.app) as c:
        assert "mo_td3" not in c.get("/api/state").json()["controllers"]
        assert c.post("/api/controller/mo_td3").status_code == 400


def test_replay_engine_plays_back_recorded_payloads_in_a_loop(tmp_path, live):
    import json
    from src.inference.engines import ReplayEngine
    frames = [live.tick() for _ in range(3)]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(f) for f in frames) + "\n")
    r = ReplayEngine(str(path))
    got = [r.tick() for _ in range(5)]
    assert [g["step"] for g in got] == [0, 1, 2, 3, 4]
    assert all(g["mode"] == "replay" and g["replay"] and g["replay_length"] == 3 for g in got)
    assert got[3]["nodes"] == got[0]["nodes"]                      # looped
    r.set_controller("hpa")                                        # harmless


# ------------------------------------------------------------------ the page: two tabs, a zoomable graph
def test_page_has_two_tabs_and_the_graph_controls(client):
    html = client.get("/").text
    assert 'role="tablist"' in html and html.count('role="tabpanel"') == 2
    live, analysis = html.split('id="tabAnalysis"')
    assert 'id="tabLive"' in live
    for needle in ('id="graph"', 'id="feed"', 'id="gFit"', 'id="gExpand"', 'id="gZoomIn"', 'id="nodeInfo"'):
        assert needle in live, f"{needle} belongs on the Live tab"
    for needle in ('id="evCard"', 'id="histCard"', 'id="tbody"', 'id="prefs"', 'id="chart"', 'id="ablation"', 'id="pareto"', 'id="curves"'):
        assert needle in analysis, f"{needle} belongs on the Analysis tab"
    assert 'id="graph"' not in analysis and 'id="feed"' not in analysis


def test_page_script_escapes_payload_strings_it_puts_in_html():
    """Service names and guard reasons come from traces / the cluster; they must go through esc() before innerHTML."""
    import os
    src = open(os.path.join(os.path.dirname(server.__file__), "static", "index.html")).read()
    for needle in ("esc(short(n.id))", "esc(n.blocked_reason)", "<h3>${esc(n.id)}</h3>"):
        assert needle in src, needle
    assert "<em>${n.blocked_reason}</em>" not in src


def test_live_no_scale_up_while_the_entry_service_has_no_traffic(live, monkeypatch):
    """A starting JVM burns its whole CPU limit: with no traffic at the entry service the agent must not read that as demand."""
    monkeypatch.setattr(config, "SCALE_UP_MIN_ENTRY_RPS", 0.5)
    live._prom.set(rps=0.0, own=400.0, cpu=0.5)                 # CPU pegged at the limit, but nobody is calling
    p = live.tick()
    assert not p["stale"] and all(a["dr"] <= 0 for _, a in live._k8s.applied)
    assert any("no traffic at the entry service" in (n["blocked_reason"] or "") for n in p["nodes"]), "the HPA rules would have added capacity: the rule must say why it did not"
    live._prom.set(rps=60.0, own=400.0, cpu=0.5)                # now there is traffic: capacity may be added
    live._k8s.applied.clear()
    live.tick()
    assert any(a["dr"] > 0 for _, a in live._k8s.applied)
