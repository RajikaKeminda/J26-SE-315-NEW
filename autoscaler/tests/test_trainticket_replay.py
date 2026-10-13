"""End-to-end replay: REAL Train-Ticket traces (tests/fixtures, captured from the live cluster) go through the live engine
with a fake cluster. Starting from only the entry service, the engine must learn the topology, find the critical path
and attach the evidence that it matches what the requests waited for."""

import json
import os

import pytest

from src import config
from src.data_collection import otel_traces
from src.inference.engines import LiveEngine
from src.topologies import TRAINTICKET_LIVE

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "trainticket_traces.json")


class ReplayProm:
    """Stands in for data_collection.trainticket_metrics: serves the recorded traces as the current window."""
    def __init__(self, traces):
        self.last_traces = traces
        stats = otel_traces.window_stats(traces, window_s=60.0)
        self.rows = {s: {**v, "inflight": 0.0, "pods_scraped": 1, "cpu_cores": 0.05 + 0.002 * v["requests"],
                         "mem_mb": 350.0, "ok": True} for s, v in stats.items()}

    def collect(self, window=None):
        return self.rows

    @staticmethod
    def node_memory_headroom_mb():
        return 4000.0


class Cluster:
    def __init__(self, services):
        self.state = {s: {"replicas": 1, "ready": 1, "cpu": 1.0, "mem": 800.0} for s in services}
        self.applied = []

    def get_state(self, services, namespace=None):
        return {s: dict(v) for s, v in self.state.items() if s in services}

    def get_all_state(self, namespace=None):
        return {s: dict(v) for s, v in self.state.items()}

    def apply_action(self, svc, action, namespace=None):
        self.applied.append((svc, dict(action)))
        return {}


@pytest.fixture
def engine(monkeypatch):
    traces = json.load(open(FIXTURE))
    monkeypatch.setattr(config, "METRICS_SOURCE", "trainticket")
    monkeypatch.setattr(config, "TOPOLOGY", TRAINTICKET_LIVE)
    monkeypatch.setattr(config, "SERVICES", list(TRAINTICKET_LIVE.services))
    monkeypatch.setattr(config, "ENTRY_SERVICE", TRAINTICKET_LIVE.entry)
    monkeypatch.setattr(config, "CALL_GRAPH", {k: list(v) for k, v in TRAINTICKET_LIVE.call_graph.items()})
    monkeypatch.setattr(config, "CONTROLLED_SERVICES", ["ts-order-service"])
    e = LiveEngine(None, None, controller="hpa", shadow=True)
    assert e.services == ["ts-gateway-service"]                      # nothing declared but the entry point
    e._prom = ReplayProm(traces)
    # the cluster also has Deployments that never appear in the traces, and the traces mention a service with no Deployment
    names = sorted({p["serviceName"] for t in traces for p in t["processes"].values()} - {"ts-station-service"}) + ["ts-auth-service"]
    e._k8s = Cluster(names)
    return e


def test_engine_learns_topology_and_proves_the_critical_path(engine):
    p = engine.tick()
    assert not p["stale"], p.get("stale_reason")
    assert p["topology"]["source"] == "jaeger-live"
    assert "ts-station-service" not in engine.services               # traced, but no Deployment: external, not an error
    assert len(engine.services) == 10 and p["topology"]["edges"] >= 15
    assert p["critical_path"][0] == "ts-gateway-service" and "ts-order-service" in p["critical_path"]
    ev = p["critical_path_evidence"]
    assert ev["coverage"] > 0.7 and ev["top1"] > 0.8 and ev["random"]["coverage"] < ev["coverage"]
    assert ev["slowest_trace"] and p["jaeger_url"]
    assert {n["id"]: n["controlled"] for n in p["nodes"]}["ts-order-service"] is True
    assert all(not n["controlled"] for n in p["nodes"] if n["id"] != "ts-order-service")
    assert engine._k8s.applied == [] or all(s == "ts-order-service" for s, _ in engine._k8s.applied)


def test_untraced_running_services_are_observed_as_isolated_nodes_never_controlled(engine, monkeypatch):
    monkeypatch.setattr(config, "OBSERVE_UNTRACED", True)
    engine._k8s.state["ts-auth-service"]["ready"] = 1
    engine._k8s.state["ts-paused-service"] = {"replicas": 0, "ready": 0, "cpu": 1.0, "mem": 800.0}
    p = engine.tick()
    ids = {n["id"] for n in p["nodes"]}
    assert "ts-auth-service" in ids and "ts-paused-service" not in ids        # running yes, paused no
    assert not any(e["source"] == "ts-auth-service" or e["target"] == "ts-auth-service" for e in p["edges"])
    auth = next(n for n in p["nodes"] if n["id"] == "ts-auth-service")
    assert auth["controlled"] is False and all(a == "ts-order-service" for a, _ in engine._k8s.applied)


def test_payload_carries_the_critical_profile_and_defaults_to_the_dfs_features(engine):
    p = engine.tick()
    prof = p["critical_profile"]
    assert prof["requests"] > 20 and prof["coverage"] >= 0.9                  # the set explains at least 90 % of the downstream time
    assert prof["set"][0] == "ts-order-service" and list(prof["overall"])[0] == "ts-order-service"
    assert prof["types"] and all(len(t["top"]) <= 3 for t in prof["types"])
    assert list(prof["tail"]) and sum(prof["overall"].values()) <= 1.0001
    assert config.CRITICAL_PATH_SOURCE == "dfs"                               # the shipped models were trained with the DFS chain


def test_crisp_features_have_the_same_shape_but_describe_the_measured_critical_set(engine, monkeypatch):
    engine.tick()
    from src.topology_intel.graph_builder import build_observation
    metrics = {s: {"requests": 1.0, "latency_ms": 10.0 + i, "latency_p95_ms": 20.0, "utilization": 0.3, "replicas": 1, "cpu": 1.0,
                   "mem": 800.0, "edge_p95": {}} for i, s in enumerate(engine.services)}
    e2e = {"mean_ms": 100.0, "p50_ms": 90.0, "p95_ms": 200.0, "p99_ms": 300.0, "error_rate": 0.0}
    args = (engine.services, engine.call_graph, engine.entry, metrics, e2e, 10.0, 33.0, 1400.0)
    a = build_observation(*args)
    b = build_observation(*args, critical_override={"set": ["ts-order-service"], "shares": {"ts-order-service": 0.74}})
    assert a["X"].shape == b["X"].shape                                       # no retraining of the architecture needed
    i = engine.services.index("ts-order-service")
    assert b["X"][i, 15] == pytest.approx(0.74) and b["critical_set"] == ["ts-order-service"]
    assert b["critical_path"] == a["critical_path"]                           # the chain used for display is unchanged
    others = [j for j in range(len(engine.services)) if j != i and engine.services[j] != engine.entry]
    assert all(b["X"][j, 14] == 0.0 for j in others)                          # off the critical set


def test_services_discovered_later_get_the_measured_cpu_priors(engine):
    engine.tick()
    assert engine.builder.priors["ts-order-service"] == pytest.approx(81.6)       # discovered after start-up, still has its measured prior
    assert engine.builder.priors["ts-station-service"] == pytest.approx(6.6)


def test_vetoed_scale_up_does_not_start_a_cooldown(engine, monkeypatch):
    """The guard approves, the node-memory check vetoes: no pod was added, so no stabilisation window may start."""
    monkeypatch.setattr(config, "SCALE_DOWN_AFTER_UP_STEPS", 10)
    monkeypatch.setattr(config, "NODE_MEMORY_RESERVE_MB", 10**9)                 # nothing fits: every scale-up is vetoed
    engine.controller = type("C", (), {"name": "x", "decide": lambda self, obs, breaker_open=False: {
        s: {"dr": 1, "cpu": 1.0, "mem": 800.0} for s in engine.services}, "reset": lambda self: None})()
    for _ in range(2):
        engine.tick()
    assert not engine.guard._last_scale_up_step, "a vetoed scale-up must leave no trace in the guard"
