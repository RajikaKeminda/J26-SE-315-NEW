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
