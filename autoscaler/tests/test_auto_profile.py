"""auto_profile: the settings that were hand-written for Train-Ticket are derived from the cluster itself."""

import json
import os

import pytest

from src.inference import auto_profile as A

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "trainticket_traces.json")


def test_entry_is_the_most_common_trace_root():
    assert A.choose_entry({"gw": 61, "order": 3}) == "gw" and A.choose_entry({}) is None


def test_startup_is_the_median_of_plausible_durations():
    assert A.estimate_startup_seconds([60, 70, 94, 0.2, 5000]) == 70          # the absurd ones are ignored
    assert A.estimate_startup_seconds([]) == 4.0


def test_step_is_a_third_of_startup_within_limits():
    assert A.derive_timing(94)["STEP_SECONDS"] == 30 and A.derive_timing(4)["STEP_SECONDS"] == 10
    assert A.derive_timing(600)["STEP_SECONDS"] == 60
    t = A.derive_timing(94)
    assert t["POD_STARTUP_SECONDS"] == 94 and t["JAEGER_LOOKBACK_SECONDS"] >= 60


def test_bounds_follow_current_limits_and_measured_memory():
    deps = {"a": {"replicas": 1, "cpu": 1.0, "mem": 800.0}, "b": {"replicas": 1, "cpu": 0.5, "mem": 800.0}}
    b = A.derive_bounds(deps, {"a": 410.0, "b": 300.0}, node_headroom_mb=900.0)
    assert b["CPU_MAX"] == 1.0 and 0.1 <= b["CPU_MIN"] <= 0.25
    assert b["MEM_MIN_MB"] == 576 and b["MEM_MIN_MB"] >= 1.25 * 410           # never below 1.25x the largest working set
    assert b["MEM_MAX_MB"] == 832 and b["MAX_REPLICAS"] == 2                  # a nearly full node allows no big scale-out
    roomy = A.derive_bounds(deps, {"a": 300.0}, node_headroom_mb=64_000.0)
    assert roomy["MAX_REPLICAS"] == 6                                           # capped, even with plenty of room
    tight = A.derive_bounds({"a": {"replicas": 1, "cpu": 1.0, "mem": 300.0}}, {"a": 400.0})
    assert tight["MEM_MAX_MB"] > tight["MEM_MIN_MB"]                            # limits below the working set: made workable


def test_allocation_cost_needs_limits():
    assert A.allocation_cost({"a": {"replicas": 2, "cpu": 1.0, "mem": 1024.0}}) == pytest.approx(2 * (1.0 + 0.13))
    assert A.allocation_cost({"a": {"replicas": 1, "cpu": None, "mem": 128.0}}) is None


def _probes(traced_ready=True):
    deps = {n: {"replicas": 1, "ready": 1 if traced_ready else 0, "cpu": 1.0, "mem": 800.0}
            for n in ("ts-gateway-service", "ts-order-service", "ts-auth-service")}
    prom = {"cadvisor": True, "kube_state_metrics": True, "demo_request_metrics": False, "istio_request_metrics": False,
            "node_headroom_mb": 911.0, "working_set_mb": {"ts-gateway-service": 340.0, "ts-order-service": 410.0}}
    traces = json.load(open(FIXTURE))

    def probe_traces(jaeger, names, get=None, per_service=8):
        from src.data_collection import otel_traces
        roots = {}
        for t in traces[:40]:
            for r in otel_traces.analyse_trace(t):
                if r["parent_service"] is None:
                    roots[r["service"]] = roots.get(r["service"], 0) + 1
        return roots, ["ts-gateway-service", "ts-order-service"]            # ts-auth-service emits no traces
    return {"cluster": lambda ns: (deps, [60.0, 70.0, 94.0]), "prometheus": lambda url, ns: prom, "traces": probe_traces}


def test_discovery_derives_a_complete_safe_profile_from_real_traces():
    settings, report = A.discover("default", "http://p", "http://j", probes=_probes())
    assert settings["METRICS_SOURCE"] == "otel" and settings["AUTOSCALER_ENTRY_SERVICE"] == "ts-gateway-service"
    assert settings["CONTROLLED_SERVICES"] == "ts-gateway-service,ts-order-service"        # traced and running only
    assert settings["ALLOW_RESIZE"] == 0 and settings["STEP_SECONDS"] == 25                  # 70 s median start-up
    assert settings["CLUSTER_BUDGET_CORES"] == pytest.approx(2 * (1.0 + 0.13 * 800 / 1024) * 1.3, abs=0.1)
    env = A.render_env(settings)
    assert "AUTOSCALER_ENTRY_SERVICE=ts-gateway-service" in env and "CONTROLLED_SERVICES=ts-gateway-service,ts-order-service" in env
    assert any("entry service = ts-gateway-service" in l for l in report)


def test_discovery_without_traces_or_metrics_warns_instead_of_guessing():
    pr = _probes()
    pr["traces"] = lambda *a, **k: ({}, [])
    settings, report = A.discover("default", "http://p", "http://j", probes=pr)
    assert "AUTOSCALER_ENTRY_SERVICE" not in settings and settings["CONTROLLED_SERVICES"] == ""
    assert any("neither traces nor request metrics" in l for l in report)


def test_entry_can_be_given_and_config_builds_a_live_topology_from_it(monkeypatch):
    import importlib
    from src import config
    monkeypatch.setenv("AUTOSCALER_ENTRY_SERVICE", "front-door")
    monkeypatch.setenv("METRICS_SOURCE", "otel")
    try:
        importlib.reload(config)
        assert config.ENTRY_SERVICE == "front-door" and config.SERVICES == ["front-door"]
        assert config.METRICS_SOURCE == "trainticket"                         # "otel" is the generic name for it
    finally:
        monkeypatch.delenv("AUTOSCALER_ENTRY_SERVICE")
        monkeypatch.delenv("METRICS_SOURCE")
        importlib.reload(config)


def test_metric_window_is_several_scrape_intervals_long():
    assert A.parse_duration_s("30s") == 30 and A.parse_duration_s("2m30s") == 150 and A.parse_duration_s(None) == 15.0
    assert A.derive_timing(94, 30.0)["METRIC_WINDOW"] == "120s"           # 60 s over a 30 s scrape dropped CPU series
    assert A.derive_timing(94, 5.0)["METRIC_WINDOW"] == "60s"             # but a fast scrape needs no more than 2 steps
