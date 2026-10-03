"""The deployed microservice (services/generic-service/app.py) and the traffic tools."""

import asyncio
import importlib.util
import os
import sys
import time

import httpx
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("prometheus_client")


def _load(name, path, env=None, monkeypatch=None):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _service(monkeypatch, name, downstream="", work_ms="5"):
    # OTLP exporter points nowhere; spans are dropped by the batch processor, which is fine here.
    from opentelemetry import trace
    from prometheus_client import REGISTRY
    for collector, names in list(REGISTRY._collector_to_names.items()):   # app.py registers metrics at import
        if any(n.startswith(("http_", "service_")) for n in names):       # keep the default process collector
            REGISTRY.unregister(collector)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: None)
    return _load(f"svc_{name}", os.path.join(ROOT, "services", "generic-service", "app.py"),
                 {"SERVICE_NAME": name, "DOWNSTREAM": downstream, "WORK_MS": work_ms,
                  "MAX_CONCURRENCY": "2", "TRACE_SAMPLE_RATIO": "0",
                  "JAEGER_OTLP_ENDPOINT": "http://127.0.0.1:9/v1/traces"}, monkeypatch)


def test_leaf_service_health_work_and_metrics(monkeypatch):
    from fastapi.testclient import TestClient
    mod = _service(monkeypatch, "leaf-service")
    c = TestClient(mod.app)
    assert c.get("/health").json() == {"status": "ok", "service": "leaf-service"}
    r = c.get("/work").json()
    assert r["service"] == "leaf-service" and r["called"] is None and r["latency_ms"] >= 5
    text = c.get("/metrics").text
    assert 'http_requests_total{service="leaf-service"} 1.0' in text
    assert 'http_request_duration_seconds_count{service="leaf-service"} 1.0' in text
    # signals the autoscaler depends on
    assert 'service_own_duration_seconds_count{service="leaf-service"} 1.0' in text
    assert 'http_requests_inflight{service="leaf-service"} 0.0' in text
    if os.path.exists("/proc/self/stat"):          # the process collector reads /proc: Linux (= the pod) only
        assert "process_cpu_seconds_total" in text and "process_resident_memory_bytes" in text


def test_own_latency_excludes_downstream_but_inclusive_includes_it(monkeypatch):
    from fastapi.testclient import TestClient
    mod = _service(monkeypatch, "mid", "http://down:8000|1.0", work_ms="5")

    class SlowDownstream:
        async def get(self, url):
            await asyncio.sleep(0.15)
            return httpx.Response(200, json={"service": "down"}, request=httpx.Request("GET", url))
    monkeypatch.setattr(mod, "_get_client", lambda: SlowDownstream())
    c = TestClient(mod.app)
    assert c.get("/work").status_code == 200
    text = c.get("/metrics").text
    val = lambda metric: float([l for l in text.splitlines() if l.startswith(metric + "{")][0].split()[-1])
    own, incl = val("service_own_duration_seconds_sum"), val("http_request_duration_seconds_sum")
    assert own < 0.1 and incl > 0.14 and incl > own + 0.1


def test_downstream_failure_returns_502_and_is_counted(monkeypatch):
    from fastapi.testclient import TestClient
    mod = _service(monkeypatch, "mid2", "http://down:8000|1.0")

    class Boom:
        async def get(self, url):
            raise httpx.ConnectError("refused")
    monkeypatch.setattr(mod, "_get_client", lambda: Boom())
    c = TestClient(mod.app)
    r = c.get("/work")
    assert r.status_code == 502 and r.json()["error"] == "ConnectError"
    text = c.get("/metrics").text
    assert 'http_request_errors_total{service="mid2"} 1.0' in text
    assert 'http_requests_inflight{service="mid2"} 0.0' in text, "in-flight gauge must not leak on errors"
    assert 'http_requests_total{service="mid2"} 1.0' in text


def test_downstream_spec_parsing_and_weighted_choice(monkeypatch):
    mod = _service(monkeypatch, "mid-service", "http://a:8000|0.9,http://b:8000|0.1")
    assert mod.DOWNSTREAM_TARGETS == [("http://a:8000", 0.9), ("http://b:8000", 0.1)]
    picks = [mod._choose_downstream() for _ in range(2000)]
    assert 0.85 < picks.count("http://a:8000") / len(picks) < 0.95
    assert _service(monkeypatch, "leaf2")._choose_downstream() is None


def test_cpu_work_burns_cpu_time_not_wall_clock(monkeypatch):
    """A CPU limit can only slow the work down if the work is measured in CPU time."""
    mod = _service(monkeypatch, "burn")
    t, c = time.monotonic(), time.thread_time()
    mod._cpu_bound_work(40)
    assert 0.035 < time.thread_time() - c < 0.2
    assert 0.035 < time.monotonic() - t < 0.2
    # while another thread hogs the GIL, wall time stretches but the CPU time requested does not
    import threading
    stop = threading.Event()
    hog = threading.Thread(target=lambda: [None for _ in iter(stop.is_set, True)], daemon=True)
    hog.start()
    t, c = time.monotonic(), time.thread_time()
    mod._cpu_bound_work(40)
    wall, cpu = time.monotonic() - t, time.thread_time() - c
    stop.set()
    assert cpu < 0.2 and wall >= cpu


# -------------------------------------------------------------- traffic tools
def _gen():
    return _load("traffic_gen", os.path.join(ROOT, "traffic-generator", "generate_traffic.py"))


def test_pattern_curve_is_the_simulators_curve():
    sys.path.insert(0, os.path.join(ROOT, "autoscaler"))
    import numpy as np
    from src.simulation.simulator import generate_traffic
    gen = _gen()
    for pattern in ("normal", "burst", "diurnal", "rampup"):
        curve = gen.pattern_curve(pattern, 40, 12.0, seed=7)
        expected = generate_traffic(pattern, 40, 12.0, np.random.default_rng(7))
        assert curve == pytest.approx(list(expected))


def test_open_loop_generator_hits_target_rate_and_reports_percentiles(tmp_path):
    gen = _gen()

    async def handler(request):
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={})

    real = httpx.AsyncClient

    class Mocked(real):
        def __init__(self, *a, **k):
            k["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **k)
    gen.httpx.AsyncClient = Mocked
    try:
        out = tmp_path / "t.json"
        summary = asyncio.run(gen.run("http://x", [40.0, 40.0], step_seconds=1.0, out=str(out), verbose=False))
    finally:
        gen.httpx.AsyncClient = real
    assert len(summary) == 2 and out.exists()
    for step in summary:
        assert 20 <= step["ok"] <= 70 and step["errors"] == 0           # Poisson around 40
        assert step["p50_ms"] is not None and step["p95_ms"] >= step["p50_ms"]


def test_generator_counts_errors():
    gen = _gen()
    real = httpx.AsyncClient

    class Failing(real):
        def __init__(self, *a, **k):
            k["transport"] = httpx.MockTransport(lambda r: httpx.Response(500))
            super().__init__(*a, **k)
    gen.httpx.AsyncClient = Failing
    try:
        summary = asyncio.run(gen.run("http://x", [30.0], step_seconds=1.0, out=None, verbose=False))
    finally:
        gen.httpx.AsyncClient = real
    assert summary[0]["errors"] > 5 and summary[0]["ok"] == 0


def test_calibrate_percentile_helper_and_scrape_regex():
    cal = _load("calibrate", os.path.join(ROOT, "traffic-generator", "calibrate.py"))
    vals = list(range(1, 101))
    assert cal._percentile(vals, 0.5) == pytest.approx(50.5) and cal._percentile(vals, 0.95) == pytest.approx(95.05)
    assert cal._percentile([], 0.5) != cal._percentile([], 0.5)          # NaN

    async def go():
        transport = httpx.MockTransport(lambda r: httpx.Response(
            200, text="process_cpu_seconds_total 12.5\nprocess_resident_memory_bytes 6.29145e+07\n"))
        async with httpx.AsyncClient(transport=transport) as c:
            return await cal._scrape(c, "http://x")
    cpu, rss = asyncio.run(go())
    assert cpu == 12.5 and rss == pytest.approx(62914500.0)
