"""The deployed microservice (services/generic-service/app.py) and the traffic generator."""

import importlib
import importlib.util
import os
import sys

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
    for collector in list(REGISTRY._collector_to_names):   # app.py registers metrics at import
        REGISTRY.unregister(collector)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: None)
    return _load(f"svc_{name}", os.path.join(ROOT, "services", "generic-service", "app.py"),
                 {"SERVICE_NAME": name, "DOWNSTREAM": downstream, "WORK_MS": work_ms,
                  "MAX_CONCURRENCY": "2",
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


def test_downstream_spec_parsing_and_weighted_choice(monkeypatch):
    mod = _service(monkeypatch, "mid-service", "http://a:8000|0.9,http://b:8000|0.1")
    assert mod.DOWNSTREAM_TARGETS == [("http://a:8000", 0.9), ("http://b:8000", 0.1)]
    picks = [mod._choose_downstream() for _ in range(2000)]
    assert 0.85 < picks.count("http://a:8000") / len(picks) < 0.95
    assert _service(monkeypatch, "leaf2")._choose_downstream() is None


def test_cpu_work_burns_roughly_the_requested_time(monkeypatch):
    import time
    mod = _service(monkeypatch, "burn")
    t = time.monotonic()
    mod._cpu_bound_work(40)
    assert 0.035 < time.monotonic() - t < 0.2


def test_traffic_curves():
    gen = _load("traffic_gen", os.path.join(ROOT, "traffic-generator", "generate_traffic.py"))
    for pattern in ("normal", "burst", "diurnal"):
        curve = gen.traffic_curve(pattern, 60, 20.0)
        assert len(curve) == 60 and min(curve) >= 1
    burst = gen.traffic_curve("burst", 60, 20.0)
    assert max(burst) == pytest.approx(120.0) and burst[0] == 20.0   # 6x spike in the middle
    d = gen.traffic_curve("diurnal", 60, 20.0)
    assert d.index(max(d)) > 20 and d[0] < max(d)
    with pytest.raises(ValueError):
        gen.traffic_curve("sawtooth", 10, 1.0)
