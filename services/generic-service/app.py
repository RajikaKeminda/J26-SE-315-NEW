"""
app.py

ONE real, generic FastAPI microservice, deployed 6 times (via different env
vars in each Kubernetes Deployment) to form the full topology:

    gateway-service  --0.6--> catalog-service --0.5--> inventory-service
                     --0.4--> auth-service       --0.5--> pricing-service
                                    \--1.0--> user-service

Each instance is a REAL, independently running service with:
  - a genuine CPU-bound workload (so more replicas == more real throughput)
  - Prometheus metrics exposed at /metrics
  - OpenTelemetry distributed tracing exported to Jaeger (real spans,
    real parent/child relationships across services)
  - weighted probabilistic calls to downstream services (mirrors the
    topology weight = "fraction of traffic routed" convention used
    throughout this project)

Configured entirely via environment variables (set per-Deployment in the
Kubernetes manifests under k8s/01-services/):
  SERVICE_NAME      - e.g. "gateway-service" (used in traces/logs)
  DOWNSTREAM        - e.g. "http://catalog-service:8000|0.6,http://auth-service:8000|0.4"
                      empty string for leaf services
  WORK_MS           - how many milliseconds of real CPU work this service
                      does per request (simulates real processing cost)
  MAX_CONCURRENCY   - size of the thread pool used for CPU work; this is
                      what creates genuine queueing/backpressure once
                      concurrent requests exceed capacity
"""

import hashlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import FastAPI, Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown-service")
DOWNSTREAM_RAW = os.environ.get("DOWNSTREAM", "")          # "url|weight,url|weight,..."
WORK_MS = float(os.environ.get("WORK_MS", "15"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
JAEGER_OTLP_ENDPOINT = os.environ.get("JAEGER_OTLP_ENDPOINT", "http://jaeger:4318/v1/traces")

DOWNSTREAM_TARGETS: list[tuple[str, float]] = []
if DOWNSTREAM_RAW.strip():
    for part in DOWNSTREAM_RAW.split(","):
        url, weight = part.split("|")
        DOWNSTREAM_TARGETS.append((url.strip(), float(weight)))

# ---------------------------------------------------------------------------
# OpenTelemetry tracing -> Jaeger (real distributed traces)
# ---------------------------------------------------------------------------
resource = Resource.create({"service.name": SERVICE_NAME})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=JAEGER_OTLP_ENDPOINT)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(SERVICE_NAME)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter("http_requests_total", "Total requests handled", ["service"])
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency in seconds", ["service"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# App + real CPU-bound work simulation
# ---------------------------------------------------------------------------
app = FastAPI(title=SERVICE_NAME)
FastAPIInstrumentor.instrument_app(app)
HTTPXClientInstrumentor().instrument()

# a bounded thread pool: this IS the service's real capacity. Once concurrent
# requests exceed MAX_CONCURRENCY, extra requests genuinely queue and wait -
# real backpressure, not simulated. More replicas = more total pool capacity
# across the cluster via Kubernetes Service load-balancing.
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)


def _cpu_bound_work(duration_ms: float):
    """Burns real CPU for approximately duration_ms by repeated hashing."""
    end = time.monotonic() + duration_ms / 1000.0
    data = SERVICE_NAME.encode()
    while time.monotonic() < end:
        data = hashlib.sha256(data).digest()


async def _do_local_work():
    loop = __import__("asyncio").get_event_loop()
    with tracer.start_as_current_span("cpu_work"):
        await loop.run_in_executor(executor, _cpu_bound_work, WORK_MS)


def _choose_downstream() -> str | None:
    if not DOWNSTREAM_TARGETS:
        return None
    urls, weights = zip(*DOWNSTREAM_TARGETS)
    return random.choices(urls, weights=weights, k=1)[0]


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/work")
async def work():
    start = time.monotonic()
    with tracer.start_as_current_span(f"{SERVICE_NAME}.handle_request"):
        await _do_local_work()

        downstream_result = None
        target = _choose_downstream()
        if target:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{target}/work")
                downstream_result = resp.json()

    elapsed_ms = (time.monotonic() - start) * 1000
    REQUEST_COUNT.labels(service=SERVICE_NAME).inc()
    REQUEST_LATENCY.labels(service=SERVICE_NAME).observe(elapsed_ms / 1000.0)

    return {
        "service": SERVICE_NAME,
        "latency_ms": round(elapsed_ms, 2),
        "called": target,
        "downstream": downstream_result,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
