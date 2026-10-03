"""
app.py

ONE real, generic FastAPI microservice, deployed N times (via different env
vars in each Kubernetes Deployment) to form the benchmark topology:

    gateway-service  --0.6--> catalog-service --0.5--> inventory-service
                     --0.4--> auth-service       --0.5--> pricing-service
                                    \--1.0--> user-service

Each instance is a REAL, independently running service with:
  - a genuine CPU-bound workload measured in CPU *time* (not wall-clock), so a
    pod's CPU limit genuinely governs its capacity and CPU throttling genuinely
    slows requests down. This is what makes "resize the CPU allocation" a real
    action rather than a cosmetic one.
  - Prometheus metrics at /metrics: request counter, inclusive latency
    histogram, OWN-work latency histogram, in-flight gauge, error counter, plus
    the default process collector (process_cpu_seconds_total,
    process_resident_memory_bytes) which the autoscaler uses as the real
    CPU / memory utilisation signal.
  - OpenTelemetry distributed tracing exported to Jaeger (real spans, real
    parent/child relationships across services), head-sampled at
    TRACE_SAMPLE_RATIO with the parent's decision honoured downstream.
  - weighted probabilistic calls to downstream services (the weight is the
    fraction of traffic routed to that callee).

Configured entirely via environment variables (set per-Deployment under
k8s/01-services/):
  SERVICE_NAME       - e.g. "gateway-service" (used in traces/metrics)
  DOWNSTREAM         - e.g. "http://catalog-service:8000|0.6,http://auth-service:8000|0.4"
                       empty string for leaf services
  WORK_MS            - milliseconds of CPU time burnt per request
  MAX_CONCURRENCY    - size of the worker thread pool; extra requests queue
  TRACE_SAMPLE_RATIO - fraction of root requests that are traced (default 0.25)
"""

import asyncio
import hashlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown-service")
DOWNSTREAM_RAW = os.environ.get("DOWNSTREAM", "")          # "url|weight,url|weight,..."
WORK_MS = float(os.environ.get("WORK_MS", "15"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
JAEGER_OTLP_ENDPOINT = os.environ.get("JAEGER_OTLP_ENDPOINT", "http://jaeger:4318/v1/traces")
TRACE_SAMPLE_RATIO = float(os.environ.get("TRACE_SAMPLE_RATIO", "0.25"))
DOWNSTREAM_TIMEOUT_S = float(os.environ.get("DOWNSTREAM_TIMEOUT_S", "10"))

DOWNSTREAM_TARGETS: list[tuple[str, float]] = []
if DOWNSTREAM_RAW.strip():
    for part in DOWNSTREAM_RAW.split(","):
        url, weight = part.split("|")
        DOWNSTREAM_TARGETS.append((url.strip(), float(weight)))

# ---------------------------------------------------------------------------
# OpenTelemetry tracing -> Jaeger (real distributed traces)
# ---------------------------------------------------------------------------
resource = Resource.create({"service.name": SERVICE_NAME})
provider = TracerProvider(resource=resource,
                          sampler=ParentBased(TraceIdRatioBased(TRACE_SAMPLE_RATIO)))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=JAEGER_OTLP_ENDPOINT)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(SERVICE_NAME)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
_BUCKETS = (0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.3, 0.45,
            0.7, 1.0, 1.5, 2.5, 5.0, 10.0)
REQUEST_COUNT = Counter("http_requests_total", "Total requests handled", ["service"])
REQUEST_ERRORS = Counter("http_request_errors_total", "Requests that failed", ["service"])
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Inclusive request latency in seconds (includes downstream calls)",
    ["service"], buckets=_BUCKETS)
OWN_LATENCY = Histogram(
    "service_own_duration_seconds",
    "Latency of this service's OWN work (thread-pool queueing + CPU), excluding downstream calls",
    ["service"], buckets=_BUCKETS)
INFLIGHT = Gauge("http_requests_inflight", "Requests currently being handled", ["service"])

# ---------------------------------------------------------------------------
# App + real CPU-bound work
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app):
    yield
    if _client is not None:
        await _client.aclose()


app = FastAPI(title=SERVICE_NAME, lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)
HTTPXClientInstrumentor().instrument()

# A bounded thread pool: once concurrent requests exceed MAX_CONCURRENCY the
# extra requests genuinely queue. Combined with CPU-time-based work (below)
# and the pod's CPU limit, this gives a physically meaningful capacity of
# roughly  min(cpu_limit, 1 core) / WORK_MS  requests/sec per replica.
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)


def _cpu_bound_work(duration_ms: float):
    """Burns `duration_ms` of THIS THREAD'S CPU time by repeated hashing.

    thread_time() only advances while the thread is actually on a CPU, so if the
    pod is CFS-throttled (CPU limit) or the GIL is contended the request takes
    proportionally longer in wall-clock terms - exactly like real CPU-bound work.
    (A wall-clock spin would finish on time regardless of the CPU limit.)
    """
    target = duration_ms / 1000.0
    start = time.thread_time()
    data = SERVICE_NAME.encode()
    while time.thread_time() - start < target:
        data = hashlib.sha256(data).digest()


async def _do_local_work():
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    with tracer.start_as_current_span("cpu_work"):
        await loop.run_in_executor(executor, _cpu_bound_work, WORK_MS)
    OWN_LATENCY.labels(service=SERVICE_NAME).observe(time.monotonic() - t0)


def _choose_downstream() -> str | None:
    if not DOWNSTREAM_TARGETS:
        return None
    urls, weights = zip(*DOWNSTREAM_TARGETS)
    return random.choices(urls, weights=weights, k=1)[0]


# One shared client: avoids a TCP handshake per request (which was both slow
# and a measurable slice of the CPU budget in the per-request-client version).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=DOWNSTREAM_TIMEOUT_S,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
    return _client


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/work")
async def work():
    start = time.monotonic()
    INFLIGHT.labels(service=SERVICE_NAME).inc()
    try:
        with tracer.start_as_current_span(f"{SERVICE_NAME}.handle_request"):
            await _do_local_work()

            downstream_result = None
            target = _choose_downstream()
            if target:
                resp = await _get_client().get(f"{target}/work")
                resp.raise_for_status()
                downstream_result = resp.json()
        body = {
            "service": SERVICE_NAME,
            "latency_ms": round((time.monotonic() - start) * 1000, 2),
            "called": target,
            "downstream": downstream_result,
        }
        return body
    except Exception as e:  # downstream failure/timeout -> 502, counted
        REQUEST_ERRORS.labels(service=SERVICE_NAME).inc()
        return JSONResponse(status_code=502,
                            content={"service": SERVICE_NAME, "error": type(e).__name__})
    finally:
        INFLIGHT.labels(service=SERVICE_NAME).dec()
        elapsed = time.monotonic() - start
        REQUEST_COUNT.labels(service=SERVICE_NAME).inc()
        REQUEST_LATENCY.labels(service=SERVICE_NAME).observe(elapsed)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
