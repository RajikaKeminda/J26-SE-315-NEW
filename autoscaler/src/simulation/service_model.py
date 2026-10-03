"""
service_model.py

The physics of ONE replica of the generic service, fitted to measurements taken
on a real Kubernetes cluster (calibration/leaf_cpu*.json, produced by
traffic-generator/calibrate.py; fit + error report: `python -m src.simulation.calibration`).

What the measurements showed, and what is modelled:

  * Capacity.   A request costs ~15.8 ms of CPU, so a replica saturates at
                min(cpu_limit, 1 core) / 15.8 ms  ~= 63 req/s per core
                (the service is Python behind the GIL: >1 core is useless).
  * Latency.    mean = base + kappa * (service time at this CPU limit) * rho/(1-rho),
                an M/M/1-style queueing term with kappa ~ 0.67 (measured).
  * CFS.        With a CPU limit below request-cost/period (e.g. 0.1 core -> 10 ms
                quota per 100 ms period vs ~16 ms of work) every request is
                throttled into a second period: +100 ms. This is why tiny CPU
                limits are expensive in latency even when the pod is nearly idle.
  * Tail.       p95/mean rises from ~1.45 (idle) to ~2.4 (rho >= 0.5); p99/p95 ~ 1.35.
  * Memory.     Resident set is ~55 MiB almost independent of load.

`rho` here is OFFERED load / capacity and may exceed 1; the simulator turns the
excess into a backlog (see simulator.py). The functions are pure so the same
model is used by the simulator, the calibration fit and the tests.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, fields

from src import config


@dataclass
class ModelParams:
    cpu_overhead_ms: float = 0.8        # CPU ms per request on top of WORK_MS
    idle_cores: float = 0.012           # CPU burnt by an idle replica (exporters, scrapes)
    base_extra_ms: float = 8.0          # non-CPU latency floor above the CPU time
    kappa: float = 0.67                 # queueing-term multiplier
    cfs_period_ms: float = 100.0
    gil_cores: float = 1.0              # max useful cores per replica
    rho_cap: float = 0.92               # queueing term is evaluated at min(rho, rho_cap)
    tail_idle: float = 1.4              # p95/mean at rho -> 0
    tail_busy: float = 2.4              # p95/mean at rho >= tail_knee
    tail_knee: float = 0.5
    p99_over_p95: float = 1.35
    mem_base_mb: float = 55.0
    mem_per_inflight_mb: float = 0.25
    request_timeout_s: float = 10.0     # backlog older than this is dropped (client timeout)


_PARAMS: ModelParams | None = None


def load_params(path: str | None = None) -> ModelParams:
    """Fitted parameters from sim_params.json, falling back to the dataclass defaults."""
    path = path or config.SIM_PARAMS_PATH
    p = ModelParams()
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
        known = {f_.name for f_ in fields(ModelParams)}
        for k, v in blob.get("params", blob).items():
            if k in known:
                setattr(p, k, float(v))
    return p


def params() -> ModelParams:
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = load_params()
    return _PARAMS


def set_params(p: ModelParams | None):
    """Override (tests / calibration). None resets to the file/defaults."""
    global _PARAMS
    _PARAMS = p


def params_dict() -> dict:
    return asdict(params())


# ---------------------------------------------------------------------------
def cpu_ms_per_request(work_ms: float, p: ModelParams | None = None) -> float:
    p = p or params()
    return work_ms + p.cpu_overhead_ms


def capacity_rps(cpu_cores: float, work_ms: float, p: ModelParams | None = None) -> float:
    """Requests/sec ONE replica can sustain at this CPU allocation."""
    p = p or params()
    useful = min(cpu_cores, p.gil_cores)
    return max(useful - p.idle_cores, 1e-6) / (cpu_ms_per_request(work_ms, p) / 1000.0)


def base_latency_ms(cpu_cores: float, work_ms: float, p: ModelParams | None = None) -> float:
    """Latency of one request on an otherwise idle replica."""
    p = p or params()
    w = cpu_ms_per_request(work_ms, p)
    quota_ms = max(cpu_cores, 1e-6) * p.cfs_period_ms
    throttled_periods = max(0, math.ceil(w / quota_ms) - 1)
    return w + p.base_extra_ms + throttled_periods * p.cfs_period_ms


def mean_latency_ms(rho: float, cpu_cores: float, work_ms: float,
                    p: ModelParams | None = None) -> float:
    """Mean OWN latency (queueing + service) of one replica at offered load `rho`."""
    p = p or params()
    w = cpu_ms_per_request(work_ms, p)
    service_at_limit = w / max(min(cpu_cores, p.gil_cores), 1e-6)
    r = min(max(rho, 0.0), p.rho_cap)
    queue = p.kappa * service_at_limit * r / (1.0 - r)
    return base_latency_ms(cpu_cores, work_ms, p) + queue


def tail_factor(rho: float, p: ModelParams | None = None) -> float:
    """p95 / mean as a function of offered load."""
    p = p or params()
    x = min(1.0, max(0.0, rho) / max(p.tail_knee, 1e-6))
    return p.tail_idle + (p.tail_busy - p.tail_idle) * x


def memory_working_set_mb(inflight_per_replica: float, p: ModelParams | None = None) -> float:
    p = p or params()
    return p.mem_base_mb + p.mem_per_inflight_mb * max(0.0, inflight_per_replica)
