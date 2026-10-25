"""
config.py  -  single source of truth for the autoscaler.

Deployment-specific values (endpoints, namespace, polling) can be overridden
with environment variables so the same code runs on a laptop, in docker
compose, or in-cluster without edits.

Nothing here bakes the NUMBER of services into the model architecture. The
topology in use is described by `src/topologies.py`; the shared policy
(rl_agent/mo_td3.py) works for any node count.
"""

import os

from src.topologies import DEMO6, Topology, get_topology

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../autoscaler


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_int(name, default):
    return int(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Active topology (the application deployed under k8s/01-services/)
# ---------------------------------------------------------------------------
# Pointing the autoscaler at an application needs no code: name its entry service and the rest (services, call
# graph) is learned from live traces. Otherwise a named topology from src/topologies.py is used.
_ENTRY = os.environ.get("AUTOSCALER_ENTRY_SERVICE")
TOPOLOGY = (Topology(name=f"live:{_ENTRY}", services=[_ENTRY], entry=_ENTRY, call_graph={_ENTRY: []})
            if _ENTRY else get_topology(os.environ.get("AUTOSCALER_TOPOLOGY", DEMO6.name)))
SERVICES = list(TOPOLOGY.services)
ENTRY_SERVICE = TOPOLOGY.entry
CALL_GRAPH = {k: list(v) for k, v in TOPOLOGY.call_graph.items()}

NAMESPACE = os.environ.get("AUTOSCALER_NAMESPACE", "autoscaler-poc")

# ---------------------------------------------------------------------------
# Observability endpoints (assume `kubectl port-forward`, see README)
# ---------------------------------------------------------------------------
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
JAEGER_PUBLIC_URL = os.environ.get("JAEGER_PUBLIC_URL", os.environ.get("JAEGER_QUERY_URL", "http://localhost:16686"))   # link target for the browser
JAEGER_QUERY_URL = os.environ.get("JAEGER_QUERY_URL", "http://localhost:16686")
PROMETHEUS_TIMEOUT_SECONDS = float(os.environ.get("PROMETHEUS_TIMEOUT_SECONDS", "5"))   # raise over an SSH tunnel
METRIC_WINDOW = os.environ.get("METRIC_WINDOW", "10s")     # PromQL range for rates/quantiles

# ---------------------------------------------------------------------------
# Time. One control step = one observation window = one decision.
# Calibrated on the real cluster (calibration/): a new pod is Ready after ~4 s and a
# rolling resize costs ~4.4 s per replica (maxSurge=1, maxUnavailable=0).
# ---------------------------------------------------------------------------
STEP_SECONDS = _env_float("STEP_SECONDS", 10.0)
POD_STARTUP_SECONDS = _env_float("POD_STARTUP_SECONDS", 4.0)
ROLLING_SECONDS_PER_REPLICA = _env_float("ROLLING_SECONDS_PER_REPLICA", 4.4)

# ---------------------------------------------------------------------------
# Workload / simulation scale.  BASE_RPS is sized for a laptop-class cluster: the
# whole 6-service app costs ~48 ms of CPU per external request, so 12 rps is ~0.6
# cores of work and a 4x burst (48 rps) is ~2.3 cores - inside the cluster budget.
# ---------------------------------------------------------------------------
BASE_RPS = _env_float("BASE_RPS", 12.0)
EPISODE_LENGTH_STEPS = 40
WORK_MS = _env_float("WORK_MS", 15.0)    # CPU ms per request burnt by services/generic-service

# ---------------------------------------------------------------------------
# Resources per pod (requests == limits -> Guaranteed QoS, so "allocation" is
# unambiguous) and the cluster budget the agent must live within.
# ---------------------------------------------------------------------------
# All overridable (see profiles/trainticket.env): a JVM service needs far more memory than the demo app.
CPU_MIN, CPU_MAX = _env_float("CPU_MIN", 0.1), _env_float("CPU_MAX", 1.0)             # cores / pod
CPU_QUANTUM, CPU_DEFAULT = _env_float("CPU_QUANTUM", 0.05), _env_float("CPU_DEFAULT", 0.25)
MEM_MIN_MB, MEM_MAX_MB = _env_int("MEM_MIN_MB", 96), _env_int("MEM_MAX_MB", 512)
MEM_QUANTUM_MB, MEM_DEFAULT_MB = _env_int("MEM_QUANTUM_MB", 32), _env_int("MEM_DEFAULT_MB", 128)
MIN_REPLICAS = _env_int("MIN_REPLICAS", 1)
MAX_REPLICAS = _env_int("MAX_REPLICAS", 8)
INITIAL_REPLICAS = _env_int("INITIAL_REPLICAS", 2)
# 1 vCPU ~ 7.5 GiB of RAM on the common cloud price lists -> 0.13 core-equivalents per GiB
MEM_COST_PER_GIB = 0.13
# Budget in core-equivalents per service. Scaling is bounded by what the cluster can
# schedule, so "scale everything to the maximum" is not an available action.
BUDGET_PER_SERVICE = _env_float("BUDGET_PER_SERVICE", 1.0)
# Live safety cap: the real cluster can schedule only what its nodes can hold. Set this to your
# cluster's free CPU (in cores) so the guard never asks for pods the scheduler cannot place.
CLUSTER_BUDGET_CORES = float(os.environ["CLUSTER_BUDGET_CORES"]) if os.environ.get("CLUSTER_BUDGET_CORES") else None


def live_budget(n_services: int) -> float:
    per_service = BUDGET_PER_SERVICE * n_services
    return min(per_service, CLUSTER_BUDGET_CORES) if CLUSTER_BUDGET_CORES else per_service

# ---------------------------------------------------------------------------
# Hybrid action space (per service):
#   discrete   replica delta in {-3..+3}            (proposal: "-3 to +3")
#   continuous CPU allocation   u in [-1, 1] -> a TARGET between CPU_MIN and CPU_MAX
#   continuous memory allocation u in [-1, 1] -> a TARGET between MEM_MIN and MEM_MAX
#              (log scale: equal steps in u are equal RELATIVE changes in resources).
# Absolute targets make the policy stationary - "what size should this service be?" -
# instead of an integrator; the guard's deadband/cooldown stops needless restarts.
# ---------------------------------------------------------------------------
REPLICA_DELTA_OPTIONS = [-3, -2, -1, 0, 1, 2, 3]
N_REPLICA_ACTIONS = len(REPLICA_DELTA_OPTIONS)
NO_OP_INDEX = REPLICA_DELTA_OPTIONS.index(0)
N_CONTINUOUS_ACTIONS = 2

# ---------------------------------------------------------------------------
# SLO: p95 end-to-end latency at the entry service.
#   slo_ms(depth) = SLO_BASE_MS + SLO_PER_HOP_MS * depth   (demo6 has depth 3 -> 250 ms)
# SLO_LATENCY_MS pins an absolute value instead (e.g. for a live cluster).
# ---------------------------------------------------------------------------
SLO_BASE_MS = 70.0
SLO_PER_HOP_MS = 60.0
SLO_LATENCY_MS_OVERRIDE = os.environ.get("SLO_LATENCY_MS")
SLO_LATENCY_MS = float(SLO_LATENCY_MS_OVERRIDE) if SLO_LATENCY_MS_OVERRIDE else (
    SLO_BASE_MS + SLO_PER_HOP_MS * 3)


def slo_for_depth(depth: int) -> float:
    if SLO_LATENCY_MS_OVERRIDE:
        return float(SLO_LATENCY_MS_OVERRIDE)
    return SLO_BASE_MS + SLO_PER_HOP_MS * depth


# ---------------------------------------------------------------------------
# Multi-objective reward. Three objectives, all minimised, each normalised to ~[0, 1]:
#   0 SLO      smooth violation of the p95 end-to-end latency target
#   1 COST     provisioned CPU+memory as a fraction of the cluster budget
#   2 BALANCE  variance of per-service CPU utilisation (load distribution)
# The agent is conditioned on a preference vector w over these objectives.
# ---------------------------------------------------------------------------
OBJECTIVES = ("slo", "cost", "balance")
N_OBJECTIVES = len(OBJECTIVES)
DEFAULT_PREFERENCE = (0.50, 0.35, 0.15)
SLO_VIOLATION_STEP = 0.25     # fixed penalty for ANY p95 > SLO (a breach is a breach)
SLO_SATURATION = 2.0          # the graded part saturates at (1 + 2) x SLO

# ---------------------------------------------------------------------------
# Graph / model dims
# ---------------------------------------------------------------------------
NODE_FEATURE_DIM = 18         # see topology_intel/graph_builder.py
GLOBAL_FEATURE_DIM = 6
GCN_HIDDEN_DIM = 128          # proposal: 2-layer GCN with 128 hidden units
GCN_EMBED_DIM = 32            # per branch (SDG + EHG) -> concatenated = 64
EMBEDDING_DIM = 2 * GCN_EMBED_DIM   # proposal: 64-d graph embedding
POLICY_HIDDEN = 128

# ---------------------------------------------------------------------------
# MO-TD3 training (see rl_agent/mo_td3.py)
# ---------------------------------------------------------------------------
TD3_GAMMA = 0.95
TD3_N_STEP = 3
TD3_TAU = 0.005
TD3_ACTOR_LR = 3e-4
TD3_CRITIC_LR = 6e-4
TD3_BATCH = 128
TD3_POLICY_DELAY = 2
TD3_TARGET_NOISE = 0.15
TD3_NOISE_CLIP = 0.3
TD3_EXPLORE_NOISE = 0.20
TD3_EPSILON_START = 0.20       # random-replica-action probability (decays)
TD3_EPSILON_END = 0.03
TD3_BC_WEIGHT = 1.0            # behaviour-cloning term weight on demonstration samples
TD3_REPLAY_EPISODES = 4000

# ---------------------------------------------------------------------------
# Safety guards (see safety/guards.py).  Counted in control steps.
# ---------------------------------------------------------------------------
# Scaling DOWN is slower than UP, and never right after a scale-UP (a stabilisation window). Counted in control steps; it must
# outlast a pod's start-up (about 4 s for the demo app, 95-240 s for Train-Ticket's JVMs).
# Demand gate (safety/guards.py): the agent may ADD capacity (replicas or CPU) only to a service that is busy (CPU utilisation >= SCALE_UP_MIN_UTIL), and may
# REMOVE capacity only from a service that is not (utilisation <= SCALE_DOWN_MAX_UTIL). It ties every action to where the load really is, like an HPA's target.
# Off by default (0 / 1.01), so the shipped simulator results stay valid; the Train-Ticket profile switches it on.
SCALE_UP_MIN_UTIL = _env_float("SCALE_UP_MIN_UTIL", 0.0)
SCALE_DOWN_MAX_UTIL = _env_float("SCALE_DOWN_MAX_UTIL", 1.01)
SCALE_DOWN_COOLDOWN_STEPS = _env_int("SCALE_DOWN_COOLDOWN_STEPS", 3)
# Stabilisation window AFTER a scale-up (0 = off, which keeps the simulated results and the trained models exactly as shipped).
# The live Train-Ticket profile turns it on: a JVM pod needs minutes to start, so undoing a scale-up within that time wastes it.
SCALE_DOWN_AFTER_UP_STEPS = _env_int("SCALE_DOWN_AFTER_UP_STEPS", 0)
RESIZE_COOLDOWN_STEPS = _env_int("RESIZE_COOLDOWN_STEPS", 6)         # a resize is a rolling restart: do not thrash
RESIZE_DEADBAND = 0.15            # ignore resizes smaller than 15 %
CIRCUIT_BREAKER_VIOLATIONS = 5    # consecutive SLO breaches -> fall back to HPA rules
METRIC_STALENESS_SECONDS = 45     # refuse to act on data older than this

# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = _env_int("POLL_INTERVAL_SECONDS", int(STEP_SECONDS))
JAEGER_LOOKBACK_SECONDS = _env_int("JAEGER_LOOKBACK_SECONDS", 30)
JAEGER_TRACE_LIMIT = _env_int("JAEGER_TRACE_LIMIT", 100)
JAEGER_TIMEOUT_SECONDS = _env_float("JAEGER_TIMEOUT_SECONDS", 5.0)
# Where per-service request rate / latency come from: "prometheus" (the demo services export
# http_request_* metrics) or "trainticket" (OpenTelemetry spans in Jaeger + cAdvisor; see
# data_collection/trainticket_metrics.py).
METRICS_SOURCE = {"otel": "trainticket"}.get(os.environ.get("METRICS_SOURCE", "prometheus"),
                                                      os.environ.get("METRICS_SOURCE", "prometheus"))   # "otel" = any app with OpenTelemetry traces
# Fraction of requests that are traced (rates computed from traces are divided by it).
OTEL_SAMPLE_RATE = _env_float("OTEL_SAMPLE_RATE", 1.0)
TRACE_WINDOW_SECONDS = 5          # proposal: aggregate traces over 5-second sliding windows

# ---------------------------------------------------------------------------
# Baselines (for the evaluation)
# ---------------------------------------------------------------------------
HPA_TARGET_UTILIZATION = 0.60
HPA_TOLERANCE = 0.10
HPA_SCALE_DOWN_STABILIZATION_STEPS = 6      # 60 s (the Kubernetes default is 300 s)
HPA_SCALE_UP_MAX_PODS = 4                    # K8s default: max(4 pods, 100 %) per period
VPA_TARGET_PERCENTILE = 0.90
VPA_SAFETY_MARGIN = 0.15
VPA_WINDOW_STEPS = 12
VPA_MIN_CHANGE = 0.10
THRESHOLD_UP, THRESHOLD_DOWN = 0.80, 0.30

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "results"))
POLICY_PATH = os.path.join(MODELS_DIR, "shared_policy.pt")
DEFAULT_ARM = "mo_td3_full"       # policy served when shared_policy.pt is absent
RESULTS_PATH = os.path.join(RESULTS_DIR, "ablation_results.json")
PARETO_PATH = os.path.join(RESULTS_DIR, "pareto_results.json")
SIM_PARAMS_PATH = os.environ.get("SIM_PARAMS_PATH", os.path.join(BASE_DIR, "src", "simulation", "sim_params.json"))

# Discover the call graph from live Jaeger traces instead of the declared one.
AUTODISCOVER_TOPOLOGY = os.environ.get("AUTODISCOVER_TOPOLOGY", "0") == "1"

# Live topology (METRICS_SOURCE=trainticket): the graph is learned only from traces. Observed call counts
# decay with this half-life; an edge not seen for TTL seconds is dropped; edges seen fewer than MIN_CALLS
# (decayed) times are ignored as noise.
# Also show the running Deployments that emit no traces, as isolated nodes (CPU and memory only, no call edges). They
# are observed, never controlled unless listed in CONTROLLED_SERVICES.
# What the agent's two path features describe: "dfs" = the chain found by the depth-first search of the proposal (what the shipped
# models were trained with); "crisp" = the critical SET measured on real requests (topology_intel/crisp.py). "crisp" changes the meaning
# of the inputs, so it is OFF by default: switch it on only for an agent trained or evaluated with it.
CRITICAL_PATH_SOURCE = os.environ.get("CRITICAL_PATH_SOURCE", "dfs")
OBSERVE_UNTRACED = os.environ.get("OBSERVE_UNTRACED", "0") == "1"
TOPOLOGY_HALF_LIFE_S = _env_float("TOPOLOGY_HALF_LIFE_S", 120.0)
TOPOLOGY_TTL_S = _env_float("TOPOLOGY_TTL_S", 300.0)
TOPOLOGY_MIN_CALLS = _env_float("TOPOLOGY_MIN_CALLS", 2.0)
# Services the agent may ACT on (comma-separated Deployment names). Empty = every discovered service.
# Services outside the list are still observed and shown, but never scaled or resized.
# Real-cluster safety beyond the guard: a scale-up is refused unless the node has this much memory (MiB) left AFTER
# the extra pods (measured working set) are placed; ALLOW_RESIZE=0 restricts the agent to replica scaling
# (a resize is a rolling restart that briefly runs old and new pod side by side).
NODE_MEMORY_RESERVE_MB = _env_float("NODE_MEMORY_RESERVE_MB", 768.0)
ALLOW_RESIZE = os.environ.get("ALLOW_RESIZE", "1") == "1"
# ALLOW_MEMORY_RESIZE=0: the agent may change CPU but never memory (decode_action ignores the memory output, in training and live alike). For a node
# whose memory is the scarce resource, where a memory increase could stall it but a CPU limit costs nothing.
ALLOW_MEMORY_RESIZE = os.environ.get("ALLOW_MEMORY_RESIZE", "1") == "1"
CONTROLLED_SERVICES = [x.strip() for x in os.environ.get("CONTROLLED_SERVICES", "").split(",") if x.strip()]

# ---------------------------------------------------------------------------
# Persistence (src/persistence/): PostgreSQL history of what the agent saw and did, kept apart per cluster.
# Empty AUTOSCALER_DATABASE_URL = off (nothing is written, nothing else changes). Only the modes listed are stored; demo and
# replay never are. The cluster is identified by AUTOSCALER_CLUSTER_NAME and/or the UID of its kube-system namespace.
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("AUTOSCALER_DATABASE_URL", "")
CLUSTER_NAME = os.environ.get("AUTOSCALER_CLUSTER_NAME", "").strip()
PERSIST_MODES = [x.strip() for x in os.environ.get("AUTOSCALER_PERSIST_MODES", "live").split(",") if x.strip()]
STORE_FRAMES = os.environ.get("AUTOSCALER_STORE_FRAMES", "0") == "1"
RETENTION_METRICS_HOURS = _env_float("RETENTION_METRICS_HOURS", 24.0)        # service_metrics and frames
RETENTION_HISTORY_DAYS = _env_float("RETENTION_HISTORY_DAYS", 90.0)          # steps, actions, critical paths, violations, events
CRITICAL_PATH_SNAPSHOT_EVERY = _env_int("CRITICAL_PATH_SNAPSHOT_EVERY", 10)  # steps between stored critical paths (a change is always stored)
