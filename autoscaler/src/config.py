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

from src.topologies import DEMO6, get_topology

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../autoscaler


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_int(name, default):
    return int(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Active topology (the application deployed under k8s/01-services/)
# ---------------------------------------------------------------------------
TOPOLOGY = get_topology(os.environ.get("AUTOSCALER_TOPOLOGY", DEMO6.name))
SERVICES = list(TOPOLOGY.services)
ENTRY_SERVICE = TOPOLOGY.entry
CALL_GRAPH = {k: list(v) for k, v in TOPOLOGY.call_graph.items()}

NAMESPACE = os.environ.get("AUTOSCALER_NAMESPACE", "autoscaler-poc")

# ---------------------------------------------------------------------------
# Observability endpoints (assume `kubectl port-forward`, see README)
# ---------------------------------------------------------------------------
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
JAEGER_QUERY_URL = os.environ.get("JAEGER_QUERY_URL", "http://localhost:16686")
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
CPU_MIN, CPU_MAX, CPU_QUANTUM, CPU_DEFAULT = 0.1, 1.0, 0.05, 0.25     # cores / pod
MEM_MIN_MB, MEM_MAX_MB, MEM_QUANTUM_MB, MEM_DEFAULT_MB = 96, 512, 32, 128
MIN_REPLICAS = 1
MAX_REPLICAS = 8
INITIAL_REPLICAS = 2
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
SCALE_DOWN_COOLDOWN_STEPS = 3     # asymmetric: scaling DOWN is slower than UP
RESIZE_COOLDOWN_STEPS = 6         # a resize is a rolling restart: do not thrash
RESIZE_DEADBAND = 0.15            # ignore resizes smaller than 15 %
CIRCUIT_BREAKER_VIOLATIONS = 5    # consecutive SLO breaches -> fall back to HPA rules
METRIC_STALENESS_SECONDS = 45     # refuse to act on data older than this

# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = _env_int("POLL_INTERVAL_SECONDS", int(STEP_SECONDS))
JAEGER_LOOKBACK_SECONDS = 30
JAEGER_TRACE_LIMIT = 100
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
SIM_PARAMS_PATH = os.path.join(BASE_DIR, "src", "simulation", "sim_params.json")

# Discover the call graph from live Jaeger traces instead of the declared one.
AUTODISCOVER_TOPOLOGY = os.environ.get("AUTODISCOVER_TOPOLOGY", "0") == "1"
