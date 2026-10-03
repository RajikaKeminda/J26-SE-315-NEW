"""
config.py  —  single source of truth for the v2 autoscaler.

Deployment-specific values (endpoints, namespace, polling) can be overridden
with environment variables so the same code runs on a laptop, in docker
compose, or in-cluster without edits.

Key difference from v1: nothing here bakes the NUMBER of services into the
model architecture. SERVICES/CALL_GRAPH describe the *current* topology, but
the shared policy (rl_agent/shared_policy.py) works for any node count.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../autoscaler


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_int(name, default):
    return int(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Current topology (matches k8s/01-services/*.yaml).
# The agent does NOT depend on this being 6 services - see shared_policy.py.
# ---------------------------------------------------------------------------
SERVICES = ["gateway-service", "catalog-service", "auth-service",
            "inventory-service", "pricing-service", "user-service"]
ENTRY_SERVICE = "gateway-service"

CALL_GRAPH = {
    "gateway-service":   [("catalog-service", 0.6), ("auth-service", 0.4)],
    "catalog-service":   [("inventory-service", 0.5), ("pricing-service", 0.5)],
    "auth-service":      [("user-service", 1.0)],
    "inventory-service": [],
    "pricing-service":   [],
    "user-service":      [],
}

NAMESPACE = os.environ.get("AUTOSCALER_NAMESPACE", "autoscaler-poc")

# ---------------------------------------------------------------------------
# Observability endpoints (assume `kubectl port-forward`, see README)
# ---------------------------------------------------------------------------
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
JAEGER_QUERY_URL = os.environ.get("JAEGER_QUERY_URL", "http://localhost:16686")

# ---------------------------------------------------------------------------
# Workload / simulation scale
# ---------------------------------------------------------------------------
BASE_RPS = 400.0    # calibrated so 2 replicas sits near saturation and bursts genuinely breach the SLO
EPISODE_LENGTH_STEPS = 40
WORK_MS = _env_float("WORK_MS", 15.0)   # matches services/generic-service/app.py
# Effective concurrent capacity of ONE replica. 4 matches app.py's MAX_CONCURRENCY,
# but the real service is CPU-bound Python (GIL) so measured capacity is lower on
# small clusters - run scripts/calibrate.py against a live service and override
# via env var (see README, "Sim-to-real calibration").
MAX_CONCURRENCY_PER_REPLICA = _env_float("MAX_CONCURRENCY_PER_REPLICA", 4)

# ---------------------------------------------------------------------------
# Reward (scalarized multi-objective)
# ---------------------------------------------------------------------------
SLO_LATENCY_MS = 150.0   # achievable when well-scaled (~50ms), badly breached when under-provisioned (>600ms)
W_LATENCY = 0.5
W_COST = 0.55   # raised from 0.3: at 0.3 the agent found "max out replicas" optimal (see results/)
W_VARIANCE = 0.1
CPU_CORES_PER_POD = 0.1           # matches "cpu: 100m" request in the manifests

# ---------------------------------------------------------------------------
# Action space (discrete replica delta, per service)
# ---------------------------------------------------------------------------
REPLICA_DELTA_OPTIONS = [-1, 0, 1]
MIN_REPLICAS = 1
MAX_REPLICAS = 8

# ---------------------------------------------------------------------------
# Graph / model dims.  NODE_FEATURE_DIM is per-node and topology-independent.
# ---------------------------------------------------------------------------
NODE_FEATURE_DIM = 5      # [latency, util, rps, replicas, is_entry]
GCN_HIDDEN_DIM = 32
GCN_EMBED_DIM = 16        # per branch (SDG + EHG) -> concatenated = 32
POLICY_HIDDEN = 64

# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------
PPO_LR = 3e-4
PPO_GAMMA = 0.95
PPO_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_ROLLOUT_EPISODES = 8
PPO_ITERATIONS = 120

# ---------------------------------------------------------------------------
# Safety guards (see safety/guards.py)
# ---------------------------------------------------------------------------
RATE_LIMIT_STEPS = 2              # min steps between rescales of the same service
SCALE_DOWN_COOLDOWN_STEPS = 4     # asymmetric: scaling DOWN is slower than up
CIRCUIT_BREAKER_VIOLATIONS = 5    # consecutive SLO breaches -> fall back to HPA rules
METRIC_STALENESS_SECONDS = 45     # refuse to act on data older than this

# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = _env_int("POLL_INTERVAL_SECONDS", 10)
JAEGER_LOOKBACK_SECONDS = 30
JAEGER_TRACE_LIMIT = 30

# ---------------------------------------------------------------------------
# HPA baseline (for the ablation study)
# ---------------------------------------------------------------------------
HPA_TARGET_UTILIZATION = 0.70
HPA_STABILIZATION_STEPS = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Absolute paths (anchored to the autoscaler/ dir) so scripts work from any cwd.
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "results"))
POLICY_PATH = os.path.join(MODELS_DIR, "shared_policy.pt")
DEFAULT_ARM = "rl_full"           # policy served when shared_policy.pt is absent
RESULTS_PATH = os.path.join(RESULTS_DIR, "ablation_results.json")
PARETO_PATH = os.path.join(RESULTS_DIR, "pareto_results.json")

# Discover the call graph from live Jaeger traces instead of CALL_GRAPH above.
AUTODISCOVER_TOPOLOGY = os.environ.get("AUTODISCOVER_TOPOLOGY", "0") == "1"
