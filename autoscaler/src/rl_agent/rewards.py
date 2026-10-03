"""
rewards.py

The multi-objective reward. Three objectives, all expressed as PENALTIES in
roughly [0, 1] so they are comparable without per-run tuning (the proposal's
"normalised multi-objective reward"):

    slo      fixed step for any p95 breach + graded excess + request errors
    cost     provisioned CPU+memory as a fraction of the cluster budget
    balance  variance of per-service CPU utilisation (load distribution)

The reward is the NEGATED penalty VECTOR. It is deliberately not collapsed to a
scalar here: MO-TD3 learns one Q-value per objective and a preference vector w
decides the trade-off, both while training and at deployment time.
"""

import numpy as np

from src import config

N_OBJ = config.N_OBJECTIVES


def objective_penalties(metrics, services, e2e, cost, budget, slo_ms):
    """-> (penalty vector (3,), info dict). Lower is better on every component."""
    p95 = e2e["p95_ms"]
    excess = max(0.0, p95 / slo_ms - 1.0)
    slo = 0.0
    if p95 > slo_ms:
        slo = config.SLO_VIOLATION_STEP + (1.0 - config.SLO_VIOLATION_STEP) * min(
            1.0, excess / config.SLO_SATURATION)
    slo = min(1.0, slo + e2e.get("error_rate", 0.0))

    cost_norm = cost / budget

    utils = np.array([metrics[s]["utilization"] for s in services], dtype=float)
    load_var = float(np.var(utils))
    balance = min(1.0, 4.0 * load_var)       # var of values in [0,1] is <= 0.25

    vec = np.array([slo, cost_norm, balance], dtype=np.float32)
    return vec, {
        "slo_penalty": slo, "cost_norm": cost_norm, "balance_penalty": balance,
        "load_variance": load_var, "slo_violated": p95 > slo_ms,
    }


def scalarize(reward_vec, w):
    """Weighted sum. `reward_vec` is the (negated) penalty vector; w sums to 1."""
    return float(np.dot(np.asarray(reward_vec, dtype=np.float64), np.asarray(w, dtype=np.float64)))


def normalize_preference(w):
    w = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    total = w.sum()
    if total <= 0:
        return np.ones(N_OBJ) / N_OBJ
    return w / total


def preference_grid(step=0.25):
    """Evenly spaced points on the 3-objective simplex (corners, edges, interior)."""
    k = int(round(1.0 / step))
    pts = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            pts.append(np.array([i, j, k - i - j], dtype=np.float64) / k)
    return pts
