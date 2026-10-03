"""
metrics.py

Episode runner and metrics shared by training-time validation, the ablation study,
the Pareto sweep and the dashboard. Every controller - learned or rule-based - is run
through the SAME environment (same physics, same constraints) on the SAME traffic and
the SAME measurement-noise stream (matched seeds), so differences between rows are
attributable to the controller, not to luck.

Metrics (per the proposal's evaluation plan):
  p50 / p95 / p99 latency   end-to-end, mean over the episode's control windows
  slo_violation_rate        fraction of windows whose p95 exceeded the SLO
  mean_cost                 provisioned cores-equivalent (CPU + priced memory)
  mean_util / load_variance CPU utilisation and its spread across services
  adaptation_time_s         time from burst onset until the SLO is met again
  objectives (3)            the reward-vector components (slo, cost, balance)
"""

import numpy as np

from src import config
from src.baselines.rule_based import BASELINES
from src.rl_agent.env import AutoscalerEnv, topology_base_rps

PATTERNS = ["normal", "burst", "diurnal", "rampup"]


def adaptation_time(traffic, base, p95s, slo_ms, window=2):
    """Seconds from load onset (traffic > 1.5x base) until p95 <= SLO for `window`
    consecutive windows. 0 if the SLO was never broken after onset; the remaining
    episode length if it never recovered."""
    onset = next((i for i, v in enumerate(traffic) if v > 1.5 * base), None)
    if onset is None:
        return 0.0
    first_viol = next((i for i in range(onset, len(p95s)) if p95s[i] > slo_ms), None)
    if first_viol is None:
        return 0.0
    for i in range(first_viol, len(p95s) - window + 1):
        if all(p95s[j] <= slo_ms for j in range(i, i + window)):
            return (i - first_viol) * config.STEP_SECONDS
    return (len(p95s) - first_viol) * config.STEP_SECONDS


def run_episode(controller, topology, seed, pattern, w=config.DEFAULT_PREFERENCE,
                obs_noise=0.05, physical_only=None, record=False):
    """Runs one episode, returns a metrics dict (and the per-step trace if record)."""
    if physical_only is None:
        physical_only = controller.name in BASELINES
    env = AutoscalerEnv(topology=topology, seed=seed, obs_noise=obs_noise)
    obs = env.reset(pattern, initial=getattr(controller, "initial", None))
    controller.reset()
    base = topology_base_rps(topology)

    p50, p95, p99, errs, costs, utils, lvar, viol = [], [], [], [], [], [], [], []
    reps, rvecs, trace = [], [], []
    for t in range(env.episode_length):
        proposals = controller.decide(obs, breaker_open=env.guard.breaker_open)
        obs, r, done, info = env.step_proposals(proposals, physical_only=physical_only)
        e = info["e2e"]
        p50.append(e["p50_ms"]); p95.append(e["p95_ms"]); p99.append(e["p99_ms"])
        errs.append(e["error_rate"]); costs.append(info["cost"]); utils.append(info["mean_util"])
        lvar.append(info["penalties"]["load_variance"]); viol.append(float(info["penalties"]["slo_violated"]))
        reps.append(info["total_replicas"]); rvecs.append(r)
        if record:
            trace.append({"t": t, "rps": float(env.traffic[env.t]), "p95_ms": e["p95_ms"], "cost": info["cost"],
                          "replicas": info["total_replicas"], "violated": bool(viol[-1])})
    rvecs = np.array(rvecs)
    out = {
        "p50_latency": float(np.mean(p50)), "p95_latency": float(np.mean(p95)),
        "p99_latency": float(np.mean(p99)), "p95_latency_max": float(np.max(p95)),
        "slo_violation_rate": float(np.mean(viol)), "error_rate": float(np.mean(errs)),
        "mean_cost": float(np.mean(costs)), "mean_util": float(np.mean(utils)),
        "load_variance": float(np.mean(lvar)), "mean_replicas": float(np.mean(reps)),
        "adaptation_time_s": adaptation_time(env.traffic, base, p95, env.slo_ms),
        "obj_slo": float(-rvecs[:, 0].mean()), "obj_cost": float(-rvecs[:, 1].mean()),
        "obj_balance": float(-rvecs[:, 2].mean()),
        "scalarized": float(-(rvecs.mean(0) * np.asarray(w)).sum()),
        "slo_ms": env.slo_ms,
    }
    if record:
        out["trace"] = trace
    return out


def summarize(episodes):
    keys = [k for k in episodes[0] if k not in ("trace", "slo_ms")]
    s = {k: float(np.mean([e[k] for e in episodes])) for k in keys}
    s["p95_latency_std"] = float(np.std([e["p95_latency"] for e in episodes]))
    s["n_episodes"] = len(episodes)
    s["slo_ms"] = float(episodes[0]["slo_ms"])
    return s


def evaluate_controller(make_controller, topology, n_episodes=20, patterns=PATTERNS,
                        w=config.DEFAULT_PREFERENCE, seed0=1000, obs_noise=0.05):
    """Mean metrics over `n_episodes` matched-seed episodes cycling through `patterns`;
    also returns a per-pattern breakdown."""
    eps, by_pattern = [], {p: [] for p in patterns}
    for i in range(n_episodes):
        pat = patterns[i % len(patterns)]
        e = run_episode(make_controller(), topology, seed0 + i, pat, w=w, obs_noise=obs_noise)
        eps.append(e)
        by_pattern[pat].append(e)
    out = summarize(eps)
    out["by_pattern"] = {p: summarize(v) for p, v in by_pattern.items() if v}
    return out
