"""
evaluate.py

Head-to-head comparison of every controller on IDENTICAL traffic, identical simulator
physics and identical measurement-noise streams (matched seeds):

  baselines   static (no autoscaling), threshold rule, Kubernetes HPA (default 60 %),
              HPA tuned (best CPU size x utilisation target found on a held-out grid -
              the strongest conventional baseline, so the learned agent is not
              compared against a strawman), Kubernetes VPA
  learned     the full MO-TD3 agent and its ablation arms (no topology, SDG only,
              single objective, replica-only, no demonstrations)

on two topologies:
  demo6          the 6-service application deployed on the real cluster (and the one the
                 simulator was calibrated on)
  trainticket14  a 14-service application with Train-Ticket's call structure that the
                 policy NEVER saw during training - the generalisation test

Seeds: when models/seeds/<arm>_s<k>.pt exist, results are also reported per training seed
(mean +- std across seeds) - one RL run is an anecdote, several are evidence.

    python -m src.evaluation.evaluate --episodes 24
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from src import config
from src.evaluation.metrics import PATTERNS, evaluate_controller
from src.rl_agent.controllers import RLController, RuleController
from src.rl_agent.mo_td3 import MOTD3Agent
from src.rl_agent.trainer import ARMS
from src.topologies import DEMO6, TRAINTICKET14

HPA_GRID_CPU = (0.25, 0.5, 1.0)
HPA_GRID_TARGET = (0.3, 0.4, 0.5, 0.6, 0.7)
KEY_METRICS = ("p95_latency", "slo_violation_rate", "mean_cost", "scalarized", "adaptation_time_s")


def tune_hpa(topology=DEMO6, n_episodes=8, w=config.DEFAULT_PREFERENCE, seed0=10_000):
    """Grid-search the HPA's CPU sizing and utilisation target on a HELD-OUT seed range
    (disjoint from the evaluation seeds) by the same scalarised objective the agent uses.
    Returns (best_params, table)."""
    table = []
    for cpu in HPA_GRID_CPU:
        for target in HPA_GRID_TARGET:
            r = evaluate_controller(
                lambda: RuleController("hpa", topology.services, initial={"cpu": cpu},
                                       target_utilization=target),
                topology, n_episodes=n_episodes, w=w, seed0=seed0)
            table.append({"cpu": cpu, "target": target, **{k: r[k] for k in KEY_METRICS}})
    best = min(table, key=lambda t: t["scalarized"])
    return best, table


def baseline_factories(topology, hpa_best):
    S = topology.services
    return {
        "static":    lambda: RuleController("static", S),
        "threshold": lambda: RuleController("threshold", S),
        "hpa":       lambda: RuleController("hpa", S),
        "hpa_tuned": lambda: RuleController("hpa", S, label="hpa_tuned", initial={"cpu": hpa_best["cpu"]},
                                            target_utilization=hpa_best["target"]),
        "vpa":       lambda: RuleController("vpa", S),
    }


def load_arm(arm, seed=None):
    from src.evaluation.train_agent import checkpoint_path
    path = checkpoint_path(arm, 0 if seed is None else seed)
    if not os.path.exists(path):
        return None
    try:
        return MOTD3Agent.load(path)
    except Exception as e:
        print(f"(skipping {arm}: {e})")
        return None


def arm_preference(arm):
    fw = ARMS[arm]["fixed_w"]
    return np.asarray(fw if fw is not None else config.DEFAULT_PREFERENCE, dtype=np.float32)


def available_seeds(arm):
    seeds = []
    for p in glob.glob(os.path.join(config.MODELS_DIR, "seeds", f"{arm}_s*.pt")):
        m = re.search(rf"{arm}_s(\d+)\.pt$", p)
        if m:
            seeds.append(int(m.group(1)))
    if os.path.exists(os.path.join(config.MODELS_DIR, f"{arm}.pt")):
        seeds.append(0)
    return sorted(set(seeds))


def average_runs(runs):
    """Mean of several evaluation summaries (one per training seed) - metric by metric, including the
    per-pattern breakdown. A single run is returned unchanged."""
    if len(runs) == 1:
        return runs[0]
    out = {k: float(np.mean([r[k] for r in runs])) for k, v in runs[0].items()
           if isinstance(v, (int, float)) and k != "n_episodes"}
    out["n_episodes"] = runs[0]["n_episodes"]
    out["p95_latency_std"] = float(np.std([r["p95_latency"] for r in runs]))   # across seeds
    if "by_pattern" in runs[0]:
        out["by_pattern"] = {p: average_runs([r["by_pattern"][p] for r in runs]) for p in runs[0]["by_pattern"]}
    out["n_seeds"] = len(runs)
    return out


def evaluate_topology(topology, n_episodes, hpa_best, w=config.DEFAULT_PREFERENCE, with_seeds=True, log=print):
    results = {}
    for name, make in baseline_factories(topology, hpa_best).items():
        results[name] = evaluate_controller(make, topology, n_episodes=n_episodes, w=w)
    per_seed = {}
    for arm in ARMS:
        seeds = available_seeds(arm) if with_seeds else ([0] if load_arm(arm) else [])
        if not seeds:
            log(f"(skipping {arm} - no checkpoint in {config.MODELS_DIR})")
            continue
        runs = []
        for sd in seeds:
            agent = load_arm(arm, sd)
            if agent is None:
                continue
            ew = arm_preference(arm)
            runs.append(evaluate_controller(
                lambda a=agent, ew=ew: RLController(a, topology.services, w=ew, name=arm),
                topology, n_episodes=n_episodes, w=w))
        if not runs:
            continue
        # headline row = the MEAN over training seeds (never a cherry-picked seed)
        results[arm] = average_runs(runs)
        if len(runs) > 1:
            per_seed[arm] = {k: [r[k] for r in runs] for k in KEY_METRICS}
            per_seed[arm]["seeds"] = seeds[:len(runs)]
    return results, per_seed


HELDOUT_SEED0 = 777_000          # topology seeds disjoint from everything used in training / validation


def heldout_topologies(n=16):
    """Random topologies from the SAME generator the agent trained on, but never trained on (their
    seeds are disjoint). Tests structural generalisation without Train-Ticket's specific shape."""
    from src.simulation.simulator import random_topology
    return [random_topology(np.random.default_rng(HELDOUT_SEED0 + i)) for i in range(n)]


def evaluate_heldout(hpa_best, n_topologies=16, episodes_per_topology=4, w=config.DEFAULT_PREFERENCE, log=print):
    """Every controller on `n_topologies` unseen random topologies (one episode per pattern each);
    returns {controller: mean metrics over all topologies and seeds-of-training}."""
    topos = heldout_topologies(n_topologies)

    def over_topologies(make_for):
        runs = [evaluate_controller(make_for(t), t, n_episodes=episodes_per_topology, w=w, seed0=9_000 + 7 * i)
                for i, t in enumerate(topos)]
        return average_runs(runs)

    out = {}
    for name in ("static", "threshold", "hpa", "hpa_tuned", "vpa"):
        def make_for(t, name=name):
            if name == "hpa_tuned":
                return lambda: RuleController("hpa", t.services, label="hpa_tuned", initial={"cpu": hpa_best["cpu"]},
                                               target_utilization=hpa_best["target"])
            return lambda: RuleController(name, t.services)
        out[name] = over_topologies(make_for)
    for arm in ARMS:
        seeds = available_seeds(arm)
        runs = []
        for sd in seeds:
            agent = load_arm(arm, sd)
            if agent is None:
                continue
            ew = arm_preference(arm)
            runs.append(over_topologies(lambda t, a=agent, ew=ew, arm=arm: (
                lambda: RLController(a, t.services, w=ew, name=arm))))
        if runs:
            out[arm] = average_runs(runs)
    log(f"  evaluated on {len(topos)} held-out random topologies "
        f"({min(len(t.services) for t in topos)}-{max(len(t.services) for t in topos)} services)")
    return out


def print_table(title, results, per_seed=None):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    print(f"{'controller':<22}{'p50 ms':>8}{'p95 ms':>9}{'p99 ms':>9}{'SLO viol':>10}{'cost':>7}{'util':>6}"
          f"{'adapt s':>9}{'score':>8}")
    for name, r in results.items():
        print(f"{name:<22}{r['p50_latency']:8.1f}{r['p95_latency']:9.1f}{r['p99_latency']:9.1f}"
              f"{r['slo_violation_rate'] * 100:9.1f}%{r['mean_cost']:7.2f}{r['mean_util']:6.2f}"
              f"{r['adaptation_time_s']:9.0f}{r['scalarized']:8.3f}")
    if per_seed:
        print("\nacross training seeds (mean ± std; the table above shows the seed mean):")
        for arm, d in per_seed.items():
            print(f"  {arm:<22} p95 {np.mean(d['p95_latency']):7.1f} ± {np.std(d['p95_latency']):6.1f}   "
                  f"viol {np.mean(d['slo_violation_rate']) * 100:5.1f}% ± {np.std(d['slo_violation_rate']) * 100:4.1f}   "
                  f"cost {np.mean(d['mean_cost']):.2f} ± {np.std(d['mean_cost']):.2f}   "
                  f"score {np.mean(d['scalarized']):.3f} ± {np.std(d['scalarized']):.3f}   (n={len(d['seeds'])})")
    print("(lower is better everywhere; score = preference-weighted penalty, the quantity the agent minimises)")


def main(n_episodes=24, save=True, out_path=None, skip_unseen=False, tune_episodes=8):
    print("tuning the HPA baseline on held-out scenarios ...")
    hpa_best, hpa_table = tune_hpa(DEMO6, n_episodes=tune_episodes)
    print(f"  best HPA: cpu={hpa_best['cpu']} target={hpa_best['target']} "
          f"(score {hpa_best['scalarized']:.3f})")

    out = {"meta": {"episodes": n_episodes, "patterns": PATTERNS,
                    "preference": list(config.DEFAULT_PREFERENCE), "objectives": list(config.OBJECTIVES),
                    "hpa_tuned": hpa_best, "hpa_tuning_table": hpa_table}}
    res, per_seed = evaluate_topology(DEMO6, n_episodes, hpa_best)
    out["demo6"], out["demo6_seeds"] = res, per_seed
    print_table("demo6 - the 6-service application (deployed on the real cluster)", res, per_seed)

    if not skip_unseen:
        res2, per_seed2 = evaluate_topology(TRAINTICKET14, n_episodes, hpa_best, with_seeds=True)
        out["trainticket14"], out["trainticket14_seeds"] = res2, per_seed2
        print_table("trainticket14 - 14 services, Train-Ticket call structure, NEVER seen in training", res2, per_seed2)
        res3 = evaluate_heldout(hpa_best)
        out["heldout_random"] = res3
        print_table("16 held-out RANDOM topologies (same generator as training, disjoint seeds; 4-12 services)", res3)

    if save:
        out_path = out_path or config.RESULTS_PATH
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"\nSaved -> {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--out", default=None, help="results JSON path (default results/ablation_results.json)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--skip-unseen", action="store_true", help="skip the 14-service generalisation test")
    args = ap.parse_args()
    main(n_episodes=args.episodes, save=not args.no_save, out_path=args.out, skip_unseen=args.skip_unseen)
