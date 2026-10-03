"""
pareto.py

Multi-objective trade-off analysis. The reward is a scalarisation
(W_LATENCY*slo + W_COST*cost + W_VARIANCE*variance), so a single trained
policy is ONE point on the latency/cost trade-off curve. Sweeping the cost
weight and re-training traces out the frontier, and HPA / static baselines can
then be placed against it: a policy is only "better" if it is not dominated.

    python -m src.evaluation.pareto --iterations 55 --episodes 20

Outputs results/pareto_results.json (served by the dashboard at /api/pareto).
Objectives minimised: p95 latency (ms), mean cost (cores), SLO-violation rate.
"""

import argparse
import json
import os

import numpy as np

from src import config
from src.baselines.rule_based import HPABaseline, StaticBaseline
from src.evaluation.evaluate import make_rl_decider, run_episode
from src.evaluation.train_arm import save_policy, train_policy
from src.simulation.simulator import fixed_topology

DEFAULT_WEIGHTS = [0.05, 0.15, 0.30, 0.55, 0.90, 1.50]
PATTERNS = ["burst", "diurnal", "rampup", "normal"]
OBJECTIVES = ("p95_latency", "mean_cost", "slo_violation_rate")


def dominates(a: dict, b: dict) -> bool:
    """a dominates b if a is no worse on every objective and better on one."""
    no_worse = all(a[k] <= b[k] + 1e-9 for k in OBJECTIVES)
    better = any(a[k] < b[k] - 1e-9 for k in OBJECTIVES)
    return no_worse and better


def pareto_front(points: list[dict]) -> list[bool]:
    """Returns a parallel list: True where the point is non-dominated."""
    return [not any(dominates(q, p) for j, q in enumerate(points) if j != i)
            for i, p in enumerate(points)]


def _evaluate(make_decider, episodes):
    services, entry, graph = fixed_topology()
    eps = [run_episode(make_decider(), seed=1000 + i, pattern=PATTERNS[i % len(PATTERNS)],
                       services=services, entry=entry, call_graph=graph)
           for i in range(episodes)]
    return {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}


def sweep(weights, iterations, episodes, seed=0, save_models=True, verbose=True):
    services, entry, graph = fixed_topology()
    points = []

    original = config.W_COST
    try:
        for w in weights:
            config.W_COST = w            # compute_reward() reads this at call time
            if verbose:
                print(f"\n=== W_COST = {w} ===")
            policy, history, flags = train_policy("rl_full", iterations, seed, verbose=False)
            policy.eval()
            if save_models:
                save_policy(policy, history, flags,
                            os.path.join(config.MODELS_DIR, "pareto", f"w_cost_{w:.2f}.pt"),
                            extra={"w_cost": w})
            r = _evaluate(lambda p=policy: make_rl_decider(p, services, entry, graph), episodes)
            r.update(name=f"rl_full@w_cost={w}", kind="rl", w_cost=w)
            points.append(r)
            if verbose:
                print(f"  p95={r['p95_latency']:.1f}ms  cost={r['mean_cost']:.2f}  "
                      f"viol={r['slo_violation_rate']*100:.1f}%")
    finally:
        config.W_COST = original

    for name, make in (("hpa", lambda: HPABaseline(services).decide),
                       ("static", lambda: StaticBaseline(services).decide)):
        r = _evaluate(make, episodes)
        r.update(name=name, kind="baseline", w_cost=None)
        points.append(r)

    for p, nd in zip(points, pareto_front(points)):
        p["pareto_optimal"] = nd
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="*", default=DEFAULT_WEIGHTS)
    ap.add_argument("--iterations", type=int, default=55)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=config.PARETO_PATH)
    args = ap.parse_args()

    points = sweep(args.weights, args.iterations, args.episodes, args.seed)

    print("\n" + "=" * 86)
    print(f"{'point':<24}{'p95 ms':>9}{'cost':>8}{'SLO viol':>11}{'pareto':>9}")
    print("=" * 86)
    for p in sorted(points, key=lambda x: x["mean_cost"]):
        print(f"{p['name']:<24}{p['p95_latency']:9.1f}{p['mean_cost']:8.2f}"
              f"{p['slo_violation_rate']*100:10.1f}%{'  *' if p['pareto_optimal'] else '':>9}")
    print("(* = non-dominated on p95 latency, cost and SLO-violation rate)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"iterations": args.iterations, "episodes": args.episodes,
                   "points": points}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
