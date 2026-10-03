"""
pareto.py

Multi-objective trade-off analysis - the proposal's "Pareto-optimal trade-offs" between
latency SLO adherence, infrastructure cost and load balance.

The MO-TD3 agent is conditioned on a preference vector w. ONE trained network therefore
traces a whole trade-off curve: sweep w over the simplex at INFERENCE time and evaluate
each point on identical traffic. (The earlier PPO version needed a full re-training per
reward weighting, ~1.5 min each, and still collapsed to one corner.)

The learned front is compared against the conventional alternative - tuning the
Kubernetes HPA, which also trades cost for latency but only through its CPU size and
utilisation target - and against VPA / threshold / static. A point is Pareto-optimal if
no other point (of ANY controller) is at least as good on all objectives and better on one.

    python -m src.evaluation.pareto --episodes 16

Objectives minimised: SLO-violation rate, mean p95 latency, mean cost. The scalar
quality number reported per family is the 2-D hypervolume in (SLO-violation rate,
cost / budget) against the reference point (1, 1): larger = a better front.
"""

import argparse
import json
import os

from src import config
from src.evaluation.evaluate import HPA_GRID_CPU, HPA_GRID_TARGET, load_arm
from src.evaluation.metrics import evaluate_controller
from src.rl_agent import rewards
from src.rl_agent.controllers import RLController, RuleController
from src.topologies import DEMO6

OBJECTIVES = ("slo_violation_rate", "p95_latency", "mean_cost")


def dominates(a: dict, b: dict) -> bool:
    """a dominates b if a is no worse on every objective and better on one."""
    no_worse = all(a[k] <= b[k] + 1e-9 for k in OBJECTIVES)
    better = any(a[k] < b[k] - 1e-9 for k in OBJECTIVES)
    return no_worse and better


def pareto_front(points: list[dict]) -> list[bool]:
    """Parallel list: True where the point is non-dominated."""
    return [not any(dominates(q, p) for j, q in enumerate(points) if j != i)
            for i, p in enumerate(points)]


def hypervolume_2d(points, budget, ref=(1.0, 1.0)):
    """Area dominated in (violation rate, cost/budget), up to `ref`. Larger is better."""
    pts = sorted({(min(p["slo_violation_rate"], ref[0]), min(p["mean_cost"] / budget, ref[1]))
                  for p in points})
    hv, best_cost = 0.0, ref[1]
    for x, y in pts:
        if y < best_cost:
            hv += (ref[0] - x) * (best_cost - y)
            best_cost = y
    return hv


def _point(name, kind, family, r, **extra):
    return {"name": name, "kind": kind, "family": family, **extra,
            **{k: r[k] for k in ("p95_latency", "p99_latency", "mean_cost", "slo_violation_rate",
                                 "mean_util", "load_variance", "obj_slo", "obj_cost", "obj_balance")}}


def sweep(agent, topology, episodes, grid_step=0.25, include_baselines=True, arm="mo_td3_full", log=print):
    points = []
    for w in rewards.preference_grid(grid_step):
        r = evaluate_controller(lambda w=w: RLController(agent, topology.services, w=w, name=arm),
                                topology, n_episodes=episodes)
        points.append(_point(f"{arm}@w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})", "rl", "mo_td3", r,
                             preference=[float(x) for x in w]))
        log(f"  w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})  p95={r['p95_latency']:7.1f}ms  "
            f"cost={r['mean_cost']:.2f}  viol={r['slo_violation_rate'] * 100:5.1f}%")
    if include_baselines:
        S = topology.services
        for name in ("static", "threshold", "vpa"):
            r = evaluate_controller(lambda n=name: RuleController(n, S), topology, n_episodes=episodes)
            points.append(_point(name, "baseline", name, r))
        for cpu in HPA_GRID_CPU:
            for target in HPA_GRID_TARGET:
                r = evaluate_controller(
                    lambda c=cpu, t=target: RuleController("hpa", S, initial={"cpu": c}, target_utilization=t),
                    topology, n_episodes=episodes)
                points.append(_point(f"hpa(cpu={cpu},target={target})", "baseline", "hpa", r,
                                     hpa_cpu=cpu, hpa_target=target))
    for p, nd in zip(points, pareto_front(points)):
        p["pareto_optimal"] = nd
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mo_td3_full")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--grid-step", type=float, default=0.25)
    ap.add_argument("--out", default=config.PARETO_PATH)
    args = ap.parse_args()

    agent = load_arm(args.arm)
    if agent is None:
        raise SystemExit(f"no checkpoint for {args.arm}: run `python -m src.evaluation.train_agent {args.arm}`")
    points = sweep(agent, DEMO6, args.episodes, args.grid_step, arm=args.arm)

    budget = config.BUDGET_PER_SERVICE * len(DEMO6.services)
    fam = {}
    for fname in ("mo_td3", "hpa"):
        sel = [p for p in points if p["family"] == fname]
        fam[fname] = {"hypervolume": hypervolume_2d(sel, budget), "n_points": len(sel),
                      "n_pareto_optimal": sum(p["pareto_optimal"] for p in sel)}

    print("\n" + "=" * 96)
    print(f"{'point':<46}{'p95 ms':>9}{'cost':>8}{'SLO viol':>11}{'pareto':>9}")
    print("=" * 96)
    for p in sorted(points, key=lambda x: x["mean_cost"]):
        print(f"{p['name']:<46}{p['p95_latency']:9.1f}{p['mean_cost']:8.2f}"
              f"{p['slo_violation_rate'] * 100:10.1f}%{'  *' if p['pareto_optimal'] else '':>9}")
    print("(* = non-dominated on SLO-violation rate, p95 latency and cost, over ALL controllers)")
    print(f"\nhypervolume (higher = better front):  MO-TD3 {fam['mo_td3']['hypervolume']:.3f}   "
          f"HPA tuning sweep {fam['hpa']['hypervolume']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"episodes": args.episodes, "arm": args.arm, "objectives": list(OBJECTIVES),
                   "families": fam, "points": points}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
