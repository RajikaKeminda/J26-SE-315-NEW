"""
evaluate.py

Loads every trained arm and evaluates it against IDENTICAL topologies and
IDENTICAL traffic (matched seeds), alongside the HPA and static baselines.
Identical conditions are what make the comparison attributable to the
method rather than to luck.
"""

import argparse
import json
import os

import numpy as np
import torch

from src import config
from src.rl_agent.shared_policy import SharedPolicy
from src.baselines.rule_based import HPABaseline, StaticBaseline
from src.simulation.simulator import Simulator, fixed_topology, generate_traffic, critical_path
from src.topology_intel.graph_builder import build_all
from src.evaluation.train_arm import ARMS


def run_episode(decide_fn, seed, pattern, services, entry, call_graph):
    rng = np.random.default_rng(seed)
    sim = Simulator(services, entry, call_graph)
    base = config.BASE_RPS * (len(services) / 6.0)
    traffic = generate_traffic(pattern, config.EPISODE_LENGTH_STEPS, base, rng)

    latencies, costs, violations = [], [], []
    metrics = sim.step(traffic[0])

    for t in range(1, config.EPISODE_LENGTH_STEPS):
        for svc, d in decide_fn(metrics).items():
            sim.apply(svc, d)
        metrics = sim.step(traffic[t])
        crit_lat, _ = critical_path(
            call_graph, entry, {s: m["latency_ms"] for s, m in metrics.items()})
        latencies.append(crit_lat)
        costs.append(sum(m["replicas"] for m in metrics.values()) * config.CPU_CORES_PER_POD)
        violations.append(1.0 if crit_lat > config.SLO_LATENCY_MS else 0.0)

    return {
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "p99_latency": float(np.percentile(latencies, 99)),
        "mean_cost": float(np.mean(costs)),
        "slo_violation_rate": float(np.mean(violations)),
    }


def load_arm(arm):
    path = os.path.join(config.MODELS_DIR, f"{arm}.pt")
    return load_policy_file(path)


def load_policy_file(path):
    """Loads a checkpoint written by train_arm.save_policy; None if missing."""
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    policy = SharedPolicy(
        config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM, config.GCN_EMBED_DIM,
        config.POLICY_HIDDEN,
        use_ehg=ckpt.get("use_ehg", True), use_topology=ckpt.get("use_topology", True))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def make_rl_decider(policy, services, entry, call_graph):
    def decide(metrics):
        X, A_sdg, A_ehg = build_all(services, call_graph, metrics, entry, config.MAX_REPLICAS)
        crit_lat, _ = critical_path(
            call_graph, entry, {s: m["latency_ms"] for s, m in metrics.items()})
        total_rep = sum(m["replicas"] for m in metrics.values())
        utils = np.array([m["utilization"] for m in metrics.values()])
        g = torch.tensor([crit_lat / config.SLO_LATENCY_MS,
                          total_rep / (len(services) * config.MAX_REPLICAS),
                          min(float(np.var(utils)), 1.0)], dtype=torch.float32)
        with torch.no_grad():
            actions, _, _ = policy.act(X, A_sdg, A_ehg, g, deterministic=True)
        return {s: config.REPLICA_DELTA_OPTIONS[int(actions[i])] for i, s in enumerate(services)}
    return decide


def main(n_eval_episodes=20, save=True, out_path=None):
    services, entry, call_graph = fixed_topology()
    patterns = ["burst", "diurnal", "rampup", "normal"]

    deciders = {
        "static": lambda: StaticBaseline(services).decide,
        "hpa": lambda: HPABaseline(services).decide,
    }
    for arm in ARMS:
        policy = load_arm(arm)
        if policy is None:
            print(f"(skipping {arm} - {config.MODELS_DIR}/{arm}.pt not found)")
            continue
        deciders[arm] = (lambda p=policy: make_rl_decider(p, services, entry, call_graph))

    results = {}
    for name, make_decider in deciders.items():
        eps = []
        for i in range(n_eval_episodes):
            eps.append(run_episode(make_decider(), seed=1000 + i,
                                   pattern=patterns[i % len(patterns)],
                                   services=services, entry=entry, call_graph=call_graph))
        results[name] = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
        results[name]["p95_latency_std"] = float(np.std([e["p95_latency"] for e in eps]))

    # composite score: same weighting as the training reward, for a single
    # comparable number across arms
    for r in results.values():
        slo = max(0.0, r["p95_latency"] - config.SLO_LATENCY_MS) / config.SLO_LATENCY_MS
        cost_norm = r["mean_cost"] / (len(services) * config.MAX_REPLICAS * config.CPU_CORES_PER_POD)
        r["composite_penalty"] = config.W_LATENCY * slo + config.W_COST * cost_norm

    print("\n" + "=" * 96)
    print(f"{'arm':<18}{'p50 ms':>9}{'p95 ms':>10}{'p99 ms':>10}"
          f"{'SLO viol':>11}{'cost':>9}{'composite':>12}")
    print("=" * 96)
    for name, r in results.items():
        print(f"{name:<18}{r['p50_latency']:9.1f}{r['p95_latency']:10.1f}{r['p99_latency']:10.1f}"
              f"{r['slo_violation_rate']*100:10.1f}%{r['mean_cost']:9.2f}{r['composite_penalty']:12.3f}")
    print("=" * 96)
    print("(lower is better everywhere; composite = weighted latency+cost penalty)")

    if "hpa" in results and "rl_full" in results:
        h, f = results["hpa"], results["rl_full"]
        print(f"\nFull system vs HPA baseline:")
        print(f"  p95 latency     {h['p95_latency']:7.1f}ms -> {f['p95_latency']:7.1f}ms "
              f"({(f['p95_latency']-h['p95_latency'])/max(h['p95_latency'],1e-9)*100:+.1f}%)")
        print(f"  SLO violations  {h['slo_violation_rate']*100:7.1f}%  -> {f['slo_violation_rate']*100:6.1f}%")
        print(f"  cost (cores)    {h['mean_cost']:7.2f}   -> {f['mean_cost']:7.2f} "
              f"({(f['mean_cost']-h['mean_cost'])/max(h['mean_cost'],1e-9)*100:+.1f}%)")

    if save:
        out_path = out_path or config.RESULTS_PATH
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nSaved -> {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None, help="results JSON path (default results/ablation_results.json)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    main(n_eval_episodes=args.episodes, save=not args.no_save, out_path=args.out)
