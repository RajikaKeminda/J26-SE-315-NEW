"""
env.py

Environment that exposes GRAPH-STRUCTURED state (X, A_sdg, A_ehg, globals)
rather than a flat vector, because the shared policy consumes graphs
directly. Not a gym.Env subclass - the custom PPO loop talks to it directly,
which keeps the graph tensors intact instead of forcing them through a
flat Box space.
"""

import numpy as np
import torch

from src import config
from src.simulation.simulator import (
    Simulator, random_topology, fixed_topology, generate_traffic, critical_path)
from src.topology_intel.graph_builder import build_all


def compute_reward(services, metrics, crit_latency, single_objective=False):
    """
    Scalarized multi-objective reward. `single_objective=True` gives the
    latency-only variant used as an ablation arm (mimics a conventional
    single-metric RL autoscaler).
    """
    slo_violation = max(0.0, crit_latency - config.SLO_LATENCY_MS) / config.SLO_LATENCY_MS

    total_cores = sum(metrics[s]["replicas"] for s in services) * config.CPU_CORES_PER_POD
    max_cores = len(services) * config.MAX_REPLICAS * config.CPU_CORES_PER_POD
    cost_norm = total_cores / max_cores

    utils = np.array([metrics[s]["utilization"] for s in services])
    load_var = float(np.var(utils))
    var_norm = min(load_var, 1.0)

    if single_objective:
        penalty = slo_violation
    else:
        penalty = (config.W_LATENCY * slo_violation
                   + config.W_COST * cost_norm
                   + config.W_VARIANCE * var_norm)

    return -penalty, {
        "slo_violation": slo_violation,
        "cost_norm": cost_norm,
        "total_cores": total_cores,
        "load_variance": load_var,
        "crit_latency_ms": crit_latency,
    }


class AutoscalerEnv:
    def __init__(self, use_random_topology=True, seed=None, single_objective=False):
        self.rng = np.random.default_rng(seed)
        self.use_random_topology = use_random_topology
        self.single_objective = single_objective
        self.reset()

    def reset(self, pattern="random"):
        if self.use_random_topology:
            services, entry, graph = random_topology(self.rng)
        else:
            services, entry, graph = fixed_topology()

        self.services, self.entry, self.call_graph = services, entry, graph
        self.sim = Simulator(services, entry, graph)
        self.t = 0
        # scale traffic with topology size so bigger graphs aren't trivially easy
        base = config.BASE_RPS * (len(services) / 6.0)
        self.traffic = generate_traffic(pattern, config.EPISODE_LENGTH_STEPS, base, self.rng)
        metrics = self.sim.step(self.traffic[0])
        return self._observe(metrics)

    def _observe(self, metrics):
        crit_lat, crit_path = critical_path(
            self.call_graph, self.entry, {s: m["latency_ms"] for s, m in metrics.items()})

        X, A_sdg, A_ehg = build_all(
            self.services, self.call_graph, metrics, self.entry, config.MAX_REPLICAS)

        total_rep = sum(m["replicas"] for m in metrics.values())
        utils = np.array([m["utilization"] for m in metrics.values()])
        globals_vec = torch.tensor([
            crit_lat / config.SLO_LATENCY_MS,
            total_rep / (len(self.services) * config.MAX_REPLICAS),
            min(float(np.var(utils)), 1.0),
        ], dtype=torch.float32)

        self._last = (metrics, crit_lat, crit_path)
        return X, A_sdg, A_ehg, globals_vec

    def step(self, actions):
        """actions: tensor/array of length N, values indexing REPLICA_DELTA_OPTIONS."""
        for i, svc in enumerate(self.services):
            self.sim.apply(svc, config.REPLICA_DELTA_OPTIONS[int(actions[i])])

        self.t += 1
        done = self.t >= config.EPISODE_LENGTH_STEPS - 1
        metrics = self.sim.step(self.traffic[self.t])
        obs = self._observe(metrics)
        _, crit_lat, crit_path = self._last
        reward, info = compute_reward(self.services, metrics, crit_lat, self.single_objective)
        info["critical_path"] = crit_path
        info["metrics"] = metrics
        return obs, reward, done, info
