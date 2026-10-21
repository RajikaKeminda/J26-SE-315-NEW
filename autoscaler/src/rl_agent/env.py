"""
env.py

The reinforcement-learning environment: a simulated cluster (calibrated against a
real one) behind the SAME safety guard the live system uses, exposing
GRAPH-STRUCTURED state and a vector-valued reward.

    obs   = dict(X (N,F) node features, A_sdg (N,N), A_ehg (N,N), g (G,) cluster scalars)
    action= per service: replica-delta index (7 options: -3..+3) and two continuous
            numbers in [-1, 1] that scale the pod's CPU and memory allocation
    reward= -(slo, cost, balance) penalty vector   (see rewards.py)

Not a gym.Env: the agent consumes graphs of varying size, which a flat Box space
cannot carry without padding to a fixed maximum - the limitation the shared policy
exists to remove.

The environment can be driven two ways:
    step(action_arrays)        - what the RL agent produces
    step_proposals(proposals)  - what baselines (HPA/VPA/threshold) produce
Both run the same sim -> guard -> reward pipeline, so every controller is compared
under identical physics and identical constraints.
"""

import numpy as np

from src import config
from src.rl_agent import rewards
from src.safety.guards import SafetyGuard, quantize
from src.simulation import service_model as sm
from src.simulation.simulator import (
    Simulator, budget_for, generate_traffic, random_topology)
from src.topologies import DEMO6, Topology
from src.topology_intel.critical_path import path_distribution
from src.topology_intel.graph_builder import build_observation


def u_to_resource(u, lo, hi):
    """u in [-1, 1] -> a resource TARGET on a log scale between lo and hi."""
    return float(lo * (hi / lo) ** ((float(np.clip(u, -1, 1)) + 1.0) / 2.0))


def resource_to_u(value, lo, hi):
    return float(np.clip(2.0 * np.log(max(value, lo) / lo) / np.log(hi / lo) - 1.0, -1.0, 1.0))


def decode_action(services, metrics, idx, cont, replica_only=False):
    """Network outputs -> absolute proposals {svc: {"dr","cpu","mem"}}."""
    proposals = {}
    for i, s in enumerate(services):
        m = metrics[s]
        dr = config.REPLICA_DELTA_OPTIONS[int(idx[i])]
        cpu, mem = m["cpu"], m["mem"]
        if not replica_only:
            cpu = quantize(u_to_resource(cont[i][0], config.CPU_MIN, config.CPU_MAX),
                           config.CPU_QUANTUM, config.CPU_MIN, config.CPU_MAX)
            mem = quantize(u_to_resource(cont[i][1], config.MEM_MIN_MB, config.MEM_MAX_MB),
                           config.MEM_QUANTUM_MB, config.MEM_MIN_MB, config.MEM_MAX_MB)
        proposals[s] = {"dr": dr, "cpu": cpu, "mem": mem}
    return proposals


def encode_executed(services, metrics_before, approved):
    """The action that was ACTUALLY executed (after the guard), back in the policy's own
    action space - this, not the raw proposal, is what the critic must learn from."""
    idx = np.zeros(len(services), dtype=np.int64)
    cont = np.zeros((len(services), 3), dtype=np.float32)          # [cpu u, memory u, replica delta / 3]
    for i, s in enumerate(services):
        a = approved[s]
        idx[i] = int(np.clip(a["dr"], config.REPLICA_DELTA_OPTIONS[0],
                             config.REPLICA_DELTA_OPTIONS[-1])) - config.REPLICA_DELTA_OPTIONS[0]
        cont[i, 0] = resource_to_u(a["cpu"], config.CPU_MIN, config.CPU_MAX)
        cont[i, 1] = resource_to_u(a["mem"], config.MEM_MIN_MB, config.MEM_MAX_MB)
        cont[i, 2] = config.REPLICA_DELTA_OPTIONS[idx[i]] / config.REPLICA_DELTA_OPTIONS[-1]
    return idx, cont


def topology_base_rps(topology: Topology) -> float:
    """Base external load, scaled so every topology presents the same relative
    pressure on its own budget (identical to config.BASE_RPS for the 6-service app)."""
    return (config.BASE_RPS * (budget_for(topology) / budget_for(DEMO6))
            * (_cpu_per_request(DEMO6) / _cpu_per_request(topology)))


def _cpu_per_request(t: Topology) -> float:
    """CPU-seconds burnt across the whole app per external request."""
    visits = {s: 0.0 for s in t.services}
    visits[t.entry] = 1.0
    indeg = {s: 0 for s in t.services}
    for targets in t.call_graph.values():
        for c, _ in targets:
            indeg[c] += 1
    queue = [s for s, d in indeg.items() if d == 0]
    while queue:
        n = queue.pop(0)
        for c, w in t.call_graph.get(n, []):
            visits[c] += visits[n] * w
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    return sum(visits[s] * sm.cpu_ms_per_request(t.work(s)) for s in t.services) / 1000.0


class AutoscalerEnv:
    def __init__(self, topology: Topology | None = None, seed=None, obs_noise=0.05,
                 use_guard=True, replica_only=False, episode_length=None):
        self.rng = np.random.default_rng(seed)
        self.fixed_topology = topology            # None -> a random topology every episode
        self.topology_sizes = (4, 12)             # range of the random topologies (training can widen it)
        self.randomize_initial = False             # training: start each service at a different CPU limit (heterogeneous cluster)
        self.obs_noise = obs_noise
        self.use_guard = use_guard
        self.replica_only = replica_only
        self.episode_length = episode_length or config.EPISODE_LENGTH_STEPS
        self.t = 0
        # No implicit reset here: it would consume the RNG, so that "seed s" would no longer mean
        # "traffic = generate_traffic(..., default_rng(s))" - which the live experiment relies on to
        # give a real run and its simulated twin identical traffic. Call reset() to start an episode.

    # ------------------------------------------------------------------
    def reset(self, pattern="random", topology: Topology | None = None, traffic=None, initial=None):
        topo = topology or self.fixed_topology or random_topology(self.rng, *self.topology_sizes)
        self.topology = topo
        self.services, self.entry, self.call_graph = list(topo.services), topo.entry, topo.call_graph
        self.sim = Simulator(topo, rng=self.rng, obs_noise=self.obs_noise)
        if initial:
            self.sim.reset(**initial)
        elif self.randomize_initial and self.rng.random() < 0.7:
            # Real clusters are not uniform (one service has a tight limit, another a generous one): a policy that has only
            # ever seen identical limits cannot be expected to cope. Levels span the allowed CPU range.
            levels = np.array(sorted({round(config.CPU_MAX * f / config.CPU_QUANTUM) * config.CPU_QUANTUM
                                      for f in (0.3, 0.5, 0.75, 1.0)}))
            levels = np.clip(levels, config.CPU_MIN, config.CPU_MAX)
            self.sim.reset(cpu={s2: float(self.rng.choice(levels)) for s2 in self.services})
        self.slo_ms, self.budget = self.sim.slo_ms, self.sim.budget
        self.guard = SafetyGuard(services=self.services, budget=self.budget)
        self.t = 0
        base = topology_base_rps(topo)
        self.traffic = (np.asarray(traffic, dtype=float) if traffic is not None
                        else generate_traffic(pattern, self.episode_length + 1, base, self.rng))
        self._metrics = self.sim.step(self.traffic[0])
        return self._observe()

    def observe(self):
        return self._observe_cached

    def _observe(self):
        e2e = self._metrics["__e2e__"]
        ready = self.guard.readiness(self.t)
        metrics = {s: dict(self._metrics[s], resize_cooldown=ready[s][0], scaledown_cooldown=ready[s][1])
                   for s in self.services}
        cost = self.sim.cost()
        override = None
        if config.CRITICAL_PATH_SOURCE == "crisp":        # opt-in: the measured-critical-share inputs (see topology_intel/crisp.py)
            from src.topology_intel import crisp
            if getattr(self, "_chains_for", None) is not self.topology:
                self._chains, self._chains_for = path_distribution(self.call_graph, self.entry, max_paths=256), self.topology
            prof = crisp.expected_critical_profile(self._chains, {s: metrics[s]["latency_ms"] for s in self.services})
            override = {"set": prof["set"], "shares": prof["shares"]}
        obs = build_observation(self.services, self.call_graph, self.entry, metrics, e2e,
                                cost, self.budget, self.slo_ms, critical_override=override)
        obs.update(metrics=metrics, e2e=e2e, cost=cost, external_rps=float(self.traffic[min(self.t, len(self.traffic) - 1)]))
        self._observe_cached = obs
        return obs

    # ------------------------------------------------------------------
    def step(self, action, physical_only=False):
        """action = (idx (N,), cont (N,2)) from the agent."""
        idx, cont = action
        proposals = decode_action(self.services, self._observe_cached["metrics"], idx, cont,
                                  self.replica_only)
        return self.step_proposals(proposals, physical_only=physical_only)

    def step_proposals(self, proposals, physical_only=False):
        metrics = self._observe_cached["metrics"]
        current = {s: metrics[s] for s in self.services}
        if self.use_guard or physical_only:
            approved, reasons = self.guard.filter(proposals, current, self.t,
                                                  physical_only=physical_only or not self.use_guard)
        else:
            approved, reasons = proposals, {}

        executed = encode_executed(self.services, current, approved)
        for s, a in approved.items():
            self.sim.apply(s, a["dr"], a["cpu"], a["mem"])

        self.t += 1
        self._metrics = self.sim.step(self.traffic[min(self.t, len(self.traffic) - 1)])
        obs = self._observe()
        penalty_vec, pen = rewards.objective_penalties(
            obs["metrics"], self.services, obs["e2e"], obs["cost"], self.budget, self.slo_ms)
        self.guard.observe_slo(obs["e2e"]["p95_ms"], self.slo_ms, self.t)
        done = self.t >= self.episode_length
        info = {
            "e2e": obs["e2e"], "cost": obs["cost"], "penalties": pen,
            "proposed": proposals, "approved": approved, "reasons": reasons,
            "executed": executed,
            "critical_path": obs["critical_path"], "metrics": obs["metrics"],
            "slo_ms": self.slo_ms, "budget": self.budget, "breaker_open": self.guard.breaker_open,
            "total_replicas": sum(m["replicas"] for m in obs["metrics"].values()),
            "mean_util": float(np.mean([m["utilization"] for m in obs["metrics"].values()])),
        }
        return obs, -penalty_vec, done, info
