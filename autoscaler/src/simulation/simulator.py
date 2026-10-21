"""
simulator.py

A fast cluster simulator whose per-service physics (capacity, queueing latency,
CFS throttling, tail inflation, memory) is FITTED to measurements from a real
cluster - see service_model.py and calibration.py - and whose temporal dynamics
mirror what the live control loop faces:

  * pods are not instantly useful: a scale-up takes POD_STARTUP_SECONDS to become
    Ready, so capacity arrives part-way through the step;
  * a CPU/memory resize is a rolling restart (ROLLING_SECONDS_PER_REPLICA per
    replica), so it takes effect late;
  * overload builds a backlog that outlives the burst that caused it, and requests
    that wait longer than the client timeout are dropped;
  * the end-to-end latency a client sees is a probability mixture over the call
    chains (each request is routed to ONE downstream by the traffic shares), and
    percentiles are taken from that mixture, not from a single "worst" chain.

Training uses RANDOM TOPOLOGIES (random_topology) so the shared policy learns a
general rule rather than memorising one graph.
"""

from dataclasses import dataclass, field

import numpy as np

from src import config
from src.simulation import service_model as sm
from src.topologies import Topology
from src.topology_intel.critical_path import (
    critical_path_dfs, path_distribution, reachable)


# ---------------------------------------------------------------------------
# Topology generation
# ---------------------------------------------------------------------------
def random_topology(rng, min_services=4, max_services=12, heterogeneous_work=True) -> Topology:
    """A random acyclic call graph with STRUCTURAL variety: 2-5 tiers, fan-out 1-4, and shared
    callees (a service with several callers), over 4-12 services.

    Why this matters: a GCN embedding is only as general as the graphs it was trained on. A generator
    that always produced entry -> mid -> leaf trees would teach "what a service next to a hub looks
    like" for ONE shape; the agent would then be applied to chains five services deep, or to a
    callee shared by three callers, that it has never seen. (The earlier 3-tier generator did exactly
    that, and the topology-aware agent generalised worse to unseen graphs than the topology-blind one.)

    Construction: place services in tiers; every non-entry service gets 1-2 parents from earlier tiers
    (preferring the previous tier), so edges only go forward (acyclic) and everything is reachable."""
    n = int(rng.integers(min_services, max_services + 1))
    services = [f"svc-{i}" for i in range(n)]
    entry = services[0]

    n_tiers = int(rng.integers(2, min(5, n) + 1))                    # entry tier + (n_tiers - 1) below it
    tier_of = {entry: 0}
    rest = services[1:]
    for t, s in enumerate(rest[:n_tiers - 1], start=1):               # every tier gets at least one service
        tier_of[s] = t
    for s in rest[n_tiers - 1:]:
        tier_of[s] = int(rng.integers(1, n_tiers))
    by_tier = {}
    for s, t in tier_of.items():
        by_tier.setdefault(t, []).append(s)

    graph = {s: {} for s in services}
    for s in rest:
        t = tier_of[s]
        above = [x for x in services if tier_of[x] < t]
        prev = by_tier.get(t - 1, [])
        n_parents = 1 if (len(above) == 1 or rng.random() < 0.65) else 2
        parents = []
        for _ in range(n_parents):
            pool = prev if (prev and rng.random() < 0.7) else above
            cand = [x for x in pool if x not in parents]
            if cand:
                parents.append(str(rng.choice(cand)))
        for par in parents:
            graph[par][s] = 0.0

    for s, targets in graph.items():                                  # random shares summing to 1 per caller
        if targets:
            w = rng.uniform(0.5, 1.0, size=len(targets))
            w /= w.sum()
            for t, val in zip(list(targets), w):
                targets[t] = float(val)

    work = ({s: float(rng.uniform(8.0, 26.0)) for s in services}
            if heterogeneous_work else {})
    return Topology(name=f"rand-{n}", services=services, entry=entry,
                    call_graph={k: [(c, w) for c, w in v.items()] for k, v in graph.items()},
                    work_ms=work)


def topology_depth(topology: Topology) -> int:
    """Number of services on the longest call chain (entry included)."""
    _, path = critical_path_dfs(topology.call_graph, topology.entry,
                                {s: 1.0 for s in topology.services})
    return len(path)


def slo_for(topology: Topology) -> float:
    return config.slo_for_depth(topology_depth(topology))


def budget_for(topology: Topology) -> float:
    """Cluster budget in core-equivalents."""
    return config.BUDGET_PER_SERVICE * len(topology.services)


def fixed_topology() -> Topology:
    return config.TOPOLOGY


# ---------------------------------------------------------------------------
# Traffic patterns
# ---------------------------------------------------------------------------
def generate_traffic(pattern, length, base_rps, rng):
    t = np.arange(length)
    if pattern == "random":
        pattern = rng.choice(["normal", "burst", "diurnal", "rampup"])

    if pattern == "normal":
        traffic = base_rps + rng.normal(0, base_rps * 0.05, length)
    elif pattern == "burst":
        traffic = np.full(length, base_rps, dtype=float)
        s = length // 3
        e = s + max(2, length // 4)
        peak = rng.uniform(2.5, 4.0)
        traffic[s:e] = base_rps * peak
        ramp = max(1, (e - s) // 4)
        traffic[max(0, s - ramp):s] = np.linspace(base_rps, base_rps * peak, min(ramp, s))
        tail = np.linspace(base_rps * peak, base_rps, ramp)[:max(0, len(traffic) - e)]
        traffic[e:e + len(tail)] = tail
        traffic += rng.normal(0, base_rps * 0.03, length)
    elif pattern == "diurnal":
        traffic = base_rps * (1.5 + 1.2 * np.sin(2 * np.pi * t / length - np.pi / 2))
        traffic += rng.normal(0, base_rps * 0.04, length)
    elif pattern == "rampup":
        traffic = np.linspace(base_rps * 0.5, base_rps * 3.2, length)
        traffic += rng.normal(0, base_rps * 0.04, length)
    else:
        raise ValueError(pattern)
    return np.clip(traffic, 0.5, None)


# ---------------------------------------------------------------------------
# Numerics for the end-to-end mixture
# ---------------------------------------------------------------------------
def _erf(x):
    """Vectorised erf (Abramowitz-Stegun 7.1.26, |err| < 1.5e-7)."""
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def _mixture_quantiles(means, sigmas, probs, qs):
    """Quantiles of a mixture of lognormals matched to (mean, sigma) per component."""
    means = np.maximum(np.asarray(means, dtype=float), 1e-6)
    sig = np.maximum(np.asarray(sigmas, dtype=float), 1e-6)
    probs = np.asarray(probs, dtype=float)
    s2 = np.log1p((sig / means) ** 2)
    m = np.log(means) - s2 / 2.0
    s = np.sqrt(s2)

    def cdf(x):
        z = (np.log(x) - m) / (s * np.sqrt(2.0))
        return float(np.sum(probs * 0.5 * (1.0 + _erf(z))))

    out = []
    for q in qs:
        lo, hi = 1e-3, float(np.max(means) * 50.0 + 1000.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


# ---------------------------------------------------------------------------
@dataclass
class ServiceState:
    replicas: int                              # desired replicas (ready + starting)
    cpu: float                                 # per-pod CPU allocation currently applied
    mem: float                                 # per-pod memory limit currently applied (MiB)
    starting: list = field(default_factory=list)   # seconds until each new pod is Ready
    resize: tuple | None = None                # (target_cpu, target_mem, seconds_remaining)
    backlog: float = 0.0                       # queued requests


class Simulator:
    """One simulated cluster. `step(rps)` advances STEP_SECONDS and returns the metrics
    a monitoring stack would report for that window."""

    def __init__(self, topology: Topology, rng=None, obs_noise: float = 0.0):
        self.topology = topology
        self.services = list(topology.services)
        self.entry_service = topology.entry
        self.call_graph = topology.call_graph
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.obs_noise = obs_noise
        self._paths = path_distribution(self.call_graph, self.entry_service)
        self._order = reachable(self.call_graph, self.entry_service)
        self.slo_ms = slo_for(topology)
        self.budget = budget_for(topology)
        self.reset()

    # kept for callers that used the v2 attribute names
    @property
    def replicas(self):
        return {s: st.replicas for s, st in self.state.items()}

    def reset(self, replicas=None, cpu=None, mem=None):
        """Fresh cluster at the manifest's initial allocation (overridable per controller; replicas / cpu / mem may be a
        number for every service or a {service: value} dict)."""
        def per(v, default, svc):
            return v.get(svc, default) if isinstance(v, dict) else (v or default)
        self.state = {s2: ServiceState(
            replicas=per(replicas, config.INITIAL_REPLICAS, s2),
            cpu=per(cpu, config.CPU_DEFAULT, s2),
            mem=float(per(mem, config.MEM_DEFAULT_MB, s2))) for s2 in self.services}
        self.prev_rps = {s: 0.0 for s in self.services}

    # -- actions ------------------------------------------------------------
    def apply(self, service, replica_delta=0, cpu=None, mem=None):
        """Applies an (already guard-approved) action to one service."""
        st = self.state[service]
        if replica_delta:
            target = int(np.clip(st.replicas + replica_delta, config.MIN_REPLICAS, config.MAX_REPLICAS))
            diff = target - st.replicas
            if diff > 0:
                st.starting.extend([config.POD_STARTUP_SECONDS] * diff)
            elif diff < 0:
                drop = -diff
                cancel = min(drop, len(st.starting))     # cancel not-yet-ready pods first
                if cancel:
                    st.starting = st.starting[:-cancel]
            st.replicas = target
        want_cpu = st.cpu if cpu is None else float(np.clip(cpu, config.CPU_MIN, config.CPU_MAX))
        want_mem = st.mem if mem is None else float(np.clip(mem, config.MEM_MIN_MB, config.MEM_MAX_MB))
        current_target = st.resize[:2] if st.resize else (st.cpu, st.mem)
        if (want_cpu, want_mem) != tuple(current_target):
            if (want_cpu, want_mem) == (st.cpu, st.mem):
                st.resize = None
            else:
                st.resize = (want_cpu, want_mem, st.replicas * config.ROLLING_SECONDS_PER_REPLICA)

    def target_allocation(self, service):
        st = self.state[service]
        return (st.resize[0], st.resize[1]) if st.resize else (st.cpu, st.mem)

    def cost(self):
        """Provisioned cost in core-equivalents (what the cluster bill would track)."""
        total = 0.0
        for s in self.services:
            cpu, mem = self.target_allocation(s)
            total += self.state[s].replicas * (cpu + config.MEM_COST_PER_GIB * mem / 1024.0)
        return total

    # -- dynamics -----------------------------------------------------------
    def _propagate(self, external_rps):
        incoming = {s: 0.0 for s in self.services}
        incoming[self.entry_service] = external_rps
        for node in self._topological_order():
            for callee, share in self.call_graph.get(node, []):
                incoming[callee] += incoming[node] * share
        return incoming

    def _topological_order(self):
        indeg = {s: 0 for s in self.services}
        for targets in self.call_graph.values():
            for c, _ in targets:
                indeg[c] += 1
        order, queue = [], [s for s, d in indeg.items() if d == 0]
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c, _ in self.call_graph.get(n, []):
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        return order

    def step(self, external_rps):
        dt = config.STEP_SECONDS
        p = sm.params()
        incoming = self._propagate(external_rps)
        out = {}

        for s in self.services:
            st = self.state[s]
            work = self.topology.work(s)
            lam = incoming[s]

            # ---- which pods are usable, and with what allocation, during this step ----
            ready = st.replicas - len(st.starting)
            n_eff = float(ready) + sum(max(0.0, min(1.0, (dt - t) / dt)) for t in st.starting)
            n_eff = max(n_eff, 0.0)
            cpu_eff, mem_eff = st.cpu, st.mem
            if st.resize:
                tc, tm, remaining = st.resize
                f_new = max(0.0, min(1.0, (dt - remaining) / dt))
                cpu_eff = st.cpu * (1 - f_new) + tc * f_new
                mem_eff = st.mem * (1 - f_new) + tm * f_new

            # memory pressure: a limit below the working set OOM-kills the pod
            inflight0 = lam * sm.mean_latency_ms(0.5, cpu_eff, work, p) / 1000.0 / max(n_eff, 1.0)
            ws = sm.memory_working_set_mb(inflight0, p)
            mem_util = ws / max(mem_eff, 1.0)
            oom_factor = 1.0
            if mem_util > 1.0:
                oom_factor = 0.3                       # crash-looping replicas serve little
            n_cap = n_eff * oom_factor

            cap = n_cap * sm.capacity_rps(cpu_eff, work, p)
            cap = max(cap, 1e-6)

            # ---- backlog dynamics ----
            backlog0 = st.backlog
            backlog1 = max(0.0, backlog0 + (lam - cap) * dt)
            q_max = cap * p.request_timeout_s
            dropped = 0.0
            if backlog1 > q_max:
                dropped = backlog1 - q_max
                backlog1 = q_max
            served = min(lam + backlog0 / dt, cap)
            drop_frac = min(1.0, dropped / max(lam * dt, 1e-9))
            st.backlog = backlog1

            rho = lam / cap
            steady = sm.mean_latency_ms(rho, cpu_eff, work, p)
            queue_delay_ms = (0.5 * (backlog0 + backlog1) / cap) * 1000.0
            mem_penalty = 1.0 + max(0.0, mem_util - 0.9) * 10.0
            own_mean = steady * mem_penalty + queue_delay_ms
            own_p95 = steady * mem_penalty * sm.tail_factor(rho, p) + 1.3 * queue_delay_ms

            cores_used = served * sm.cpu_ms_per_request(work, p) / 1000.0 + p.idle_cores * n_eff
            util = min(1.0, cores_used / max(n_eff * cpu_eff, 1e-6)) if n_eff > 0 else 1.0
            inflight = (lam * own_mean / 1000.0 + 0.5 * (backlog0 + backlog1)) / max(n_eff, 1.0)

            if self.obs_noise > 0:
                nz = self.rng.lognormal(0.0, self.obs_noise, 2)
                own_mean, own_p95 = own_mean * nz[0], own_p95 * nz[0]
                util = float(min(1.0, util * nz[1]))

            out[s] = {
                "requests": lam,
                "rps_prev": self.prev_rps[s],
                "latency_ms": own_mean,
                "latency_p95_ms": own_p95,
                "utilization": util,
                "offered_load": rho,
                "replicas": st.replicas,
                "ready_replicas": ready,
                "pending_replicas": len(st.starting),
                "cpu": cpu_eff if not st.resize else st.resize[0],
                "mem": mem_eff if not st.resize else st.resize[1],
                "mem_utilization": min(mem_util, 2.0),
                "inflight": inflight,
                "work_ms": work,
                "error_rate": drop_frac,
                "backlog": backlog1,
            }
            self.prev_rps[s] = lam

            # ---- advance pod / resize timers ----
            st.starting = [t - dt for t in st.starting if t - dt > 0]
            if st.resize:
                tc, tm, remaining = st.resize
                if remaining - dt <= 0:
                    st.cpu, st.mem, st.resize = tc, tm, None
                else:
                    st.resize = (tc, tm, remaining - dt)

        self._fill_inclusive(out)
        out["__e2e__"] = self._e2e(out)
        return out

    # -- derived quantities -------------------------------------------------
    def _fill_inclusive(self, metrics):
        """Per-service inclusive latency (own + expected downstream), as Prometheus'
        request-duration histogram on that service would report it."""
        spread2 = {}
        for s in reversed(self._topological_order()):
            m = metrics[s]
            callees = self.call_graph.get(s, [])
            incl = m["latency_ms"] + sum(w * metrics[c]["latency_incl_ms"] for c, w in callees)
            sp2 = (m["latency_p95_ms"] - m["latency_ms"]) ** 2 + sum(
                w * spread2[c] for c, w in callees)
            spread2[s] = sp2
            m["latency_incl_ms"] = incl
            m["latency_incl_p95_ms"] = incl + float(np.sqrt(sp2))

    def _e2e(self, metrics):
        means, sigmas, probs, ok = [], [], [], 0.0
        for path, prob in self._paths:
            means.append(sum(metrics[s]["latency_ms"] for s in path))
            var = sum(((metrics[s]["latency_p95_ms"] - metrics[s]["latency_ms"]) / 1.645) ** 2
                      for s in path)
            sigmas.append(float(np.sqrt(var)))
            probs.append(prob)
            survive = 1.0
            for s in path:
                survive *= 1.0 - metrics[s]["error_rate"]
            ok += prob * survive
        p50, p95 = _mixture_quantiles(means, sigmas, probs, (0.5, 0.95))
        return {
            "mean_ms": float(np.dot(means, probs)),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p95 * sm.params().p99_over_p95,
            "error_rate": float(1.0 - ok),
        }


# ---------------------------------------------------------------------------
# Critical path (kept here for callers that import it from the simulator)
# ---------------------------------------------------------------------------
def critical_path(call_graph, entry, latencies):
    """Longest latency chain via depth-first search. See topology_intel/critical_path.py."""
    return critical_path_dfs(call_graph, entry, latencies)
