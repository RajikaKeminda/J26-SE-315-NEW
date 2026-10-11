"""
engines.py

The control engines behind the dashboard and the headless loop. Two modes emit the
SAME payload shape, so the front end is identical:

  DemoEngine   drives the calibrated simulator. No cluster needed - for development,
               presentations and the defence.
  LiveEngine   reads real Prometheus / Kubernetes state, decides, and applies real
               scaling + resizing actions (or only logs them with shadow=True).

Both run the controller of your choice (the learned MO-TD3 agent at a chosen
preference, or the HPA / VPA / threshold / static baselines) behind the same safety
guard, and can switch controller at run time.
"""

import json
import time

import numpy as np

from src import config
from src.baselines.rule_based import BASELINES
from src.rl_agent import rewards
from src.rl_agent.controllers import RLController, RuleController
from src.rl_agent.env import AutoscalerEnv, topology_base_rps
from src.rl_agent.mo_td3 import MOTD3Agent
from src.safety.guards import SafetyGuard
from src.simulation.simulator import generate_traffic, slo_for
from src.topologies import DEMO6, Topology
from src.topology_intel.graph_builder import build_observation

CONTROLLERS = ["mo_td3", "hpa", "vpa", "threshold", "static"]
HISTORY_LEN = 80


def _clean(x):
    """Recursively converts numpy scalars to plain Python so the payload is JSON-safe."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


# ---------------------------------------------------------------------------
def agent_path(arm=None):
    import os
    if arm:
        return os.path.join(config.MODELS_DIR, f"{arm}.pt")
    if os.path.exists(config.POLICY_PATH):
        return config.POLICY_PATH
    return os.path.join(config.MODELS_DIR, f"{config.DEFAULT_ARM}.pt")


def load_agent(arm=None):
    """Returns (agent, name) or (None, None) when no usable checkpoint exists (the engines
    then run the HPA rules, and say so)."""
    import os
    path = agent_path(arm)
    if not os.path.exists(path):
        print(f"[engine] no trained policy at {path} - using HPA rules.")
        return None, None
    try:
        agent = MOTD3Agent.load(path)
    except Exception as e:                       # old-format checkpoint, corrupt file ...
        print(f"[engine] cannot load {path}: {e} - using HPA rules.")
        return None, None
    print(f"[engine] loaded policy {path}")
    return agent, os.path.splitext(os.path.basename(path))[0]


def make_controller(name, services, agent, w):
    if name == "mo_td3":
        if agent is None:
            return RuleController("hpa", services, label="hpa-fallback")
        return RLController(agent, services, w=w)
    return RuleController(name, services)


def _is_baseline(controller) -> bool:
    return getattr(controller, "name", "") in BASELINES


# ---------------------------------------------------------------------------
class BaseEngine:
    mode = "base"

    def __init__(self, agent, policy_name, controller="mo_td3", w=config.DEFAULT_PREFERENCE, shadow=False):
        self.agent, self.policy_name = agent, policy_name
        self.w = [float(x) for x in rewards.normalize_preference(w)]
        self.shadow = shadow
        self.controller_name = controller if (controller != "mo_td3" or agent is not None) else "hpa"
        self.step_i = 0
        self.history = {"p95": [], "cost": [], "rps": [], "slo": []}
        self.edge_calls = {}

    # -- control surface ------------------------------------------------
    def set_controller(self, name):
        if name not in CONTROLLERS:
            raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLERS}")
        if name == "mo_td3" and self.agent is None:
            raise ValueError("no trained policy is loaded")
        self.controller_name = name
        self._rebuild_controller()

    def set_preference(self, w):
        self.w = [float(x) for x in rewards.normalize_preference(w)]
        if isinstance(self.controller, RLController):
            self.controller.w = np.asarray(self.w, dtype=np.float32)

    def _rebuild_controller(self):
        self.controller = make_controller(self.controller_name, self.services, self.agent, self.w)

    def _controlled(self, svc):
        return not config.CONTROLLED_SERVICES or svc in config.CONTROLLED_SERVICES

    # -- payload --------------------------------------------------------
    def _push_history(self, e2e, cost, rps, slo_ms):
        h = self.history
        h["p95"].append(round(e2e["p95_ms"], 1)); h["cost"].append(round(cost, 2))
        h["rps"].append(round(rps, 1)); h["slo"].append(round(slo_ms, 1))
        for k in h:
            h[k] = h[k][-HISTORY_LEN:]

    def _payload(self, metrics, obs, e2e, cost, proposals, approved, reasons, penalties,
                 rps, slo_ms, budget, call_graph, entry, services, breaker_open, **extra):
        crit_path = obs["critical_path"]
        idx = {s: i for i, s in enumerate(services)}
        ehg = obs["A_ehg"]

        edges = []
        for caller, targets in call_graph.items():
            for callee, share in targets:
                if caller not in idx or callee not in idx:
                    continue
                on_path = (caller in crit_path and callee in crit_path
                           and crit_path.index(callee) - crit_path.index(caller) == 1)
                edges.append({"source": caller, "target": callee, "traffic_share": round(share, 3),
                              "calls": self.edge_calls.get(f"{caller}->{callee}"),
                              "latency_pressure": round(float(ehg[idx[caller], idx[callee]]), 3),
                              "on_critical_path": on_path})

        nodes = []
        for s in services:
            m = metrics[s]
            prop, app = proposals.get(s, {}), approved.get(s, {})
            util = m["utilization"]
            nodes.append({
                "id": s, "replicas": m["replicas"], "ready": m.get("ready_replicas", m["replicas"]),
                "pending": m.get("pending_replicas", 0),
                "cpu": round(m["cpu"], 3), "mem": round(m["mem"]),
                "latency_ms": round(m["latency_ms"], 1), "latency_p95_ms": round(m["latency_p95_ms"], 1),
                "requests": round(m["requests"], 1), "utilization": round(util, 3),
                "mem_utilization": round(m.get("mem_utilization", 0.0), 3),
                "on_critical_path": s in crit_path, "is_entry": s == entry,
                "controlled": self._controlled(s),
                "proposed_delta": int(prop.get("dr", 0)), "applied_delta": int(app.get("dr", 0)),
                "proposed_cpu": round(prop.get("cpu", m["cpu"]), 3), "applied_cpu": round(app.get("cpu", m["cpu"]), 3),
                "proposed_mem": round(prop.get("mem", m["mem"])), "applied_mem": round(app.get("mem", m["mem"])),
                "blocked_reason": reasons.get(s),
                "status": "critical" if util > 0.95 else "warn" if util > 0.75 else "ok",
            })

        payload = {
            "step": self.step_i, "timestamp": time.time(), "mode": self.mode,
            "external_rps": round(float(rps), 1), "slo_ms": round(slo_ms, 1),
            "e2e": {k: round(float(v), 4 if k == "error_rate" else 1) for k, v in e2e.items()},
            "slo_breached": e2e["p95_ms"] > slo_ms,
            "critical_path": crit_path,
            "critical_path_latency_ms": round(obs["critical_total"], 1),
            # the breaker only matters for the learned agent (it is what hands control to HPA rules)
            "breaker_open": bool(breaker_open and self.controller_name == "mo_td3"),
            "total_replicas": sum(m["replicas"] for m in metrics.values()),
            "cost": round(cost, 2), "budget": round(budget, 2),
            "penalties": {k: round(float(v), 3) for k, v in penalties.items()},
            "nodes": nodes, "edges": edges, "history": self.history,
            "stale": False, "paused": False, "shadow": self.shadow,
            "controller": self.controller_name, "preference": self.w,
            "objectives": list(config.OBJECTIVES),
            "policy": self.policy_name or "none",
        }
        payload.update(extra)
        return _clean(payload)


# ---------------------------------------------------------------------------
class DemoEngine(BaseEngine):
    mode = "demo"
    LENGTH = 400

    def __init__(self, agent, policy_name=None, scenario="burst", seed=None, topology: Topology = DEMO6,
                 controller="mo_td3", w=config.DEFAULT_PREFERENCE, obs_noise=0.05):
        super().__init__(agent, policy_name, controller, w)
        self.topology = topology
        self.services, self.entry, self.call_graph = list(topology.services), topology.entry, topology.call_graph
        self.scenario = scenario
        self.env = AutoscalerEnv(topology=topology, seed=seed, obs_noise=obs_noise, episode_length=self.LENGTH)
        self._rebuild_controller()
        self.reset()

    def reset(self, scenario=None):
        if scenario:
            self.scenario = scenario
        base = topology_base_rps(self.topology)
        traffic = generate_traffic(self.scenario, self.LENGTH + 1, base, self.env.rng)
        self.env.reset(topology=self.topology, traffic=traffic,
                       initial=getattr(self.controller, "initial", None))
        self.step_i = 0
        self.history = {"p95": [], "cost": [], "rps": [], "slo": []}
        self.controller.reset()

    def tick(self):
        if self.env.t >= self.env.episode_length:        # scenario finished: replay it
            self.reset()
        obs = self.env.observe()
        proposals = self.controller.decide(obs, breaker_open=self.env.guard.breaker_open)
        obs, _, _, info = self.env.step_proposals(proposals, physical_only=_is_baseline(self.controller))
        self._push_history(info["e2e"], info["cost"], obs["external_rps"], info["slo_ms"])
        payload = self._payload(
            info["metrics"], obs, info["e2e"], info["cost"], proposals, info["approved"], info["reasons"],
            {k: v for k, v in zip(config.OBJECTIVES, [info["penalties"]["slo_penalty"],
                                                        info["penalties"]["cost_norm"],
                                                        info["penalties"]["balance_penalty"]])},
            obs["external_rps"], info["slo_ms"], info["budget"], self.call_graph, self.entry,
            self.services, info["breaker_open"], scenario=self.scenario)
        self.step_i += 1
        return payload


# ---------------------------------------------------------------------------
REDISCOVER_EVERY_TICKS = 6


def _is_dag(services, graph):
    indeg = {s: 0 for s in services}
    for targets in graph.values():
        for c, _ in targets:
            if c in indeg:
                indeg[c] += 1
    queue, seen = [s for s, d in indeg.items() if d == 0], 0
    while queue:
        n = queue.pop()
        seen += 1
        for c, _ in graph.get(n, []):
            if c in indeg:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
    return seen == len(services)


class LiveEngine(BaseEngine):
    """
    Same decision logic, but state comes from real Prometheus/Kubernetes and actions go to
    the real Kubernetes API. Imports are lazy so demo mode works without `kubernetes` /
    `requests` configured.

    Safety behaviour beyond the guard:
      * If Prometheus queries fail, or a Deployment can't be read, the tick HOLDS (no
        scaling, payload flagged stale) rather than reading missing data as zero load.
      * `shadow=True` runs everything but never calls the Kubernetes API.
    """
    mode = "live"

    def __init__(self, agent, policy_name=None, shadow=False, autodiscover=None,
                 controller="mo_td3", w=config.DEFAULT_PREFERENCE):
        from src.data_collection import prometheus_client, jaeger_client
        from src.inference.live_state import LiveStateBuilder
        from src.k8s_action import k8s_client
        if config.METRICS_SOURCE == "trainticket":
            from src.data_collection import trainticket_metrics as prometheus_client   # OTel spans + cAdvisor
        self._prom, self._jaeger, self._k8s = prometheus_client, jaeger_client, k8s_client
        super().__init__(agent, policy_name, controller, w, shadow)
        self.topology = config.TOPOLOGY
        self.services, self.entry = list(config.SERVICES), config.ENTRY_SERVICE
        self.call_graph = {k: list(v) for k, v in config.CALL_GRAPH.items()}
        self.autodiscover = config.AUTODISCOVER_TOPOLOGY if autodiscover is None else autodiscover
        self.live_topology = None
        if config.METRICS_SOURCE == "trainticket":                  # graph learned only from live traces
            from src.topology_intel.live_topology import LiveTopology
            self.live_topology = LiveTopology(self.entry, config.TOPOLOGY_HALF_LIFE_S,
                                              config.TOPOLOGY_TTL_S, config.TOPOLOGY_MIN_CALLS)
            self.autodiscover = False
        self.edge_calls = {}
        self.topology_source = "config"
        self.builder = LiveStateBuilder(self.services, self.entry, self.call_graph,
                                        priors={s: self.topology.work(s) for s in self.services})
        self.guard = SafetyGuard(services=self.services, budget=config.live_budget(len(self.services)))
        self._last_ok_ts = None
        self._last_payload = None
        self._rebuild_controller()
        if self.autodiscover:
            self._rediscover()

    def reset(self, scenario=None):
        self.step_i = 0
        self.history = {"p95": [], "cost": [], "rps": [], "slo": []}
        self.guard.reset()
        self.controller.reset()

    # -- topology ---------------------------------------------------------
    def _rediscover(self):
        """Merge the topology seen in live traces into the known one. Conservative on
        purpose: trace sampling can miss rarely-hit branches, so a service/edge absent from
        the sample is NOT treated as removed. New services are added; callers with observed
        outbound calls get their edge weights replaced by the observed traffic shares."""
        found = self._jaeger.discover_topology(
            self.entry, config.JAEGER_LOOKBACK_SECONDS * 4, config.JAEGER_TRACE_LIMIT * 4)
        if not found:
            return
        new_services, _, new_graph = found
        services = list(self.services) + [s for s in new_services if s not in self.services]
        graph = {s: list(self.call_graph.get(s, [])) for s in services}
        for caller, targets in new_graph.items():
            if targets:
                graph[caller] = [(c, float(w)) for c, w in targets]
        if not _is_dag(services, graph):
            return
        if services != self.services or graph != self.call_graph:
            added = [s for s in services if s not in self.services]
            self.services, self.call_graph = services, graph
            self.guard.services = services
            self.guard.budget = config.live_budget(len(services))
            self.builder.set_topology(services, self.entry, graph)
            self._rebuild_controller()
            if added:
                print(f"[live] topology discovery added services: {added}")
        self.topology_source = "jaeger"

    def _update_live_topology(self):
        """Fold this window's traces into the live graph; services = reachable from the entry AND backed by a
        readable Deployment (a traced service without one - a database, a sidecar - is external, not an error).
        Returns the Kubernetes state of the services kept."""
        lt = self.live_topology
        lt.update(getattr(self._prom, "last_traces", []) or [])
        candidates = lt.services()
        traced = set(candidates)
        if config.OBSERVE_UNTRACED and hasattr(self._k8s, "get_all_state"):
            everything = self._k8s.get_all_state()
            state = {s: v for s, v in everything.items() if s in traced or v["ready"] > 0}
            candidates = candidates + sorted(s for s in state if s not in traced)       # untraced: isolated nodes
        else:
            state = self._k8s.get_state(candidates)
        if self.entry not in state:
            raise RuntimeError(f"entry service {self.entry} has no readable Deployment")
        services = [s for s in candidates if s in state]
        graph = lt.call_graph(services)
        self.edge_calls = {f"{a}->{b}": round(c, 1) for (a, b), c in lt.observed_calls().items()
                           if a in graph and b in dict(graph[a])}
        if services != self.services or graph != self.call_graph:
            added = [s for s in services if s not in self.services]
            removed = [s for s in self.services if s not in services]
            self.services, self.call_graph = services, graph
            self.guard.services = services
            self.guard.budget = config.live_budget(len(services))
            self.builder.set_topology(services, self.entry, graph)
            self._rebuild_controller()
            if added or removed:
                print(f"[live] topology changed: +{added} -{removed} ({len(services)} services)")
        self.topology_source = "jaeger-live"
        return {s: state[s] for s in services}

    def _apply_cluster_safety(self, approved, reasons, metrics, prom):
        """Real-cluster limits the per-service guard cannot know about (see config.NODE_MEMORY_RESERVE_MB)."""
        if not config.ALLOW_RESIZE:
            for svc, a in approved.items():
                if a["cpu"] != metrics[svc]["cpu"] or a["mem"] != metrics[svc]["mem"]:
                    a["cpu"], a["mem"] = metrics[svc]["cpu"], metrics[svc]["mem"]
                    reasons.setdefault(svc, "resize disabled (ALLOW_RESIZE=0)")
        ups = [s for s, a in approved.items() if a["dr"] > 0 and self._controlled(s)]
        if not ups or not hasattr(self._prom, "node_memory_headroom_mb"):
            return
        try:
            room = self._prom.node_memory_headroom_mb() - config.NODE_MEMORY_RESERVE_MB
        except Exception as e:                       # cannot see the node -> do not add pods
            room = -1.0
            why = f"scale-up blocked: node memory unknown ({e})"
        else:
            why = None
        for s in sorted(ups, key=lambda x: -metrics[x].get("utilization", 0.0)):     # busiest service first
            need = approved[s]["dr"] * max(prom.get(s, {}).get("mem_mb", 0.0), 256.0)
            if need <= room:
                room -= need
            else:
                approved[s]["dr"] = 0
                reasons[s] = why or f"scale-up blocked: node memory headroom ({max(room, 0):.0f} MiB left, need {need:.0f})"

    def _critical_path_evidence(self, obs):
        """Checks the current critical path against the real requests of this window (needs OpenTelemetry traces)."""
        traces = getattr(self._prom, "last_traces", None)
        if not traces:
            return None
        try:
            from src.evaluation.critical_path_eval import evidence
            return evidence(traces, obs["critical_path"], self.entry, self.call_graph)
        except Exception:                        # never let an explanation break the control loop
            return None

    def _topology_info(self):
        lt = self.live_topology
        n_edges = sum(len(t) for t in self.call_graph.values())
        info = {"source": self.topology_source, "services": len(self.services), "edges": n_edges,
                "controlled": sum(1 for s in self.services if self._controlled(s))}
        if lt is not None:
            info.update(updated_at=lt.updated_at, traces_in_window=lt.last_window_traces,
                        half_life_s=lt.half_life_s, ttl_s=lt.ttl_s)
        return info

    # -- stale hold --------------------------------------------------------
    def _hold(self, reason):
        """Payload for a tick where we refuse to act on missing data."""
        step = self.step_i
        self.step_i += 1
        age = None if self._last_ok_ts is None else round(time.time() - self._last_ok_ts, 1)
        if self._last_payload is not None:
            p = json.loads(json.dumps(self._last_payload))
            p.update(step=step, timestamp=time.time())
            for n in p["nodes"]:
                n["proposed_delta"] = n["applied_delta"] = 0
                n["blocked_reason"] = None
            p["nodes"][0]["blocked_reason"] = f"HOLD: {reason}"
        else:
            zero = {s: {"requests": 0.0, "rps_prev": 0.0, "latency_ms": 0.0, "latency_p95_ms": 0.0,
                        "utilization": 0.0, "replicas": 1, "cpu": config.CPU_DEFAULT,
                        "mem": float(config.MEM_DEFAULT_MB), "latency_incl_ms": 0.0,
                        "latency_incl_p95_ms": 0.0} for s in self.services}
            e2e = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "error_rate": 0.0}
            obs = build_observation(self.services, self.call_graph, self.entry, zero, e2e, 0.0,
                                    self.guard.budget, slo_for(self.topology))
            p = self._payload(zero, obs, e2e, 0.0, {}, {}, {self.entry: f"HOLD: {reason}"},
                              {k: 0.0 for k in config.OBJECTIVES}, 0.0, config.SLO_LATENCY_MS,
                              self.guard.budget, self.call_graph, self.entry, self.services,
                              self.guard.breaker_open)
            p["step"] = step
        p.update(stale=True, stale_reason=reason, metrics_age_s=age, shadow=self.shadow,
                 topology_source=self.topology_source, topology=self._topology_info())
        return p

    # -- main tick ----------------------------------------------------------
    def tick(self):
        if self.autodiscover and self.step_i % REDISCOVER_EVERY_TICKS == 0:
            self._rediscover()

        try:
            prom = self._prom.collect()
            if self.live_topology is not None:
                k8s = self._update_live_topology()
            else:
                k8s = self._k8s.get_state(self.services)
        except Exception as e:           # Prometheus down, kubeconfig missing, API unreachable
            return self._hold(f"cluster read failed: {e}")
        if self.live_topology is not None and len(self.services) < 2:
            return self._hold("discovering topology from traces - no cross-service calls seen yet")
        ages = self._k8s.pod_ages() if hasattr(self._k8s, "pod_ages") else {}
        for s_, v_ in k8s.items():
            v_["youngest_pod_s"] = ages.get(s_)
        missing = [s for s in self.services if s not in k8s]
        if missing:
            return self._hold(f"cannot read deployment(s): {', '.join(missing)}")
        if not prom:
            return self._hold("prometheus returned no series (targets not scraped yet?)")
        self._last_ok_ts = time.time()

        try:        # per-edge latency percentiles from traces (the EHG); optional
            traces = getattr(self._prom, "last_traces", None)       # reuse the window the metrics source fetched
            if not traces:
                traces = self._jaeger.fetch_recent_traces(
                    self.entry, config.JAEGER_LOOKBACK_SECONDS, config.JAEGER_TRACE_LIMIT)
            edges = self._jaeger.aggregate_window(traces)
        except Exception:
            edges = {}
        metrics, e2e, cost = self.builder.build(prom, k8s, edges)
        for svc, (rz, sd) in self.guard.readiness(self.step_i).items():
            if svc in metrics:
                metrics[svc]["resize_cooldown"], metrics[svc]["scaledown_cooldown"] = rz, sd
        slo_ms = slo_for_live(self.call_graph, self.entry)
        obs = build_observation(self.services, self.call_graph, self.entry, metrics, e2e, cost,
                                self.guard.budget, slo_ms)

        self.guard.observe_slo(e2e["p95_ms"], slo_ms, self.step_i)
        proposals = self.controller.decide(obs, breaker_open=self.guard.breaker_open)
        approved, reasons = self.guard.filter(proposals, metrics, self.step_i,
                                              physical_only=_is_baseline(self.controller))

        self._apply_cluster_safety(approved, reasons, metrics, prom)
        for svc in list(approved):
            if not self._controlled(svc):
                approved[svc] = {"dr": 0, "cpu": metrics[svc]["cpu"], "mem": metrics[svc]["mem"]}
                reasons[svc] = "observed only (not in CONTROLLED_SERVICES)"
        for svc, a in approved.items():
            if (a["dr"] or a["cpu"] != metrics[svc]["cpu"] or a["mem"] != metrics[svc]["mem"]) and not self.shadow:
                try:
                    self._k8s.apply_action(svc, a)
                except Exception as e:      # never let a scaling failure kill the loop
                    reasons[svc] = f"k8s error: {e}"
                    approved[svc] = {"dr": 0, "cpu": metrics[svc]["cpu"], "mem": metrics[svc]["mem"]}

        _, pen = rewards.objective_penalties(metrics, self.services, e2e, cost, self.guard.budget, slo_ms)
        rps = metrics[self.entry]["requests"]
        self._push_history(e2e, cost, rps, slo_ms)
        payload = self._payload(metrics, obs, e2e, cost, proposals, approved, reasons,
                                {"slo": pen["slo_penalty"], "cost": pen["cost_norm"],
                                 "balance": pen["balance_penalty"]},
                                rps, slo_ms, self.guard.budget, self.call_graph, self.entry,
                                self.services, self.guard.breaker_open,
                                topology_source=self.topology_source, topology=self._topology_info(),
                                critical_path_evidence=self._critical_path_evidence(obs),
                                jaeger_url=config.JAEGER_PUBLIC_URL)
        self.step_i += 1
        self._last_payload = payload
        return payload


class ReplayEngine(BaseEngine):
    """Plays back payloads recorded from a real run (src/dashboard/record.py), in a loop. A safety net for a presentation:
    if the cluster or the network is unavailable the control room still shows exactly what a real run looked like,
    and says so (mode "replay", a banner on the page). It computes nothing and touches no cluster."""
    mode = "replay"

    def __init__(self, path, agent=None, policy_name=None, controller="mo_td3", w=config.DEFAULT_PREFERENCE):
        super().__init__(agent, policy_name, controller, w, shadow=True)
        import json
        with open(path) as f:
            self.frames = [json.loads(l) for l in f if l.strip()]
        self.frames = [p for p in self.frames if p.get("nodes")]
        if not self.frames:
            raise ValueError(f"{path}: no recorded payloads")
        self.services = [n["id"] for n in self.frames[0]["nodes"]]
        self.call_graph, self.entry = {}, self.frames[0]["nodes"][0]["id"]
        self._rebuild_controller = lambda: None
        self.controller = None

    def reset(self, scenario=None):
        self.step_i = 0

    def set_controller(self, name):
        pass                                                 # a recording cannot change its mind

    def tick(self):
        p = dict(self.frames[self.step_i % len(self.frames)])
        p.update(step=self.step_i, mode="replay", replay=True, replay_length=len(self.frames), timestamp=time.time())
        self.step_i += 1
        return p


def slo_for_live(call_graph, entry):
    from src.topology_intel.critical_path import critical_path_dfs
    if config.SLO_LATENCY_MS_OVERRIDE:
        return float(config.SLO_LATENCY_MS_OVERRIDE)
    _, path = critical_path_dfs(call_graph, entry, {s: 1.0 for s in call_graph})
    return config.slo_for_depth(len(path))
