"""
bottleneck_eval.py

Does the controller find the bottleneck? Every non-entry service of the application is turned, one at a time, into the bottleneck
(topologies.bottleneck_scenario: the request mix is shifted towards it, the traffic scaled until it needs more cores than its small starting limit, no other service saturates) and each controller runs the same
episodes. Reported per controller, over all variants:

  hit          in how many scenarios the bottleneck service received capacity (a replica or CPU increase the guard let through) within the first 2 steps
  idle adds    share of all the capacity it added that went to services with CPU utilisation below 50 % (wasted)
  steps to SLO steps until the end-to-end p95 is back under the SLO (episode length + 1 if never)
  violations   share of steps with the end-to-end p95 above the SLO;  cost;  scalarized score (lower is better)

    python -m src.evaluation.bottleneck_eval --agent cpu=models/mo_td3_trainticket_cpu.pt --agent large=models/mo_td3_large.pt --out results/bottleneck_eval.json

Run it in the regime of the target cluster (source profiles/trainticket.env, export ALLOW_MEMORY_RESIZE=0 and SIM_PARAMS_PATH) as for regime_eval.
"""

import argparse
import json

import numpy as np

from src import config
from src.evaluation.evaluate import tune_hpa
from src.rl_agent import rewards
from src.rl_agent.controllers import RLController, RuleController
from src.rl_agent.env import AutoscalerEnv, topology_base_rps
from src.rl_agent.mo_td3 import MOTD3Agent
from src.topologies import bottleneck_scenario, get_topology


def run_episode(topo, controller, replica_only, seed, hot):
    env = AutoscalerEnv(topology=topo, seed=seed, replica_only=replica_only)
    obs = env.reset("normal" if seed % 2 else "burst", topology=topo)
    w = np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32)
    first_hot_add, adds, idle_adds, p95, cost, pen, viol, fixed_at = None, 0, 0, [], [], [], 0, None
    done, t = False, 0
    while not done:
        before = obs["metrics"]
        obs, r, done, info = env.step_proposals(controller.decide(obs))
        t += 1
        for s, a in info["approved"].items():
            if a["dr"] > 0 or a["cpu"] > before[s]["cpu"] * 1.10:                  # a real addition: a replica, or at least 10 % more CPU (not a 0.05-core tweak)
                adds += 1
                idle_adds += before[s]["utilization"] < IDLE_UTIL            # capacity given to a service that was not busy
                if s == hot and first_hot_add is None:
                    first_hot_add = t
        p = obs["e2e"]["p95_ms"]
        p95.append(p); viol += p > env.slo_ms; cost.append(obs["cost"]); pen.append(rewards.scalarize(-np.asarray(r), w))
        if fixed_at is None and p <= env.slo_ms:
            fixed_at = t
    return {"hit": first_hot_add is not None and first_hot_add <= 2,            # the bottleneck got capacity in the first two steps
            "idle_add_share": idle_adds / adds if adds else 0.0, "steps_to_slo": fixed_at if fixed_at is not None else len(p95) + 1,
            "violations": viol / len(p95), "p95": float(np.mean(p95)), "cost": float(np.mean(cost)), "score": float(np.mean(pen))}


KEYS = ("hit", "idle_add_share", "steps_to_slo", "violations", "p95", "cost", "score")
IDLE_UTIL = 0.5          # a service below this CPU utilisation is not busy: capacity added to it is wasted


def scenarios(base, seed=0):
    """{service: scenario topology} for every service of `base` that can be made the only bottleneck (topologies.bottleneck_scenario)."""
    rps = topology_base_rps(base)
    out = {}
    for h in base.services:
        topo, _ = bottleneck_scenario(base, np.random.default_rng(seed), rps, hot=h)
        if topo is not None:
            out[h] = topo
    return out


def evaluate(topology_name, agents, episodes=4, log=print):
    base = get_topology(topology_name)
    scen = scenarios(base)
    best, _ = tune_hpa(base, n_episodes=6)
    w = np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32)
    makers = {"hpa (tuned)": (lambda t: RuleController("hpa", t.services, label="hpa_tuned", initial={"cpu": best["cpu"]},
                                                       target_utilization=best["target"]), False),
              "static": (lambda t: RuleController("static", t.services), False)}
    for name, path in agents.items():
        a = MOTD3Agent.load(path)
        makers[name] = (lambda t, a=a, n=name: RLController(a, t.services, w=w, name=n), a.flags["replica_only"])
    out = {}
    for name, (mk, replica_only) in makers.items():
        per_service = {}
        for hot, topo in scen.items():
            rows = [run_episode(topo, mk(topo), replica_only, seed, hot) for seed in range(episodes)]
            per_service[hot] = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
        agg = {k: float(np.mean([v[k] for v in per_service.values()])) for k in KEYS}
        out[name] = {**agg, "per_service": per_service}
        log(f"  {name:26} bottleneck got capacity within 2 steps {agg['hit'] * 100:5.1f}%  | capacity wasted on idle services {agg['idle_add_share'] * 100:5.1f}%  | "
            f"steps to SLO {agg['steps_to_slo']:4.1f}  violations {agg['violations'] * 100:5.1f}%  cost {agg['cost']:5.2f}  score {agg['score']:.3f}")
    return {"topology": topology_name, "bottleneck_services": list(scen), "episodes_per_service": episodes,
            
            "scenarios": {h: {"load_scale": round(t.load_scale, 2), "start_cpu": t.initial_cpu} for h, t in scen.items()},
            "regime": {k: getattr(config, k) for k in ("STEP_SECONDS", "POD_STARTUP_SECONDS", "CPU_MAX", "MAX_REPLICAS", "ALLOW_MEMORY_RESIZE")},
            "controllers": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", default="trainticket-sim")
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--episodes", type=int, default=4, help="episodes per bottleneck service")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(a.topology, dict(x.split("=", 1) for x in a.agent), a.episodes)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)
