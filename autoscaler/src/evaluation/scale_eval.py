"""
scale_eval.py

Does the agent scale with the number of services? Two questions, measured on random acyclic topologies of N services
(same generator as training, but N fixed and far larger than the 4-12 it was trained on):

  cost     how long does one control decision take (graph embedding + policy) and one simulator step?
  quality  SLO-violation rate, p95 and cost of the learned agent against the default HPA, the tuned HPA and no scaling,
           on matched seeds and traffic patterns

    python -m src.evaluation.scale_eval --sizes 8 14 25 50 100 --topologies 3 --episodes 8
"""

import argparse
import json
import os
import time

import numpy as np

from src import config
from src.evaluation.evaluate import load_arm, tune_hpa
from src.evaluation.metrics import evaluate_controller
from src.rl_agent.controllers import RLController, RuleController
from src.rl_agent.env import AutoscalerEnv
from src.simulation.simulator import random_topology



def decision_cost(agent, topology, steps=10):
    env = AutoscalerEnv(topology=topology, seed=0)
    obs = env.reset("burst")
    ctrl = RLController(agent, topology.services, w=np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32))
    ctrl.reset()
    t_dec, t_env = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        proposals = ctrl.decide(obs, breaker_open=False)
        t1 = time.perf_counter()
        obs, _, _, _ = env.step_proposals(proposals)
        t_env.append(time.perf_counter() - t1)
        t_dec.append(t1 - t0)
    return 1000 * float(np.median(t_dec)), 1000 * float(np.median(t_env))


def run(sizes, n_topologies, n_episodes, arm="mo_td3_full", log=print):
    agent = load_arm(arm, 0)
    assert agent is not None, f"no checkpoint for {arm}"
    out = []
    for n in sizes:
        topos = [random_topology(np.random.default_rng(777_000 + 100 * n + i), n, n) for i in range(n_topologies)]
        dec, stepms = zip(*[decision_cost(agent, t) for t in topos])
        # a fair baseline: the HPA's CPU sizing and target are re-tuned for THIS size on a topology that is not evaluated
        tuning_topo = random_topology(np.random.default_rng(999_000 + n), n, n)
        hpa_best, _ = tune_hpa(tuning_topo, n_episodes=4)
        HPA_TUNED = {"cpu": hpa_best["cpu"], "target": hpa_best["target"]}
        makers = {
            "mo_td3": lambda t: (lambda: RLController(agent, t.services, w=np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32))),
            "hpa": lambda t: (lambda: RuleController("hpa", t.services)),
            "hpa_tuned": lambda t, h=None: (lambda: RuleController("hpa", t.services, label="hpa_tuned",
                                                           initial={"cpu": HPA_TUNED["cpu"]}, target_utilization=HPA_TUNED["target"])),
            "static": lambda t: (lambda: RuleController("static", t.services)),
        }
        row = {"services": n, "topologies": n_topologies, "hpa_tuned_params": HPA_TUNED, "decision_ms": round(float(np.mean(dec)), 1),
               "sim_step_ms": round(float(np.mean(stepms)), 1), "controllers": {}}
        for name, mk in makers.items():
            runs = [evaluate_controller(mk(t), t, n_episodes=n_episodes, seed0=5_000 + 13 * i) for i, t in enumerate(topos)]
            row["controllers"][name] = {k: round(float(np.mean([r[k] for r in runs])), 4)
                                        for k in ("slo_violation_rate", "p95_latency", "mean_cost", "scalarized")}
        out.append(row)
        c = row["controllers"]
        log(f"N={n:4d}  decision {row['decision_ms']:7.1f} ms  | SLO viol  agent {c['mo_td3']['slo_violation_rate']*100:5.1f}%  "
            f"hpa {c['hpa']['slo_violation_rate']*100:5.1f}%  hpa_tuned {c['hpa_tuned']['slo_violation_rate']*100:5.1f}%  "
            f"static {c['static']['slo_violation_rate']*100:5.1f}%  | cost agent {c['mo_td3']['mean_cost']:.1f} "
            f"hpa_tuned {c['hpa_tuned']['mean_cost']:.1f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 14, 25, 50, 100])
    ap.add_argument("--topologies", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--arm", default="mo_td3_full")
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "scale_results.json"))
    a = ap.parse_args()
    rows = run(a.sizes, a.topologies, a.episodes, a.arm)
    json.dump({"arm": a.arm, "episodes_per_topology": a.episodes, "rows": rows}, open(a.out, "w"), indent=1)
    print("saved", a.out)
