"""
regime_eval.py

Compare controllers on ONE topology under whatever regime the environment variables define (timing, bounds, budget,
simulator parameters). Used for the Train-Ticket regime:

    set -a; source ../profiles/trainticket.env; set +a
    export SIM_PARAMS_PATH=$PWD/src/simulation/sim_params_trainticket.json
    python -m src.evaluation.regime_eval --topology trainticket-sim \
        --agent original=models/mo_td3_replica_only.pt --agent fine_tuned=models/mo_td3_trainticket.pt

Matched seeds and traffic patterns (normal / burst / diurnal / rampup) for every controller; the HPA is tuned by a grid
search on a held-out seed range, exactly as in the main study.
"""

import argparse
import json

import numpy as np

from src import config
from src.evaluation.evaluate import KEY_METRICS, tune_hpa
from src.evaluation.metrics import evaluate_controller
from src.rl_agent.controllers import RLController, RuleController
from src.rl_agent.mo_td3 import MOTD3Agent
from src.topologies import get_topology


def run(topology_name, agents, episodes, log=print):
    topo = get_topology(topology_name)
    w = np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32)
    best, table = tune_hpa(topo, n_episodes=6)
    log(f"HPA tuned for {topology_name}: cpu {best['cpu']}, target {best['target']}")
    makers = {
        "static": lambda: RuleController("static", topo.services),
        "threshold": lambda: RuleController("threshold", topo.services),
        "hpa (default 60%)": lambda: RuleController("hpa", topo.services),
        "hpa (tuned)": lambda: RuleController("hpa", topo.services, label="hpa_tuned", initial={"cpu": best["cpu"]},
                                              target_utilization=best["target"]),
    }
    for name, path in agents.items():
        agent = MOTD3Agent.load(path)
        makers[name] = lambda a=agent, n=name: RLController(a, topo.services, w=w, name=n)
    out = {}
    for name, mk in makers.items():
        r = evaluate_controller(mk, topo, n_episodes=episodes, w=w)
        out[name] = {k: r[k] for k in (*KEY_METRICS, "mean_replicas") if k in r}
        out[name]["by_pattern"] = {p: {k: v[k] for k in ("slo_violation_rate", "p95_latency", "mean_cost")}
                                   for p, v in r["by_pattern"].items()}
        log(f"  {name:20} viol {out[name]['slo_violation_rate'] * 100:5.1f}%  p95 {out[name]['p95_latency']:7.1f} ms  "
            f"cost {out[name]['mean_cost']:6.2f}  score {out[name]['scalarized']:.3f}")
    return {"topology": topology_name, "services": len(topo.services), "episodes": episodes,
            "hpa_tuned_params": {"cpu": best["cpu"], "target": best["target"]},
            "regime": {k: getattr(config, k) for k in ("STEP_SECONDS", "POD_STARTUP_SECONDS", "CPU_MAX", "MEM_MAX_MB", "MAX_REPLICAS")},
            "slo_ms": float(config.slo_for_depth(5)), "controllers": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="trainticket-sim")
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.topology, dict(x.split("=", 1) for x in a.agent), a.episodes)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)
