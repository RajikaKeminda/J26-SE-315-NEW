"""
pareto_regime.py

The trade-off front of ONE checkpoint (swept over preferences at inference time) against the HPA tuning sweep, on one
topology under the regime defined by the environment variables (see regime_eval.py). Same measures as pareto.py:
2-D hypervolume in (SLO-violation rate, cost / budget), and how many points of each family are Pareto-optimal.

    set -a; source ../profiles/trainticket.env; set +a
    export SIM_PARAMS_PATH=$PWD/src/simulation/sim_params_trainticket.json MAX_REPLICAS=6 BUDGET_PER_SERVICE=3.0
    python -m src.evaluation.pareto_regime --topology trainticket-sim --ckpt models/mo_td3_trainticket_cap6.pt
"""

import argparse
import json

from src import config
from src.evaluation.pareto import baseline_points, flag, hypervolume_2d, sweep
from src.rl_agent.mo_td3 import MOTD3Agent
from src.topologies import get_topology


def run(topology_name, ckpt, episodes, grid_step, log=print):
    topo = get_topology(topology_name)
    budget = config.BUDGET_PER_SERVICE * len(topo.services)
    agent = MOTD3Agent.load(ckpt)
    rl = sweep(agent, topo, episodes, grid_step, include_baselines=False, arm="agent", log=log)
    base = baseline_points(topo, episodes)
    allp = flag([dict(p) for p in rl] + [dict(p) for p in base])
    hpa = [p for p in base if p["family"] == "hpa"]
    res = {"topology": topology_name, "checkpoint": ckpt, "episodes": episodes, "budget": budget,
           "hypervolume": {"agent": hypervolume_2d(rl, budget), "hpa_sweep": hypervolume_2d(hpa, budget)},
           "pareto_optimal": {fam: sum(p["pareto_optimal"] for p in allp if p["family"] == fam) for fam in ("mo_td3", "hpa", "vpa", "threshold", "static")},
           "points": allp}
    log(f"hypervolume (larger is better): agent {res['hypervolume']['agent']:.3f}  vs  HPA tuning sweep {res['hypervolume']['hpa_sweep']:.3f}")
    log(f"Pareto-optimal points: {res['pareto_optimal']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="trainticket-sim")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--grid-step", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(a.topology, a.ckpt, a.episodes, a.grid_step)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
        print("saved", a.out)
