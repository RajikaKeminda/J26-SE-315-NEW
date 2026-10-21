"""
feature_ablation.py

Does feeding the agent the MEASURED critical set (topology_intel/crisp.py) instead of the depth-first chain improve its scaling decisions?
Each checkpoint is scored on the same topologies, traffic patterns and seeds, under the feature source it was trained with:

    python -m src.evaluation.feature_ablation --ckpt base=models/mo_td3_full.pt --ckpt control=models/mo_td3_ft_control.pt   # dfs inputs
    CRITICAL_PATH_SOURCE=crisp python -m src.evaluation.feature_ablation --ckpt crisp=models/mo_td3_ft_crisp.pt               # critical-set inputs

The topologies: the deployed 6-service app, the 14-service Train-Ticket shape, 16 held-out random topologies (4-12 services, never trained on),
and random topologies of 25 and 50 services. Lower is better for every column.
"""

import argparse
import json

import numpy as np

from src import config
from src.evaluation.evaluate import heldout_topologies
from src.evaluation.metrics import evaluate_controller
from src.rl_agent.controllers import RLController
from src.rl_agent.mo_td3 import MOTD3Agent
from src.simulation.simulator import random_topology
from src.topologies import DEMO6, TRAINTICKET14

KEYS = ("slo_violation_rate", "p95_latency", "mean_cost", "scalarized")


def topology_sets():
    return {
        "demo6": [DEMO6],
        "trainticket14": [TRAINTICKET14],
        "heldout_4_12": heldout_topologies(16),
        "random_25": [random_topology(np.random.default_rng(770_025 + i), 25, 25) for i in range(3)],
        "random_50": [random_topology(np.random.default_rng(770_050 + i), 50, 50) for i in range(3)],
    }


def run(ckpts, episodes=8):
    w = np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32)
    out = {"feature_source": config.CRITICAL_PATH_SOURCE, "episodes_per_topology": episodes, "agents": {}}
    for name, path in ckpts.items():
        agent = MOTD3Agent.load(path)
        res = {}
        for tname, topos in topology_sets().items():
            runs = [evaluate_controller(lambda t=t: RLController(agent, t.services, w=w, name=name), t, n_episodes=episodes,
                                        seed0=31_000 + 11 * i) for i, t in enumerate(topos)]
            res[tname] = {k: float(np.mean([r[k] for r in runs])) for k in KEYS}
        out["agents"][name] = res
        print(f"[{config.CRITICAL_PATH_SOURCE}] {name:10} " + "  ".join(
            f"{t}: viol {res[t]['slo_violation_rate'] * 100:4.1f}% score {res[t]['scalarized']:.3f}" for t in res), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(dict(x.split("=", 1) for x in a.ckpt), a.episodes)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
