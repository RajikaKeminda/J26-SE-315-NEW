"""
train_agent.py

Trains ONE arm of the study (the full MO-TD3 agent or an ablation) on the calibrated
simulator and writes the best-validation checkpoint plus the learning curve.

    python -m src.evaluation.train_agent mo_td3_full --episodes 1500 --seed 0
    python -m src.evaluation.train_agent mo_td3_no_topology --episodes 1500 --seed 0

Outputs
    models/<arm>.pt                    (seed 0)   or   models/seeds/<arm>_s<seed>.pt
    results/training/<arm>_s<seed>.json  validation learning curve
"""

import argparse
import json
import os

import torch

from src import config
import dataclasses

from src.rl_agent.trainer import ARMS, HParams, train_agent


def checkpoint_path(arm, seed, primary_seed=0):
    if seed == primary_seed:
        return os.path.join(config.MODELS_DIR, f"{arm}.pt")
    return os.path.join(config.MODELS_DIR, "seeds", f"{arm}_s{seed}.pt")


def parse_hparams(pairs):
    hp, fields = HParams(), {f.name: f for f in dataclasses.fields(HParams)}
    for pair in pairs or []:
        k, v = pair.split("=", 1)
        if k not in fields:
            raise SystemExit(f"unknown hyper-parameter {k!r}; choose from {sorted(fields)}")
        cur = getattr(hp, k)
        setattr(hp, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    return hp


def main(arm, episodes, seed, demo_episodes, updates_per_episode, out=None, hp=None):
    torch.set_num_threads(1)           # tiny tensors: threading only adds overhead
    hp = hp or HParams()
    print(f"Training '{arm}' seed={seed} for {episodes} episodes  {ARMS[arm]}", flush=True)
    agent, best_state, meta = train_agent(arm, episodes=episodes, seed=seed,
                                          demo_episodes=demo_episodes,
                                          updates_per_episode=updates_per_episode, hp=hp,
                                          log=lambda *a: print(*a, flush=True))
    meta["hparams"] = dataclasses.asdict(hp)
    path = out or checkpoint_path(arm, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = best_state or agent.state()
    state.update({k: v for k, v in meta.items() if k != "history"})
    torch.save(state, path)
    curve = os.path.join(config.RESULTS_DIR, "training", f"{arm}_s{seed}.json")
    os.makedirs(os.path.dirname(curve), exist_ok=True)
    with open(curve, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved -> {path}   (best validation score {meta['best_val_scalarized']:.3f})")
    print(f"Learning curve -> {curve}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=list(ARMS))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-episodes", type=int, default=400)
    ap.add_argument("--updates-per-episode", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hp", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override training hyper-parameters, e.g. --hp n_step=1 bc_floor=0.2")
    ap.add_argument("--curve-out", default=None, help="learning-curve JSON path (default results/training/)")
    a = ap.parse_args()
    main(a.arm, a.episodes, a.seed, a.demo_episodes, a.updates_per_episode, a.out, parse_hparams(a.hp))
