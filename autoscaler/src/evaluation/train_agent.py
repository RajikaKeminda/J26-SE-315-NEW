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


def build_topology_options(sizes, target, target_prob, val_sizes):
    """Training topologies and validation set for fine-tuning (see rl_agent.trainer.train_agent)."""
    import numpy as np
    from src.simulation.simulator import random_topology
    from src.topologies import DEMO6, get_topology
    targets = [get_topology(t) for t in (target or [])]
    val = None
    if targets or val_sizes:
        val = [DEMO6] + targets + [random_topology(np.random.default_rng(880_000 + 31 * n), n, n) for n in (val_sizes or [])]
    return {"sizes": tuple(sizes), "target_topologies": targets or None, "target_prob": target_prob if targets else 0.0,
            "val_topologies": val}


def main(arm, episodes, seed, demo_episodes, updates_per_episode, out=None, hp=None, init_from=None, topo_opts=None, curve_out=None,
         record=None):
    torch.set_num_threads(1)           # tiny tensors: threading only adds overhead
    hp = hp or HParams()
    print(f"Training '{arm}' seed={seed} for {episodes} episodes  {ARMS[arm]}", flush=True)
    recorder = None
    if record is not None:                      # store the run, every episode and the learning curve in the database (src/persistence/training.py)
        from src.persistence.training import TrainingRecorder
        recorder = TrainingRecorder(config.DATABASE_URL, arm, seed, episodes, init_from, dataclasses.asdict(hp),
                                    {k: (v if k != "target_topologies" and k != "val_topologies" else [t.name for t in (v or [])])
                                     for k, v in (topo_opts or {}).items()},
                                    cluster_name=record.get("cluster_name"), note=record.get("note"),
                                    target_topologies=(topo_opts or {}).get("target_topologies") or ())
    try:
        agent, best_state, meta = train_agent(arm, episodes=episodes, seed=seed,
                                              demo_episodes=demo_episodes,
                                              updates_per_episode=updates_per_episode, hp=hp,
                                              log=lambda *a: print(*a, flush=True), init_from=init_from,
                                              recorder=recorder, **(topo_opts or {}))
    except BaseException:
        if recorder:
            recorder.finish(status="failed")
        raise
    meta["hparams"] = dataclasses.asdict(hp)
    path = out or checkpoint_path(arm, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = best_state or agent.state()
    state.update({k: v for k, v in meta.items() if k != "history"})
    torch.save(state, path)
    curve = curve_out or os.path.join(config.RESULTS_DIR, "training", f"{arm}_s{seed}.json")
    os.makedirs(os.path.dirname(curve), exist_ok=True)
    with open(curve, "w") as f:
        json.dump(meta, f, indent=2)
    if recorder:
        recorder.finish(path, meta)
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
    ap.add_argument("--init-from", default=None, help="fine-tune: warm-start from this checkpoint")
    ap.add_argument("--sizes", type=int, nargs=2, default=[4, 12], metavar=("MIN", "MAX"),
                    help="service-count range of the random training topologies")
    ap.add_argument("--target", nargs="*", default=[], help="named topologies mixed into training and validation, e.g. trainticket-sim")
    ap.add_argument("--target-prob", type=float, default=0.4, help="share of training episodes on the target topologies")
    ap.add_argument("--val-sizes", type=int, nargs="*", default=[], help="add random validation topologies of these sizes")
    ap.add_argument("--record", action="store_true", help="store this run (settings, every episode, trajectories, learning curve) in the database; needs AUTOSCALER_DATABASE_URL")
    ap.add_argument("--cluster-name", default=None, help="with --record: the cluster whose measurements the regime comes from")
    ap.add_argument("--note", default=None, help="with --record: a free-text note kept with the run")
    a = ap.parse_args()
    if a.record and not config.DATABASE_URL:
        ap.error("--record needs AUTOSCALER_DATABASE_URL")
    main(a.arm, a.episodes, a.seed, a.demo_episodes, a.updates_per_episode, a.out, parse_hparams(a.hp), a.init_from,
         build_topology_options(a.sizes, a.target, a.target_prob, a.val_sizes), a.curve_out,
         {"cluster_name": a.cluster_name, "note": a.note} if a.record else None)
