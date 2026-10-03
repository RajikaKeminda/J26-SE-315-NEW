"""
train_arm.py

Trains ONE ablation arm and saves it to models/<arm>.pt.
Split out from ablation.py so arms can be trained independently (and
resumed/re-run individually) rather than in one long blocking job.

Usage:
    python -m src.evaluation.train_arm rl_full --iterations 55
"""

import argparse
import os

import torch

from src import config
from src.rl_agent.env import AutoscalerEnv
from src.rl_agent.shared_policy import SharedPolicy
from src.rl_agent.ppo import train
from src.rl_agent.imitation import collect_demonstrations, pretrain

# arm name -> (use_topology, use_ehg, single_objective)
ARMS = {
    "rl_full":        (True,  True,  False),
    "rl_sdg_only":    (True,  False, False),
    "rl_no_topology": (False, False, False),
    "rl_single_obj":  (True,  True,  True),
    # same architecture/reward as rl_full, but behaviour-cloned from HPA before PPO
    # fine-tuning (see rl_agent/imitation.py for why)
    "rl_hpa_init":    (True,  True,  False),
}
WARM_START = {"rl_hpa_init"}
IMITATION_EPISODES = 40
IMITATION_EPOCHS = 6
FINETUNE_LR_SCALE = 0.3     # PPO learning-rate multiplier after warm start (don't wash it out)


def train_policy(arm: str, iterations: int, seed: int = 0, verbose: bool = True):
    """Trains one arm and returns (policy, history, flags). No file I/O."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    use_topology, use_ehg, single_obj = ARMS[arm]

    if verbose:
        print(f"Training '{arm}'  (topology={use_topology}, ehg={use_ehg}, single_objective={single_obj})")
    torch.manual_seed(seed)
    env = AutoscalerEnv(use_random_topology=True, seed=seed, single_objective=single_obj)
    policy = SharedPolicy(
        config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM, config.GCN_EMBED_DIM,
        config.POLICY_HIDDEN, use_ehg=use_ehg, use_topology=use_topology)

    lr = None
    if arm in WARM_START:
        if verbose:
            print(f"  warm start: cloning HPA on {IMITATION_EPISODES} episodes")
        demos = collect_demonstrations(env, IMITATION_EPISODES)
        pretrain(policy, demos, epochs=IMITATION_EPOCHS, seed=seed, verbose=verbose)
        lr = config.PPO_LR * FINETUNE_LR_SCALE

    history = train(policy, env, iterations=iterations,
                    log_every=max(1, iterations // 5), verbose=verbose, lr=lr)
    return policy, history, {"use_topology": use_topology, "use_ehg": use_ehg}


def save_policy(policy, history, flags, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"state_dict": policy.state_dict(), "history": history, **flags}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def main(arm: str, iterations: int, seed: int = 0, out: str | None = None):
    policy, history, flags = train_policy(arm, iterations, seed)
    path = out or os.path.join(config.MODELS_DIR, f"{arm}.pt")
    save_policy(policy, history, flags, path)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--iterations", type=int, default=config.PPO_ITERATIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="checkpoint path (default models/<arm>.pt)")
    args = ap.parse_args()
    main(args.arm, args.iterations, args.seed, args.out)
