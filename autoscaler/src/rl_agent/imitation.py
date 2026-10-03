"""
imitation.py

Warm-start for the shared policy: behaviour-cloning from the HPA rule.

WHY: from random initialisation, PPO on this task converges to "scale every
service to the maximum" - it is the safest way to dodge the large SLO-violation
penalty, and with independent per-service actions sharing one cluster-wide
reward, credit assignment is too weak to discover the cheaper policy within a
laptop-scale training budget (see README, "Honest limitations" and
results/pareto_results.json: even a 3x larger cost weight does not escape it).

Starting from a policy that already behaves like HPA (the industry baseline)
puts PPO in the right basin; PPO fine-tuning then has to IMPROVE on HPA rather
than discover reasonable behaviour from scratch.
"""

import numpy as np
import torch
import torch.nn as nn

from src import config
from src.baselines.rule_based import HPABaseline

_DELTA_TO_INDEX = {d: i for i, d in enumerate(config.REPLICA_DELTA_OPTIONS)}


def collect_demonstrations(env, n_episodes):
    """Runs HPA as the teacher in `env`; returns [(obs, action_tensor)]."""
    demos = []
    for _ in range(n_episodes):
        obs = env.reset()
        teacher = HPABaseline(env.services)
        done = False
        while not done:
            metrics = env._last[0]
            deltas = teacher.decide(metrics)
            actions = torch.tensor([_DELTA_TO_INDEX[deltas[s]] for s in env.services])
            demos.append((obs, actions))
            obs, _, done, _ = env.step(actions)
    return demos


def pretrain(policy, demos, epochs=8, lr=1e-3, seed=0, verbose=True):
    """Cross-entropy of the actor against the teacher's actions. Returns loss history."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history = []
    for ep in range(epochs):
        order = rng.permutation(len(demos))
        total, correct, n = 0.0, 0, 0
        for i in order:
            (X, A_sdg, A_ehg, g), actions = demos[i]
            logits, _ = policy(X, A_sdg, A_ehg, g)
            loss = ce(logits, actions)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total += loss.item()
            correct += int((logits.argmax(-1) == actions).sum())
            n += len(actions)
        history.append(total / len(demos))
        if verbose:
            print(f"  imitation epoch {ep}: loss {history[-1]:.3f}  action-match {correct / n * 100:.1f}%")
    return history
