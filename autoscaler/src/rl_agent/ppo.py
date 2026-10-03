"""
ppo.py

A compact, correct PPO implementation for the graph-structured shared policy.

Why not Stable-Baselines3 (which v1 used)? SB3's policies consume a flat
Box observation and emit a fixed-size action vector. Our state is a set of
graph tensors of VARYING size, and our action count varies with the number
of services. Forcing that through SB3 would mean padding to a fixed maximum
node count - which reintroduces exactly the fixed-size limitation the shared
policy exists to remove. ~150 lines of PPO is the honest trade.

Standard PPO: collect rollouts -> GAE advantages -> clipped surrogate loss.
"""

import numpy as np
import torch
import torch.nn as nn

from src import config


def compute_gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def collect_rollout(policy, env, n_episodes):
    """Runs full episodes, storing graph tensors alongside actions."""
    buf = {"obs": [], "actions": [], "log_probs": [], "rewards": [],
           "values": [], "dones": []}
    ep_returns = []

    for _ in range(n_episodes):
        obs = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():
                actions, log_prob, value = policy.act(*obs)
            next_obs, reward, done, _ = env.step(actions)

            buf["obs"].append(obs)
            buf["actions"].append(actions)
            buf["log_probs"].append(log_prob.item())
            buf["rewards"].append(reward)
            buf["values"].append(value.item())
            buf["dones"].append(done)

            obs = next_obs
            ep_ret += reward
        ep_returns.append(ep_ret)

    return buf, float(np.mean(ep_returns))


def train(policy, env, iterations=None, verbose=True, log_every=20, lr=None):
    iterations = iterations or config.PPO_ITERATIONS
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr or config.PPO_LR)
    history = []

    for it in range(iterations):
        buf, mean_return = collect_rollout(policy, env, config.PPO_ROLLOUT_EPISODES)
        history.append(mean_return)

        advantages, returns = compute_gae(
            buf["rewards"], buf["values"], buf["dones"],
            config.PPO_GAMMA, config.PPO_LAMBDA)

        adv_t = torch.tensor(advantages)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns)
        old_lp = torch.tensor(buf["log_probs"])

        for _ in range(config.PPO_EPOCHS):
            new_lps, entropies, values = [], [], []
            for i, obs in enumerate(buf["obs"]):
                lp, ent, val = policy.evaluate(*obs, buf["actions"][i])
                new_lps.append(lp); entropies.append(ent); values.append(val)

            new_lp = torch.stack(new_lps)
            entropy = torch.stack(entropies).mean()
            value_pred = torch.stack(values)

            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - config.PPO_CLIP, 1 + config.PPO_CLIP) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(value_pred, ret_t)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if verbose and (it % log_every == 0 or it == iterations - 1):
            recent = np.mean(history[-log_every:])
            print(f"  iter {it:4d} | mean_episode_return {mean_return:8.3f} "
                  f"| rolling_avg {recent:8.3f}")

    return history
