"""
replay.py

Episodic replay buffer for graph-structured transitions of VARYING size.

Each episode is stored once as a list of T+1 observations (so s' is just the next
entry, not a second copy), the executed actions, the reward VECTOR, and the
preference vector the behaviour policy was following. Demonstration episodes (from
the HPA/VPA/threshold controllers) live in a protected buffer that is never evicted
and are flagged so the actor's behaviour-cloning term can apply to them only.
"""

from collections import deque

import numpy as np
import torch

from src.rl_agent.batching import pad_actions, pad_obs

_KEEP = ("X", "A_sdg", "A_ehg", "g")


def _slim(obs):
    return {k: obs[k] for k in _KEEP}


class EpisodeReplay:
    def __init__(self, capacity_episodes=4000, seed=0):
        self.agent_eps = deque(maxlen=capacity_episodes)
        self.demo_eps = []
        self.rng = np.random.default_rng(seed)

    def add_episode(self, obs_list, idx, cont, rewards, w, demo=False):
        ep = {"obs": [_slim(o) for o in obs_list],
              "idx": [np.asarray(a, dtype=np.int64) for a in idx],
              "cont": [np.asarray(c, dtype=np.float32) for c in cont],
              "r": np.asarray(rewards, dtype=np.float32),
              "w": np.asarray(w, dtype=np.float32), "demo": bool(demo)}
        (self.demo_eps if demo else self.agent_eps).append(ep)

    @property
    def n_transitions(self):
        return sum(len(e["idx"]) for e in self.agent_eps) + sum(len(e["idx"]) for e in self.demo_eps)

    def _pick(self, pool, n):
        eps = [pool[i] for i in self.rng.integers(0, len(pool), n)]
        return [(e, int(self.rng.integers(0, len(e["idx"])))) for e in eps]

    def sample(self, batch_size, demo_fraction=0.0, n_step=1, gamma=0.95):
        """Returns n-step transitions: r = sum_{k<h} gamma^k r_{t+k}, bootstrap from s_{t+h}
        with discount gamma^h (h = min(n_step, steps left in the episode))."""
        n_demo = int(round(batch_size * demo_fraction)) if self.demo_eps else 0
        n_agent = batch_size - n_demo
        picks = []
        if n_agent and self.agent_eps:
            picks += self._pick(self.agent_eps, n_agent)
        elif n_agent and self.demo_eps:                 # nothing but demos so far
            picks += self._pick(self.demo_eps, n_agent)
        if n_demo:
            picks += self._pick(self.demo_eps, n_demo)

        horizons = [min(n_step, len(e["idx"]) - t) for e, t in picks]
        returns = np.stack([sum(gamma ** k * e["r"][t + k] for k in range(h))
                            for (e, t), h in zip(picks, horizons)]).astype(np.float32)
        obs = pad_obs([e["obs"][t] for e, t in picks])
        nxt = pad_obs([e["obs"][t + h] for (e, t), h in zip(picks, horizons)])
        n_max = obs["X"].shape[1]
        idx, cont = pad_actions([e["idx"][t] for e, t in picks],
                                [e["cont"][t] for e, t in picks], n_max)
        return {
            "obs": obs, "next_obs": nxt, "idx": idx, "cont": cont,
            "r": torch.from_numpy(returns),
            "disc": torch.tensor([[gamma ** h] for h in horizons], dtype=torch.float32),
            "w": torch.from_numpy(np.stack([e["w"] for e, t in picks])),
            "demo": torch.tensor([1.0 if e["demo"] else 0.0 for e, t in picks]),
        }
