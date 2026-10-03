"""
shared_policy.py

THE key architectural change from v1.

v1 problem: the agent's action head had one fixed output slot per NAMED
service (MultiDiscrete([3]*6)). Add a 7th service and the network shape
simply didn't fit - full retrain required.

v2 solution: ONE small actor network is applied independently to EVERY
node's embedding. Same weights, reused per service. The number of services
is never encoded in any weight matrix - it only shows up as a tensor
dimension at runtime.

    node embeddings (N, D)
        -> [shared actor MLP]  applied row-wise
        -> action logits (N, 3)

So N can be 6 today and 7 tomorrow with no architecture change.

Each node also sees a GLOBAL CONTEXT vector (mean of all embeddings +
cluster-level scalars) concatenated onto its own embedding, so a service
can act on "how is the whole system doing" as well as "how am I doing" -
without that, per-node decisions would be blind to the bigger picture.

The critic scores the WHOLE cluster state (one value per state, not per
node), because the reward is a single cluster-wide scalar.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.topology_intel.dual_gcn import DualGCN

GLOBAL_SCALAR_DIM = 3   # [crit_latency_norm, total_replicas_norm, load_variance]


class SharedPolicy(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, embed_dim: int,
                 policy_hidden: int, n_actions: int = 3,
                 use_ehg: bool = True, use_topology: bool = True):
        super().__init__()
        self.encoder = DualGCN(node_feat_dim, hidden_dim, embed_dim,
                                use_ehg=use_ehg, use_topology=use_topology)
        D = self.encoder.out_dim

        # per-node actor input: own embedding + pooled global embedding + global scalars
        actor_in = D + D + GLOBAL_SCALAR_DIM
        self.actor = nn.Sequential(
            nn.Linear(actor_in, policy_hidden), nn.Tanh(),
            nn.Linear(policy_hidden, policy_hidden), nn.Tanh(),
            nn.Linear(policy_hidden, n_actions),
        )

        critic_in = D + GLOBAL_SCALAR_DIM
        self.critic = nn.Sequential(
            nn.Linear(critic_in, policy_hidden), nn.Tanh(),
            nn.Linear(policy_hidden, policy_hidden), nn.Tanh(),
            nn.Linear(policy_hidden, 1),
        )

    # ------------------------------------------------------------------
    def _encode(self, X, A_sdg, A_ehg, globals_vec):
        node_emb = self.encoder(X, A_sdg, A_ehg)        # (N, D)
        pooled = node_emb.mean(dim=0)                    # (D,) - order/size invariant
        n = node_emb.shape[0]
        per_node = torch.cat([
            node_emb,
            pooled.unsqueeze(0).expand(n, -1),
            globals_vec.unsqueeze(0).expand(n, -1),
        ], dim=-1)                                       # (N, 2D + G)
        critic_in = torch.cat([pooled, globals_vec], dim=-1)
        return per_node, critic_in

    def forward(self, X, A_sdg, A_ehg, globals_vec):
        per_node, critic_in = self._encode(X, A_sdg, A_ehg, globals_vec)
        logits = self.actor(per_node)                    # (N, n_actions) - SHARED weights
        value = self.critic(critic_in).squeeze(-1)       # scalar
        return logits, value

    # ------------------------------------------------------------------
    def act(self, X, A_sdg, A_ehg, globals_vec, deterministic: bool = False):
        """Returns (actions (N,), log_prob scalar, value scalar)."""
        logits, value = self.forward(X, A_sdg, A_ehg, globals_vec)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        # joint log-prob across independent per-node action choices
        log_prob = dist.log_prob(actions).sum()
        return actions, log_prob, value

    def evaluate(self, X, A_sdg, A_ehg, globals_vec, actions):
        """Re-scores stored actions during a PPO update."""
        logits, value = self.forward(X, A_sdg, A_ehg, globals_vec)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions).sum(), dist.entropy().sum(), value
