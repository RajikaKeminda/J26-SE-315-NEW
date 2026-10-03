"""
dual_gcn.py

Two-branch Graph Convolutional Network: one branch reads the STRUCTURAL
graph (SDG), one reads the DYNAMIC latency graph (EHG). Their per-node
outputs are concatenated, giving each service an embedding that encodes
both "who am I connected to" and "where is the pain flowing right now".

A GCN layer is just: average each node's features with its neighbours'
(weighted by the normalized adjacency), then apply a shared linear layer.
Because that shared linear layer is applied per-node, the SAME weights
work for any number of nodes - this is what makes the whole model
topology-size-agnostic.
"""

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # X: (..., N, in_dim), A_hat: (N, N)
        return self.linear(A_hat @ X)


class GCNBranch(nn.Module):
    """Two stacked GCN layers -> each node sees 2 hops of context."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, X, A_hat):
        return self.l2(self.act(self.l1(X, A_hat)), A_hat)


class DualGCN(nn.Module):
    """
    Returns (N, 2 * embed_dim) node embeddings: [structural | latency-flow].

    Set use_ehg=False to run structure-only - used by the ablation study to
    measure how much the EHG branch actually contributes.
    Set use_topology=False to bypass graphs entirely (per-node MLP only) -
    used by the "no topology awareness" ablation arm.
    """

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int,
                 use_ehg: bool = True, use_topology: bool = True):
        super().__init__()
        self.use_ehg = use_ehg
        self.use_topology = use_topology

        if use_topology:
            self.sdg_branch = GCNBranch(in_dim, hidden_dim, embed_dim)
            if use_ehg:
                self.ehg_branch = GCNBranch(in_dim, hidden_dim, embed_dim)
        else:
            # no message passing at all: each node encoded in isolation
            self.plain = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        self.out_dim = embed_dim * (2 if (use_topology and use_ehg) else 1)

    def forward(self, X, A_sdg, A_ehg):
        if not self.use_topology:
            return self.plain(X)
        h_sdg = self.sdg_branch(X, A_sdg)
        if not self.use_ehg:
            return h_sdg
        h_ehg = self.ehg_branch(X, A_ehg)
        return torch.cat([h_sdg, h_ehg], dim=-1)
