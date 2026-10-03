"""
dual_gcn.py

Two-branch Graph Convolutional Network: one branch reads the STRUCTURAL graph
(SDG), one reads the DYNAMIC latency graph (EHG). Their per-node outputs are
concatenated, giving each service an embedding that encodes both "who am I
connected to" and "where is the latency pain flowing right now".

Per the proposal each branch is a 2-layer GCN with 128 hidden units, and the two
32-d branch outputs form a 64-d embedding (EMBEDDING_DIM).

A GCN layer is: average each node's features with its neighbours' (weighted by the
normalised adjacency), then apply a shared linear layer. Because that linear layer
is shared across nodes, the SAME weights work for any number of nodes - which is
what makes the whole model topology-size-agnostic. Inputs may be batched and padded:
X (B,N,F), A (B,N,N); padded nodes have all-zero adjacency rows and are masked out
by the caller. Unbatched (N,F)/(N,N) inputs are accepted too.
"""

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.matmul(A_hat, X))


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
    Returns (..., N, out_dim) node embeddings: [structural | latency-flow].

    Ablation switches:
      use_ehg=False       structure-only (single SDG branch, same total embedding size)
      use_topology=False  no message passing at all: each node encoded in isolation by an
                          MLP of matching width, so the ablation changes ONLY whether
                          neighbours are visible, not model capacity.
    """

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int,
                 use_ehg: bool = True, use_topology: bool = True):
        super().__init__()
        self.use_ehg = use_ehg
        self.use_topology = use_topology
        self.out_dim = embed_dim * 2          # constant across ablation arms

        if use_topology:
            branch_dim = embed_dim if use_ehg else embed_dim * 2
            self.sdg_branch = GCNBranch(in_dim, hidden_dim, branch_dim)
            if use_ehg:
                self.ehg_branch = GCNBranch(in_dim, hidden_dim, branch_dim)
        else:
            self.plain = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
            )

    def forward(self, X, A_sdg, A_ehg):
        squeeze = X.dim() == 2
        if squeeze:
            X, A_sdg, A_ehg = X.unsqueeze(0), A_sdg.unsqueeze(0), A_ehg.unsqueeze(0)
        if not self.use_topology:
            h = self.plain(X)
        else:
            h = self.sdg_branch(X, A_sdg)
            if self.use_ehg:
                h = torch.cat([h, self.ehg_branch(X, A_ehg)], dim=-1)
        return h.squeeze(0) if squeeze else h
