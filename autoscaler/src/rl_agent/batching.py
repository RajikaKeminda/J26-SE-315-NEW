"""
batching.py

Pads variable-size graph observations into batched tensors.

Every sample in a batch has its own number of services N_i. They are padded to the
batch maximum; `mask` (B, N) marks the real nodes. Padded nodes get zero features
and all-zero adjacency, so they exchange no messages with real nodes, and every
pooling/loss in the agent multiplies by `mask`.
"""

import numpy as np
import torch


def pad_obs(obs_list):
    """list of obs dicts (X, A_sdg, A_ehg, g) -> dict of float32 torch tensors."""
    B = len(obs_list)
    n_max = max(o["X"].shape[0] for o in obs_list)
    f = obs_list[0]["X"].shape[1]
    X = np.zeros((B, n_max, f), dtype=np.float32)
    A1 = np.zeros((B, n_max, n_max), dtype=np.float32)
    A2 = np.zeros((B, n_max, n_max), dtype=np.float32)
    mask = np.zeros((B, n_max), dtype=np.float32)
    g = np.zeros((B, obs_list[0]["g"].shape[0]), dtype=np.float32)
    for i, o in enumerate(obs_list):
        n = o["X"].shape[0]
        X[i, :n], A1[i, :n, :n], A2[i, :n, :n], mask[i, :n], g[i] = o["X"], o["A_sdg"], o["A_ehg"], 1.0, o["g"]
    return {"X": torch.from_numpy(X), "A_sdg": torch.from_numpy(A1), "A_ehg": torch.from_numpy(A2),
            "mask": torch.from_numpy(mask), "g": torch.from_numpy(g)}


def pad_actions(idx_list, cont_list, n_max):
    B = len(idx_list)
    idx = np.zeros((B, n_max), dtype=np.int64)
    cont = np.zeros((B, n_max, cont_list[0].shape[1]), dtype=np.float32)
    for i, (a, c) in enumerate(zip(idx_list, cont_list)):
        n = len(a)
        idx[i, :n], cont[i, :n] = a, c
    return torch.from_numpy(idx), torch.from_numpy(cont)
