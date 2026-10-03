"""
mo_td3.py

MO-TD3: a Multi-Objective Twin Delayed DDPG agent with a HYBRID action space and a
topology-agnostic, graph-conditioned policy - the agent specified in the project
proposal (sub-component 3).

                 +-----------------------------------------------+
   X, A_sdg,A_ehg|  DualGCN (2 layers, 128 hidden, 64-d embedding)|--> h_i per service
                 +-----------------------------------------------+
   h_i, mean(h), cluster scalars g, preference w
        --> ACTOR (weights shared by every service) -->
              continuous:  CPU change, memory change    in [-1, 1]
              discrete:    replica delta logits         for {-3,...,+3}
   (s, a, w) --> TWIN CRITICS (deep-set over services) --> Q vector, one per objective

What is "multi-objective" here. The reward is a VECTOR (SLO, cost, balance). Each
critic outputs one Q-value per objective; the actor is conditioned on a preference
vector w and ascends  w . Q(s, pi(s | w), w).  One trained network therefore covers
the whole latency/cost trade-off curve - the Pareto front is traced by sweeping w at
INFERENCE time (evaluation/pareto.py) rather than re-training per weighting.

What is "hybrid". CPU/memory are continuous (deterministic-policy-gradient as in
TD3); replica deltas are discrete. The discrete head is trained with a
straight-through relaxation: the critic always sees a hard one-hot (identical to what
is executed) while the gradient flows through the softmax. Exploration is Gaussian
noise on the continuous part and epsilon-random on the discrete part.

TD3 ingredients: twin critics (min of two), target-policy smoothing, delayed actor
updates, Polyak-averaged targets. Credit assignment across services comes from the
critic: it is a deep set over (embedding_i, action_i) pairs, so dQ/da_i tells each
service how ITS action moved the shared objective - something a scalar-reward policy
gradient (the PPO of the earlier version) could not do, and the reason that version
collapsed to "scale everything to the maximum".

Two ways to realise the discrete replica head (`replica_head`):
  "round"        (default, used for every shipped result) a third continuous output a in [-1, 1]
                 executed as round(3a) in {-3..+3}; the critic is always fed the EXECUTED value
                 a = delta/3. Plain deterministic-policy gradient end to end - no discrete
                 extrapolation of the critic - and the usual way TD3 is applied to integer actions.
  "categorical"  7 logits with the straight-through relaxation described above. Kept as the
                 `mo_td3_categorical` ablation: it scored worse on every evaluation set (one seed).

Behaviour cloning on demonstrations (TD3+BC, optional - the `mo_td3_demo*` arms): demonstration
episodes from the HPA/threshold controllers are mixed into the replay buffer, and the actor gets a
decaying cloning term on them. This puts the search where reasonable behaviour lives
and is itself ablated: the `mo_td3_demo*` arms are the same models WITH demonstrations, and the
study found them worse than pure RL (see the README).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.rl_agent.batching import pad_obs
from src.topology_intel.dual_gcn import DualGCN

K = config.N_OBJECTIVES
G = config.GLOBAL_FEATURE_DIM
N_DISC = config.N_REPLICA_ACTIONS
HEADS = ("categorical", "round")
MAX_DELTA = config.REPLICA_DELTA_OPTIONS[-1]


def n_continuous(replica_head):
    """Continuous outputs per service: CPU, memory (+ the replica delta in 'round' mode)."""
    return 3 if replica_head == "round" else 2


def n_discrete(replica_head):
    return 0 if replica_head == "round" else N_DISC


def delta_index(a):
    """Continuous replica output in [-1, 1] -> index into REPLICA_DELTA_OPTIONS (round(3a) + 3)."""
    return np.clip(np.rint(np.asarray(a) * MAX_DELTA), -MAX_DELTA, MAX_DELTA).astype(np.int64) + MAX_DELTA


def _mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


def _masked_mean(h, mask):
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def _context(h, mask, g, w):
    """Cluster-level context shared by every service: pooled embedding + scalars + preference."""
    return torch.cat([_masked_mean(h, mask), g, w], dim=-1)


class MOActor(nn.Module):
    def __init__(self, use_topology=True, use_ehg=True, replica_only=False, replica_head="round"):
        super().__init__()
        self.replica_only = replica_only
        self.replica_head = replica_head
        n_cont = n_continuous(replica_head)
        self.encoder = DualGCN(config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM,
                               config.GCN_EMBED_DIM, use_ehg=use_ehg, use_topology=use_topology)
        D = self.encoder.out_dim
        self.trunk = _mlp(D + D + G + K, config.POLICY_HIDDEN, config.POLICY_HIDDEN, layers=2)
        self.cont_head = nn.Linear(config.POLICY_HIDDEN, n_cont)
        self.disc_head = nn.Linear(config.POLICY_HIDDEN, N_DISC) if replica_head == "categorical" else None
        # start as a sensible static sizing (0.5 core, 128 MiB) with "no replica change"
        # preferred, so early exploration perturbs a working configuration
        from src.rl_agent.env import resource_to_u
        nn.init.uniform_(self.cont_head.weight, -3e-3, 3e-3)
        init = [resource_to_u(0.5, config.CPU_MIN, config.CPU_MAX),
                resource_to_u(128.0, config.MEM_MIN_MB, config.MEM_MAX_MB)]
        if replica_head == "round":
            init.append(0.0)                                          # replica delta 0
        with torch.no_grad():
            self.cont_head.bias.copy_(torch.atanh(torch.tensor(init) * 0.999))
        if self.disc_head is not None:
            nn.init.uniform_(self.disc_head.weight, -3e-3, 3e-3)
            with torch.no_grad():
                self.disc_head.bias.zero_()
                self.disc_head.bias[config.NO_OP_INDEX] = 1.0

    def embed(self, obs):
        return self.encoder(obs["X"], obs["A_sdg"], obs["A_ehg"])

    def forward(self, obs, w):
        h = self.embed(obs)                                           # (B,N,D)
        ctx = _context(h, obs["mask"], obs["g"], w)                   # (B, D+G+K)
        z = self.trunk(torch.cat([h, ctx.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1))
        cont = torch.tanh(self.cont_head(z))
        if self.replica_only:                    # no resource actions: CPU/memory outputs are ignored
            cont = torch.cat([cont[..., :2] * 0.0, cont[..., 2:]], dim=-1)
        return cont, (self.disc_head(z) if self.disc_head is not None else None)


class MOCritic(nn.Module):
    """Q(s, a, w) -> (B, K). Deep-set over services keeps it size-agnostic."""

    def __init__(self, use_topology=True, use_ehg=True, replica_head="round"):
        super().__init__()
        self.encoder = DualGCN(config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM,
                               config.GCN_EMBED_DIM, use_ehg=use_ehg, use_topology=use_topology)
        D = self.encoder.out_dim
        ctx_dim = D + G + K
        H = config.POLICY_HIDDEN
        self.phi = _mlp(D + n_continuous(replica_head) + n_discrete(replica_head) + ctx_dim, H, H, layers=2)
        self.head = _mlp(H + ctx_dim, H, K, layers=2)

    def forward(self, obs, cont, onehot, w):
        h = self.encoder(obs["X"], obs["A_sdg"], obs["A_ehg"])
        ctx = _context(h, obs["mask"], obs["g"], w)
        n = h.shape[1]
        parts = [h, cont] + ([onehot] if onehot is not None else []) + [ctx.unsqueeze(1).expand(-1, n, -1)]
        per_node = self.phi(torch.cat(parts, dim=-1))
        pooled = _masked_mean(per_node, obs["mask"])
        return self.head(torch.cat([pooled, ctx], dim=-1))


def _st_onehot(logits):
    """Hard one-hot forward, softmax gradient backward (straight-through)."""
    soft = F.softmax(logits, dim=-1)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(soft.dtype)
    return hard + soft - soft.detach()


class MOTD3Agent:
    def __init__(self, use_topology=True, use_ehg=True, replica_only=False, seed=0,
                 replica_head="round"):
        if replica_head not in HEADS:
            raise ValueError(f"replica_head must be one of {HEADS}")
        torch.manual_seed(seed)
        self.replica_head = replica_head
        self.n_cont = n_continuous(replica_head)
        self.flags = {"use_topology": use_topology, "use_ehg": use_ehg, "replica_only": replica_only,
                      "replica_head": replica_head}
        self.actor = MOActor(use_topology, use_ehg, replica_only, replica_head)
        self.critic1 = MOCritic(use_topology, use_ehg, replica_head)
        self.critic2 = MOCritic(use_topology, use_ehg, replica_head)
        self.actor_t, self.critic1_t, self.critic2_t = (
            copy.deepcopy(self.actor), copy.deepcopy(self.critic1), copy.deepcopy(self.critic2))
        for net in (self.actor_t, self.critic1_t, self.critic2_t):
            for p in net.parameters():
                p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.TD3_ACTOR_LR)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=config.TD3_CRITIC_LR)
        self.updates = 0
        self.rng = np.random.default_rng(seed)
        # The Polyak-averaged target actor is a smoothed version of the online actor and is much
        # less jumpy from one update to the next; it is what gets validated and deployed.
        self.use_target_for_act = False

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs, w, noise_std=0.0, epsilon=0.0):
        """One observation -> (replica index (N,), continuous (N, n_cont)).

        noise_std / epsilon > 0 only while exploring; deployment is deterministic."""
        net = self.actor_t if self.use_target_for_act else self.actor
        net.eval()
        batch = pad_obs([obs])
        wt = torch.as_tensor(np.asarray(w, dtype=np.float32)).unsqueeze(0)
        cont, logits = net(batch, wt)
        n = obs["X"].shape[0]
        cont = cont[0, :n].numpy().copy()
        if self.replica_head == "round":
            if noise_std > 0:
                sigma = np.array([noise_std, noise_std, 2.0 * noise_std])      # replica deltas need wider noise
                cont = np.clip(cont + self.rng.normal(0, 1, cont.shape) * sigma, -1, 1).astype(np.float32)
            if epsilon > 0:
                explore = self.rng.random(n) < epsilon
                if explore.any():
                    probs = np.array([1, 2, 4, 8, 4, 2, 1], dtype=float)
                    probs /= probs.sum()
                    deltas = self.rng.choice(np.arange(-MAX_DELTA, MAX_DELTA + 1), size=int(explore.sum()), p=probs)
                    cont[explore, 2] = deltas / MAX_DELTA
            return delta_index(cont[:, 2]), cont.astype(np.float32)

        idx = logits[0, :n].argmax(-1).numpy().copy()
        if noise_std > 0:
            cont = np.clip(cont + self.rng.normal(0, noise_std, cont.shape), -1, 1).astype(np.float32)
        if epsilon > 0:
            explore = self.rng.random(n) < epsilon
            if explore.any():
                # triangular weights: small moves are tried far more often than +-3
                probs = np.array([1, 2, 4, 8, 4, 2, 1], dtype=float)
                probs /= probs.sum()
                idx[explore] = self.rng.choice(N_DISC, size=int(explore.sum()), p=probs)
        return idx, cont

    @torch.no_grad()
    def graph_embedding(self, obs):
        """The proposal's sub-component-2 output: a 64-d embedding of the WHOLE application
        (mean of the per-service GCN embeddings) and the critical-path services."""
        self.actor.eval()
        h = self.actor.embed(pad_obs([obs]))                           # (1, N, 64)
        pooled = _masked_mean(h, pad_obs([obs])["mask"])[0].numpy()
        return pooled, list(obs["critical_path"])

    # ------------------------------------------------------------------ learning
    def _split_action(self, cont, idx):
        """Replay stores [cpu, mem, delta/3] and the replica index; each head uses what it needs."""
        if self.replica_head == "round":
            return cont[..., :3], None
        return cont[..., :2], F.one_hot(idx, N_DISC).float()

    def update(self, batch, bc_weight=0.0):
        obs, nxt = batch["obs"], batch["next_obs"]
        a_cont, a_onehot = self._split_action(batch["cont"], batch["idx"])
        a_idx = batch["idx"]
        r, w, demo = batch["r"], batch["w"], batch["demo"]
        self.actor.train()
        info = {}
        noise = lambda t: torch.clamp(torch.randn_like(t) * config.TD3_TARGET_NOISE,
                                      -config.TD3_NOISE_CLIP, config.TD3_NOISE_CLIP)

        # ---- critic ----
        with torch.no_grad():
            c2, l2 = self.actor_t(nxt, w)
            c2 = (c2 + noise(c2)).clamp(-1, 1)                                  # target-policy smoothing
            oh2 = None
            if l2 is not None:
                oh2 = F.one_hot((l2 + noise(l2)).argmax(-1), N_DISC).float()
            q_next = torch.min(self.critic1_t(nxt, c2, oh2, w), self.critic2_t(nxt, c2, oh2, w))
            target = r + batch.get("disc", torch.full((r.shape[0], 1), config.TD3_GAMMA)) * q_next
        q1, q2 = self.critic1(obs, a_cont, a_onehot, w), self.critic2(obs, a_cont, a_onehot, w)
        critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), 5.0)
        self.critic_opt.step()
        info["critic_loss"] = float(critic_loss.detach())
        info["q_mean"] = float(q1.detach().mean())

        # ---- delayed actor ----
        self.updates += 1
        if self.updates % config.TD3_POLICY_DELAY == 0:
            cont, logits = self.actor(obs, w)
            if logits is not None:
                q = self.critic1(obs, cont, _st_onehot(logits), w)              # (B,K)
                actor_loss = -(q * w).sum(-1).mean() + 1e-3 * (logits ** 2).mean()
            else:
                q = self.critic1(obs, cont, None, w)
                actor_loss = -(q * w).sum(-1).mean()
            if bc_weight > 0 and demo.sum() > 0:
                m = obs["mask"] * demo.unsqueeze(-1)
                denom = m.sum().clamp(min=1.0)
                bc = (((cont - a_cont) ** 2).sum(-1) * m).sum() / denom
                if logits is not None:
                    bc = bc + (F.cross_entropy(logits.reshape(-1, N_DISC), a_idx.reshape(-1),
                                               reduction="none").reshape(m.shape) * m).sum() / denom
                actor_loss = actor_loss + bc_weight * bc
                info["bc_loss"] = float(bc.detach())
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_opt.step()
            info["actor_loss"] = float(actor_loss.detach())
            self._polyak()
        return info

    @torch.no_grad()
    def _polyak(self):
        tau = config.TD3_TAU
        for net, tgt in ((self.actor, self.actor_t), (self.critic1, self.critic1_t),
                         (self.critic2, self.critic2_t)):
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.mul_(1 - tau).add_(tau * p)

    # ------------------------------------------------------------------ persistence
    def state(self, extra=None, deploy_target=True):
        """Checkpoint. The deployed actor is the Polyak-averaged target actor (see
        use_target_for_act); the raw online actor is kept as `actor_online`."""
        actor = (self.actor_t if deploy_target else self.actor).state_dict()
        blob = {"actor": {k: v.clone() for k, v in actor.items()},
                "actor_online": {k: v.clone() for k, v in self.actor.state_dict().items()},
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(), "flags": dict(self.flags), "action_mapping": "log-absolute",
                "algo": "mo_td3", "n_objectives": K,
                "node_feature_dim": config.NODE_FEATURE_DIM}
        if extra:
            blob.update(extra)
        return blob

    def save(self, path, extra=None):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state(extra), path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, weights_only=False)
        if blob.get("algo") != "mo_td3":
            raise ValueError(f"{path} is not an MO-TD3 checkpoint (algo={blob.get('algo')!r}) - "
                             "retrain with `python -m src.evaluation.train_agent`")
        f = blob["flags"]
        agent = cls(f["use_topology"], f["use_ehg"], f["replica_only"],
                    replica_head=f.get("replica_head", "categorical"))   # pre-"round" checkpoints are categorical
        agent.actor.load_state_dict(blob["actor"])
        agent.critic1.load_state_dict(blob["critic1"])
        agent.critic2.load_state_dict(blob["critic2"])
        agent.actor_t.load_state_dict(blob["actor"])
        agent.actor.eval()
        agent.meta = {k: v for k, v in blob.items()
                      if k not in ("actor", "actor_online", "critic1", "critic2")}
        return agent
