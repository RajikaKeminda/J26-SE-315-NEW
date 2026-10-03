"""
trainer.py

Training loop for the MO-TD3 agent (and its ablation arms) on the calibrated
simulator.

  * Every episode runs on a NEW RANDOM TOPOLOGY (4-9 services, varied fan-out and
    per-service work), so the shared policy must learn a general rule rather than
    memorise one graph.
  * Every episode draws a PREFERENCE vector w ~ Dirichlet(1,1,1) (30 % of the time the
    default preference), and stored transitions are re-labelled with fresh preferences
    at update time, so one network learns the whole latency/cost trade-off.
  * The replay buffer is seeded with demonstration episodes from the HPA / threshold
    controllers (TD3+BC); the behaviour-cloning term decays as the agent's own data
    takes over.
  * The deterministic policy is validated on fixed held-out scenarios; the best
    checkpoint and the full learning curve are kept (the proposal's "convergence
    behaviour").
"""

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from src import config
from src.baselines.rule_based import HPABaseline, ThresholdBaseline
from src.evaluation.metrics import evaluate_controller
from src.rl_agent.controllers import RLController
from src.rl_agent.env import AutoscalerEnv
from src.rl_agent.mo_td3 import MOTD3Agent
from src.rl_agent.replay import EpisodeReplay
from src.simulation.simulator import random_topology
from src.topologies import DEMO6

# arm -> options.  `topology/ehg/replica_only` select the architecture ablations,
# `fixed_w` makes the reward single-objective, `demos` toggles the demonstration seeding, `head`
# overrides the replica head.
ARMS = {
    "mo_td3_full":         dict(topology=True,  ehg=True,  replica_only=False, fixed_w=None,              demos=True),
    "mo_td3_sdg_only":     dict(topology=True,  ehg=False, replica_only=False, fixed_w=None,              demos=True),
    "mo_td3_no_topology":  dict(topology=False, ehg=False, replica_only=False, fixed_w=None,              demos=True),
    "mo_td3_single_obj":   dict(topology=True,  ehg=True,  replica_only=False, fixed_w=(1.0, 0.0, 0.0),   demos=True),
    "mo_td3_replica_only": dict(topology=True,  ehg=True,  replica_only=True,  fixed_w=None,              demos=True),
    "mo_td3_no_demo":      dict(topology=True,  ehg=True,  replica_only=False, fixed_w=None,              demos=False),
    # straight-through categorical replica head instead of the rounded-continuous one (see mo_td3.py)
    "mo_td3_categorical":  dict(topology=True,  ehg=True,  replica_only=False, fixed_w=None,              demos=True,
                                head="categorical"),
}


@dataclass
class HParams:
    """Training knobs that are NOT part of an ablation arm's definition."""
    n_step: int = config.TD3_N_STEP
    explore_noise: float = config.TD3_EXPLORE_NOISE
    eps_start: float = config.TD3_EPSILON_START
    eps_end: float = config.TD3_EPSILON_END
    bc_weight: float = config.TD3_BC_WEIGHT
    bc_floor: float = 0.3           # BC weight never decays below bc_weight * bc_floor
    bc_decay_frac: float = 0.8      # fraction of training over which BC decays to its floor
    demo_target_range: tuple = (0.35, 0.65)
    demo_cpus: tuple = (0.25, 0.5, 1.0)
    default_pref_prob: float = 0.3  # share of episodes that use the default preference
    relabel_prob: float = 0.5       # share of update batches whose preferences are re-drawn
    val_episodes: int = 16
    top_k: int = 3                     # checkpoints re-validated on a fresh, larger set before the winner is chosen
    final_val_episodes: int = 48
    fixed_topology_prob: float = 0.0   # share of training episodes on the deployed 6-service topology
    replica_head: str = "round"        # "round" (continuous, executed as round(3a)) | "categorical" (straight-through)
    actor_lr: float = 1e-4             # lower than TD3's usual 3e-4: the 3e-4 runs collapsed intermittently
    critic_lr: float = 3e-4
    batch_size: int = config.TD3_BATCH
    tau: float = config.TD3_TAU
    gamma: float = config.TD3_GAMMA
    policy_delay: int = config.TD3_POLICY_DELAY
    grad_clip: float = 5.0


def sample_preference(rng, fixed_w=None, default_prob=0.3):
    if fixed_w is not None:
        return np.asarray(fixed_w, dtype=np.float32)
    if rng.random() < default_prob:
        return np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32)
    return rng.dirichlet(np.ones(config.N_OBJECTIVES)).astype(np.float32)


def run_collect(env, policy_fn, pattern="random", topology=None):
    """One episode -> (obs_list, executed idx list, cont list, reward list)."""
    obs = env.reset(pattern, topology=topology)
    obs_list, idx_list, cont_list, rew = [obs], [], [], []
    done = False
    while not done:
        proposals = policy_fn(obs, env)
        obs, r, done, info = env.step_proposals(proposals)
        idx, cont = info["executed"]
        obs_list.append(obs); idx_list.append(idx); cont_list.append(cont); rew.append(r)
    return obs_list, idx_list, cont_list, rew


def collect_demonstrations(replay, n_episodes, seed, rng, hp=None):
    """Demonstration episodes from HPA (random target) and threshold controllers, with the
    pods started at a random manifest size so the buffer covers more of the state space."""
    hp = hp or HParams()
    for i in range(n_episodes):
        env = AutoscalerEnv(topology=None, seed=seed * 7919 + i)
        topo = DEMO6 if rng.random() < hp.fixed_topology_prob else random_topology(env.rng)
        kind = rng.choice(["hpa", "hpa", "threshold"])
        initial = {"cpu": float(rng.choice(hp.demo_cpus))}
        if kind == "hpa":
            teacher = HPABaseline(topo.services, target_utilization=float(rng.uniform(*hp.demo_target_range)))
        else:
            teacher = ThresholdBaseline(topo.services)

        obs = env.reset(rng.choice(["normal", "burst", "diurnal", "rampup"]), topology=topo, initial=initial)
        obs_list, idx_list, cont_list, rew, done = [obs], [], [], [], False
        while not done:
            obs, r, done, info = env.step_proposals(teacher.decide(obs["metrics"]))
            idx, cont = info["executed"]
            obs_list.append(obs); idx_list.append(idx); cont_list.append(cont); rew.append(r)
        replay.add_episode(obs_list, idx_list, cont_list, rew, w=config.DEFAULT_PREFERENCE, demo=True)


def validate(agent, w, n_episodes=16, seed0=50_000):
    """Deterministic policy (the smoothed target actor) on fixed held-out scenarios."""
    ctrl_factory = lambda: RLController(agent, DEMO6.services, w=w, fallback=False)
    agent.use_target_for_act = True
    try:
        return evaluate_controller(ctrl_factory, DEMO6, n_episodes=n_episodes, w=w, seed0=seed0)
    finally:
        agent.use_target_for_act = False


def select_checkpoint(agent, candidates, w, n_episodes, seed0=60_000):
    """Picking the single best of ~24 noisy validations over-fits those 16 scenarios (the winner's
    score is partly luck). Re-score the top candidates on a fresh, larger scenario set and take the
    best of THOSE. Returns (state, fresh_score)."""
    saved = copy.deepcopy(agent.actor_t.state_dict())
    best = (None, float("inf"))
    try:
        for state in candidates:
            agent.actor_t.load_state_dict(state["actor"])
            score = validate(agent, w, n_episodes, seed0)["scalarized"]
            if score < best[1]:
                best = (state, score)
    finally:
        agent.actor_t.load_state_dict(saved)
    return best


def train_agent(arm="mo_td3_full", episodes=1000, seed=0, demo_episodes=400, warmup_episodes=30,
                updates_per_episode=30, batch_size=None, val_every=50, verbose=True, log=print, hp=None):
    hp = hp or HParams()
    opts = ARMS[arm]
    batch_size = batch_size or hp.batch_size
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    head = opts.get("head", hp.replica_head)
    agent = MOTD3Agent(use_topology=opts["topology"], use_ehg=opts["ehg"],
                       replica_only=opts["replica_only"], seed=seed, replica_head=head)
    for g in agent.actor_opt.param_groups:
        g["lr"] = hp.actor_lr
    for g in agent.critic_opt.param_groups:
        g["lr"] = hp.critic_lr
    saved_cfg = (config.TD3_TAU, config.TD3_POLICY_DELAY)      # restored on return
    config.TD3_TAU, config.TD3_POLICY_DELAY = hp.tau, hp.policy_delay
    env = AutoscalerEnv(topology=None, seed=seed, replica_only=opts["replica_only"])
    replay = EpisodeReplay(config.TD3_REPLAY_EPISODES, seed=seed)
    if opts["demos"] and demo_episodes:
        collect_demonstrations(replay, demo_episodes, seed, rng, hp)
        if verbose:
            log(f"  seeded replay with {demo_episodes} demonstration episodes "
                f"({replay.n_transitions} transitions)")

    eval_w = np.asarray(opts["fixed_w"] if opts["fixed_w"] is not None else config.DEFAULT_PREFERENCE,
                        dtype=np.float32)
    history, best, best_state, kept = [], float("inf"), None, []
    t0 = time.time()
    last_info = {}

    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        noise = hp.explore_noise * (1.0 - 0.7 * frac)
        eps = hp.eps_start + (hp.eps_end - hp.eps_start) * min(1.0, frac / 0.7)
        w = sample_preference(rng, opts["fixed_w"], hp.default_pref_prob)

        def policy_fn(obs, env_, w=w, noise=noise, eps=eps):
            from src.rl_agent.env import decode_action
            idx, cont = agent.act(obs, w, noise_std=noise, epsilon=eps)
            return decode_action(env_.services, obs["metrics"], idx, cont, opts["replica_only"])

        topo = DEMO6 if rng.random() < hp.fixed_topology_prob else None
        obs_list, idx_list, cont_list, rew = run_collect(env, policy_fn, topology=topo)
        replay.add_episode(obs_list, idx_list, cont_list, rew, w=w, demo=False)

        if ep >= warmup_episodes:
            bc_w = (hp.bc_weight * max(hp.bc_floor, 1.0 - frac / hp.bc_decay_frac)) if opts["demos"] else 0.0
            demo_frac = 0.25 * max(0.2, 1.0 - frac) if (opts["demos"] and replay.demo_eps) else 0.0
            for _ in range(updates_per_episode):
                batch = replay.sample(batch_size, demo_fraction=demo_frac, n_step=hp.n_step,
                                      gamma=hp.gamma)
                if opts["fixed_w"] is None and rng.random() < hp.relabel_prob:     # preference relabelling
                    B = batch["w"].shape[0]
                    batch["w"] = torch.as_tensor(rng.dirichlet(np.ones(config.N_OBJECTIVES), B), dtype=torch.float32)
                last_info = agent.update(batch, bc_weight=bc_w)

        if (ep + 1) % val_every == 0 or ep == episodes - 1:
            v = validate(agent, eval_w, hp.val_episodes)
            entry = {"episode": ep + 1, "val_scalarized": v["scalarized"], "val_p95_ms": v["p95_latency"],
                     "val_cost": v["mean_cost"], "val_slo_violation": v["slo_violation_rate"],
                     "seconds": round(time.time() - t0, 1), **{k: round(x, 4) for k, x in last_info.items()}}
            history.append(entry)
            if v["scalarized"] < best:
                best = v["scalarized"]
            kept.append((v["scalarized"], agent.state()))
            kept = sorted(kept, key=lambda t: t[0])[:hp.top_k]
            if verbose:
                log(f"  ep {ep + 1:5d} | val score {v['scalarized']:.3f} | p95 {v['p95_latency']:7.1f} ms | "
                    f"cost {v['mean_cost']:.2f} | viol {v['slo_violation_rate'] * 100:5.1f}% | "
                    f"{entry['seconds']:.0f}s")

    fresh = None
    if kept:
        best_state, fresh = select_checkpoint(agent, [st for _, st in kept], eval_w, hp.final_val_episodes)
        if verbose:
            log(f"  checkpoint selection: top-{len(kept)} by validation re-scored on {hp.final_val_episodes} fresh "
                f"episodes -> best fresh score {fresh:.3f}")
    config.TD3_TAU, config.TD3_POLICY_DELAY = saved_cfg
    meta = {"arm": arm, "seed": seed, "episodes": episodes, "history": history,
            "best_val_scalarized": best, "selected_fresh_val_scalarized": fresh, "eval_preference": [round(float(x), 6) for x in eval_w],
            "topology_training": "random 4-9 services"}
    return agent, best_state, meta
