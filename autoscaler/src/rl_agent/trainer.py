"""
trainer.py

Training loop for the MO-TD3 agent (and its ablation arms) on the calibrated
simulator.

  * Every episode runs on a NEW RANDOM TOPOLOGY (4-12 services, 2-5 tiers, varied fan-out,
    shared callees, per-service work), so the shared policy must learn a general rule rather
    than memorise one graph.
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
from src.rl_agent.env import AutoscalerEnv, topology_base_rps
from src.rl_agent.mo_td3 import MOTD3Agent
from src.rl_agent.replay import EpisodeReplay
from src.simulation.simulator import random_topology
from src.topologies import DEMO6, bottleneck_scenario

# arm -> options.  `topology/ehg/replica_only` select the architecture ablations,
# `fixed_w` makes the reward single-objective, `demos` toggles the HPA-demonstration seeding
# (TD3+BC), `head` overrides the replica head.
#
# The headline agent `mo_td3_full` is trained by pure reinforcement learning. The study found that
# seeding it with HPA demonstrations makes it WORSE (it stays tethered to the teacher's behaviour),
# so every `mo_td3_demo*` arm is the same model WITH demonstrations - reported side by side, not
# hidden. Each ablation arm differs from its family's full model by exactly one factor.
def _arm(topology=True, ehg=True, replica_only=False, fixed_w=None, demos=False, head=None):
    out = dict(topology=topology, ehg=ehg, replica_only=replica_only, fixed_w=fixed_w, demos=demos)
    if head:
        out["head"] = head
    return out


ARMS = {
    # pure RL family
    "mo_td3_full":         _arm(),
    "mo_td3_sdg_only":     _arm(ehg=False),
    "mo_td3_no_topology":  _arm(topology=False, ehg=False),
    "mo_td3_single_obj":   _arm(fixed_w=(1.0, 0.0, 0.0)),
    "mo_td3_replica_only": _arm(replica_only=True),
    "mo_td3_categorical":  _arm(head="categorical"),
    # the same models, seeded with HPA / threshold demonstrations (TD3+BC)
    "mo_td3_demo":                _arm(demos=True),
    "mo_td3_demo_sdg_only":       _arm(ehg=False, demos=True),
    "mo_td3_demo_no_topology":    _arm(topology=False, ehg=False, demos=True),
    "mo_td3_demo_single_obj":     _arm(fixed_w=(1.0, 0.0, 0.0), demos=True),
    "mo_td3_demo_replica_only":   _arm(replica_only=True, demos=True),
    "mo_td3_demo_categorical":    _arm(head="categorical", demos=True),
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
    randomize_initial: bool = False     # start every service at its own CPU limit (domain randomisation)
    hot_prob: float = 0.0               # share of episodes in which ONE random service is made the only bottleneck (topologies.bottleneck_scenario)
    hot_demand: tuple = (0.25, 1.6)     # cores the bottleneck service needs (it starts with a 0.1-core limit)


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


def validate(agent, w, n_episodes=16, seed0=50_000, topologies=None):
    """Deterministic policy (the smoothed target actor) on fixed held-out scenarios: the deployed 6-service topology,
    or - for fine-tuning on other targets - the mean over several `topologies`."""
    topos = topologies or [DEMO6]
    per = max(2, n_episodes // len(topos))
    agent.use_target_for_act = True
    try:
        runs = [evaluate_controller(lambda t=t: RLController(agent, t.services, w=w, fallback=False), t,
                                    n_episodes=per, w=w, seed0=seed0 + 17 * i) for i, t in enumerate(topos)]
    finally:
        agent.use_target_for_act = False
    if len(runs) == 1:
        return runs[0]
    return {k: float(np.mean([r[k] for r in runs])) for k in ("scalarized", "p95_latency", "mean_cost", "slo_violation_rate")}


def select_checkpoint(agent, candidates, w, n_episodes, seed0=60_000, topologies=None):
    """Picking the single best of ~24 noisy validations over-fits those 16 scenarios (the winner's
    score is partly luck). Re-score the top candidates on a fresh, larger scenario set and take the
    best of THOSE. Returns (state, fresh_score)."""
    saved = copy.deepcopy(agent.actor_t.state_dict())
    best = (None, float("inf"))
    try:
        for state in candidates:
            agent.actor_t.load_state_dict(state["actor"])
            score = validate(agent, w, n_episodes, seed0, topologies)["scalarized"]
            if score < best[1]:
                best = (state, score)
    finally:
        agent.actor_t.load_state_dict(saved)
    return best


def train_agent(arm="mo_td3_full", episodes=1000, seed=0, demo_episodes=400, warmup_episodes=30,
                updates_per_episode=30, batch_size=None, val_every=50, verbose=True, log=print, hp=None,
                init_from=None, sizes=(4, 12), target_topologies=None, target_prob=0.0, val_topologies=None, recorder=None):
    """`init_from`: warm-start from a checkpoint (fine-tuning). `sizes`: service-count range of the random training
    topologies. `target_topologies` (with probability `target_prob` per episode) are mixed into training, and
    `val_topologies` replace the 6-service demo as the validation set - together they specialise a trained agent.
    `recorder` (persistence.training.TrainingRecorder) stores every episode and the validation curve in the database."""
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
    if init_from:
        agent.warm_start(init_from)
        if verbose:
            log(f"  warm start from {init_from}")
    saved_cfg = (config.TD3_TAU, config.TD3_POLICY_DELAY)      # restored on return
    config.TD3_TAU, config.TD3_POLICY_DELAY = hp.tau, hp.policy_delay
    env = AutoscalerEnv(topology=None, seed=seed, replica_only=opts["replica_only"])
    env.topology_sizes = tuple(sizes)
    env.randomize_initial = hp.randomize_initial
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
        if target_topologies and rng.random() < target_prob:
            topo = target_topologies[int(rng.integers(len(target_topologies)))]
        if hp.hot_prob and rng.random() < hp.hot_prob:
            # a BOTTLENECK episode: the request mix is shifted towards one random service and the traffic scaled until it is the only saturated one
            # (topologies.bottleneck_scenario), on whatever application this episode would have used - so the agent can only be right by reading where the load is
            base = topo if topo is not None else random_topology(rng, *sizes)
            scenario, _ = bottleneck_scenario(base, rng, topology_base_rps(base), demand_range=hp.hot_demand)
            topo = scenario or topo
        obs_list, idx_list, cont_list, rew = run_collect(env, policy_fn, topology=topo)
        replay.add_episode(obs_list, idx_list, cont_list, rew, w=w, demo=False)
        if recorder is not None:
            recorder.on_episode(ep, env, w, obs_list, idx_list, cont_list, rew)

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
            v = validate(agent, eval_w, hp.val_episodes, topologies=val_topologies)
            entry = {"episode": ep + 1, "val_scalarized": v["scalarized"], "val_p95_ms": v["p95_latency"],
                     "val_cost": v["mean_cost"], "val_slo_violation": v["slo_violation_rate"],
                     "seconds": round(time.time() - t0, 1), **{k: round(x, 4) for k, x in last_info.items()}}
            history.append(entry)
            if recorder is not None:
                recorder.on_validation(entry)
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
        best_state, fresh = select_checkpoint(agent, [st for _, st in kept], eval_w, hp.final_val_episodes, topologies=val_topologies)
        if verbose:
            log(f"  checkpoint selection: top-{len(kept)} by validation re-scored on {hp.final_val_episodes} fresh "
                f"episodes -> best fresh score {fresh:.3f}")
    config.TD3_TAU, config.TD3_POLICY_DELAY = saved_cfg
    meta = {"arm": arm, "seed": seed, "episodes": episodes, "history": history,
            "best_val_scalarized": best, "selected_fresh_val_scalarized": fresh, "eval_preference": [round(float(x), 6) for x in eval_w],
            "topology_training": f"random {sizes[0]}-{sizes[1]} services, 2-5 tiers" + (f" + {len(target_topologies)} target topologies p={target_prob}" if target_topologies else ""),
            "init_from": init_from}
    return agent, best_state, meta
