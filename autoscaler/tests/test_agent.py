"""MO-TD3: networks, hybrid action space, multi-objective updates, replay, persistence."""

import numpy as np
import pytest
import torch

from src import config
from src.rl_agent.batching import pad_actions, pad_obs
from src.rl_agent.controllers import RLController
from src.rl_agent.env import AutoscalerEnv
from src.rl_agent.mo_td3 import MOActor, MOCritic, MOTD3Agent, _st_onehot, delta_index
from src.rl_agent.replay import EpisodeReplay
from src.simulation.simulator import random_topology
from src.topologies import DEMO6, TRAINTICKET14

W = np.asarray(config.DEFAULT_PREFERENCE, dtype=np.float32)


def episode(env, agent, w=W, steps=6, pattern="burst", topology=None):
    obs = env.reset(pattern, topology=topology)
    obs_list, idx_l, cont_l, rew = [obs], [], [], []
    from src.rl_agent.env import decode_action
    for _ in range(steps):
        idx, cont = agent.act(obs, w, noise_std=0.2, epsilon=0.2)
        obs, r, _, info = env.step_proposals(decode_action(env.services, obs["metrics"], idx, cont))
        idx_e, cont_e = info["executed"]
        obs_list.append(obs); idx_l.append(idx_e); cont_l.append(cont_e); rew.append(r)
    return obs_list, idx_l, cont_l, rew


HEADS = ["categorical", "round"]


def make_batch(agent, n_eps=3, batch=16, n_step=2):
    env, replay = AutoscalerEnv(None, seed=0), EpisodeReplay(50, seed=0)
    for _ in range(n_eps):
        o, i, c, r = episode(env, agent)
        replay.add_episode(o, i, c, r, W)
    return replay.sample(batch, n_step=n_step, gamma=config.TD3_GAMMA)


# ------------------------------------------------------------ shapes / spaces
def test_categorical_actor_outputs_a_hybrid_action_per_service():
    actor = MOActor(replica_head="categorical")
    obs = pad_obs([AutoscalerEnv(DEMO6, seed=0).reset("normal")])
    cont, logits = actor(obs, torch.tensor(np.array([W])))
    assert cont.shape == (1, 6, 2) and logits.shape == (1, 6, 7)
    assert cont.abs().max() <= 1.0
    assert logits.argmax(-1).eq(config.NO_OP_INDEX).all(), "starts as 'no replica change'"


def test_actor_initialises_to_a_sensible_static_sizing():
    from src.rl_agent.env import u_to_resource
    actor = MOActor()
    obs = pad_obs([AutoscalerEnv(DEMO6, seed=0).reset("normal")])
    cont, _ = actor(obs, torch.tensor(np.array([W])))
    cpu = u_to_resource(float(cont[0, 0, 0].detach()), config.CPU_MIN, config.CPU_MAX)
    mem = u_to_resource(float(cont[0, 0, 1].detach()), config.MEM_MIN_MB, config.MEM_MAX_MB)
    assert cpu == pytest.approx(0.5, rel=0.1) and mem == pytest.approx(128, rel=0.1)


def test_gcn_embedding_dimensions_match_the_proposal():
    assert config.GCN_HIDDEN_DIM == 128 and config.EMBEDDING_DIM == 64
    actor = MOActor()
    assert actor.encoder.out_dim == 64
    obs = pad_obs([AutoscalerEnv(DEMO6, seed=0).reset("normal")])
    assert actor.embed(obs).shape == (1, 6, 64)
    assert config.REPLICA_DELTA_OPTIONS == [-3, -2, -1, 0, 1, 2, 3]


def test_graph_embedding_is_64d_and_reacts_to_the_system_state():
    agent = MOTD3Agent(seed=0)
    env = AutoscalerEnv(DEMO6, seed=0)
    calm = env.reset("normal")
    emb, path = agent.graph_embedding(calm)
    assert emb.shape == (config.EMBEDDING_DIM,) == (64,) and np.isfinite(emb).all()
    assert path[0] == DEMO6.entry and set(path) <= set(DEMO6.services)
    for _ in range(6):
        obs, *_ = env.step_proposals({s: {"dr": 0, "cpu": calm["metrics"][s]["cpu"], "mem": 128.0}
                                      for s in DEMO6.services})
    busy = AutoscalerEnv(DEMO6, seed=0)
    busy.reset("burst", traffic=np.full(41, 80.0))
    for _ in range(4):
        hot, *_ = busy.step_proposals({s: {"dr": 0, "cpu": 0.25, "mem": 128.0} for s in DEMO6.services})
    assert not np.allclose(agent.graph_embedding(hot)[0], emb)


def test_critic_returns_one_q_value_per_objective():
    critic = MOCritic(replica_head="categorical")
    obs = pad_obs([AutoscalerEnv(DEMO6, seed=0).reset("normal")])
    q = critic(obs, torch.zeros(1, 6, 2), torch.nn.functional.one_hot(torch.full((1, 6), 3), 7).float(),
               torch.tensor([W]))
    assert q.shape == (1, config.N_OBJECTIVES)


def test_straight_through_onehot_is_hard_forward_soft_backward():
    logits = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
    out = _st_onehot(logits)
    assert out.detach().tolist() == [[1.0, 0.0, 0.0]]
    (out * torch.tensor([[0.0, 1.0, 0.0]])).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


# -------------------------------------------------------------- the round head
def test_round_head_executes_round_3a_and_feeds_the_critic_the_executed_value():
    assert delta_index(np.array([-1.0, -0.8, -0.45, -0.1, 0.0, 0.1, 0.45, 0.8, 1.0, 2.0])).tolist() == \
        [0, 1, 2, 3, 3, 3, 4, 5, 6, 6]       # round(3a) clipped to -3..+3, offset by +3 into the delta table
    agent = MOTD3Agent(replica_head="round", seed=1)
    for p in agent.actor.parameters():
        p.data += 0.2 * torch.randn_like(p)
    obs = AutoscalerEnv(DEMO6, seed=0).reset("burst")
    idx, cont = agent.act(obs, W, noise_std=0.3, epsilon=0.2)
    assert cont.shape == (6, 3) and (idx == delta_index(cont[:, 2])).all()
    assert idx.min() >= 0 and idx.max() <= 6


def test_round_head_actor_has_no_discrete_logits_and_critic_takes_no_onehot():
    actor, critic = MOActor(replica_head="round"), MOCritic(replica_head="round")
    obs = pad_obs([AutoscalerEnv(DEMO6, seed=0).reset("normal")])
    cont, logits = actor(obs, torch.tensor(np.array([W])))
    assert cont.shape == (1, 6, 3) and logits is None
    assert (cont[..., 2].abs() < 0.01).all(), "starts at replica delta 0"
    assert critic(obs, cont, None, torch.tensor(np.array([W]))).shape == (1, config.N_OBJECTIVES)


def test_replica_only_round_agent_zeroes_resources_but_keeps_replicas():
    agent = MOTD3Agent(replica_head="round", replica_only=True, seed=0)
    for p in agent.actor.parameters():
        p.data += 0.3 * torch.randn_like(p)
    obs = pad_obs([AutoscalerEnv(DEMO6, seed=0).reset("burst")])
    cont, _ = agent.actor(obs, torch.tensor(np.array([W])))
    assert (cont[..., :2] == 0).all() and cont[..., 2].abs().sum() > 0


# ----------------------------------------------------- topology independence
@pytest.mark.parametrize("ablation", [dict(), dict(use_ehg=False), dict(use_topology=False)])
def test_one_set_of_weights_serves_any_number_of_services(ablation):
    agent = MOTD3Agent(**ablation)
    n_params = sum(p.numel() for p in agent.actor.parameters())
    rng = np.random.default_rng(0)
    for topo in (random_topology(rng), DEMO6, TRAINTICKET14):
        obs = AutoscalerEnv(topo, seed=0).reset("normal")
        idx, cont = agent.act(obs, W)
        assert idx.shape == (len(topo.services),) and cont.shape == (len(topo.services), agent.n_cont)
    assert sum(p.numel() for p in agent.actor.parameters()) == n_params


def test_padding_does_not_change_a_services_decision():
    """Batched/padded inference must equal unbatched inference for the real nodes."""
    agent = MOTD3Agent(seed=1)
    small = AutoscalerEnv(DEMO6, seed=0).reset("burst")
    big = AutoscalerEnv(TRAINTICKET14, seed=0).reset("burst")
    alone, _ = (agent.actor(pad_obs([small]), torch.tensor([W])))
    both, _ = agent.actor(pad_obs([small, big]), torch.tensor([W, W]))
    assert torch.allclose(alone[0], both[0, :6], atol=1e-5)
    padded = pad_obs([small, big])
    assert padded["mask"][0].sum() == 6 and padded["mask"][1].sum() == 14
    assert padded["A_sdg"][0, 6:].abs().sum() == 0, "padded nodes exchange no messages"


def test_critic_ignores_padding_too():
    agent = MOTD3Agent(seed=2, replica_head="categorical")
    small = AutoscalerEnv(DEMO6, seed=0).reset("burst")
    big = AutoscalerEnv(TRAINTICKET14, seed=0).reset("burst")
    oh = lambda n: torch.nn.functional.one_hot(torch.full((n,), 3), 7).float()

    def q(obs_list):
        b = pad_obs(obs_list)
        n = b["X"].shape[1]
        return agent.critic1(b, torch.zeros(len(obs_list), n, 2), oh(n).unsqueeze(0).expand(len(obs_list), -1, -1),
                             torch.tensor([W] * len(obs_list)))
    assert torch.allclose(q([small])[0], q([small, big])[0], atol=1e-5)


# --------------------------------------------------------------------- acting
def test_act_is_deterministic_without_noise_and_noisy_with_it():
    agent = MOTD3Agent(seed=3)
    obs = AutoscalerEnv(DEMO6, seed=0).reset("normal")
    a1, a2 = agent.act(obs, W), agent.act(obs, W)
    assert (a1[0] == a2[0]).all() and np.allclose(a1[1], a2[1])
    n1 = agent.act(obs, W, noise_std=0.5, epsilon=1.0)
    assert not np.allclose(n1[1], a1[1])
    assert (n1[1] >= -1).all() and (n1[1] <= 1).all()


def test_preference_conditions_the_policy():
    agent = MOTD3Agent(seed=4)
    for p in agent.actor.parameters():                    # break the 'start as no-op' symmetry
        p.data += 0.05 * torch.randn_like(p)
    obs = AutoscalerEnv(DEMO6, seed=0).reset("burst")
    a = agent.act(obs, [1.0, 0.0, 0.0])[1]
    b = agent.act(obs, [0.0, 1.0, 0.0])[1]
    assert not np.allclose(a, b), "the actor must actually read w"


# ------------------------------------------------------------------ learning
@pytest.mark.parametrize("head", HEADS)
def test_update_runs_and_critic_loss_falls_when_fitting_a_fixed_batch(head):
    agent = MOTD3Agent(seed=0, replica_head=head)
    batch = make_batch(agent)
    first = agent.update(batch)["critic_loss"]
    for _ in range(60):
        info = agent.update(batch)
    assert np.isfinite(info["critic_loss"]) and info["critic_loss"] < first
    info = {**info, **agent.update(batch)}                 # the actor updates every 2nd step (TD3 delay)
    assert "actor_loss" in info and np.isfinite(info["actor_loss"])


def test_targets_track_the_online_networks_slowly():
    agent = MOTD3Agent(seed=0)
    batch = make_batch(agent)
    before = [p.clone() for p in agent.actor_t.parameters()]
    for _ in range(4):
        agent.update(batch)
    moved = sum(float((a - b).abs().sum()) for a, b in zip(agent.actor_t.parameters(), before))
    online = sum(float((a - b).abs().sum()) for a, b in zip(agent.actor.parameters(), before))
    assert 0 < moved < online


@pytest.mark.parametrize("head", HEADS)
def test_behaviour_cloning_pulls_the_actor_toward_demonstrated_actions(head):
    agent = MOTD3Agent(seed=0, replica_head=head)
    env, replay = AutoscalerEnv(DEMO6, seed=0), EpisodeReplay(50)
    o, i, c, r = episode(env, agent, steps=8)
    target_idx = [np.full(6, 5) for _ in i]                # demonstrator always says "+2 pods"
    target_cont = [np.stack([x[:, 0], x[:, 1], np.full(6, 2 / 3)], 1) for x in c]
    replay.add_episode(o, target_idx, target_cont, r, W, demo=True)
    obs = pad_obs([o[0]])
    w = torch.tensor(np.array([W]))

    def prefers_plus_two():
        cont, logits = agent.actor(obs, w)
        return float(torch.softmax(logits, -1)[0, :, 5].mean()) if head == "categorical" else float(cont[0, :, 2].mean())
    before = prefers_plus_two()
    for _ in range(80):
        agent.update(replay.sample(16, demo_fraction=1.0), bc_weight=5.0)
    assert prefers_plus_two() > before + 0.2


def test_replica_only_agent_never_resizes():
    agent = MOTD3Agent(replica_only=True, seed=0)
    obs = AutoscalerEnv(DEMO6, seed=0).reset("burst")
    ctrl = RLController(agent, DEMO6.services, fallback=False)
    props = ctrl.decide(obs)
    assert all(p["cpu"] == obs["metrics"][s]["cpu"] and p["mem"] == obs["metrics"][s]["mem"]
               for s, p in props.items())


def test_controller_falls_back_to_hpa_while_the_breaker_is_open():
    agent = MOTD3Agent(seed=0)
    ctrl = RLController(agent, DEMO6.services)
    obs = AutoscalerEnv(DEMO6, seed=0).reset("burst")
    ctrl.decide(obs, breaker_open=False)
    assert ctrl.last_used_fallback is False
    ctrl.decide(obs, breaker_open=True)
    assert ctrl.last_used_fallback is True


# --------------------------------------------------------------------- replay
def test_replay_n_step_returns_and_bootstrap_horizon():
    r = np.array([[1.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0], [8.0, 0, 0]], dtype=np.float32)
    env = AutoscalerEnv(DEMO6, seed=0)
    obs = env.reset("normal")
    ep_obs = [obs] * 5
    replay = EpisodeReplay(10, seed=0)
    replay.add_episode(ep_obs, [np.zeros(6, int)] * 4, [np.zeros((6, 2), np.float32)] * 4, r, W)
    for _ in range(40):
        b = replay.sample(1, n_step=3, gamma=0.5)
        disc = float(b["disc"][0, 0])
        ret = float(b["r"][0, 0])
        assert (disc, ret) in {(0.125, 1 + 1 + 1), (0.125, 2 + 2 + 2), (0.25, 4 + 4), (0.5, 8)}


def test_replay_mixes_demonstrations_in_the_requested_fraction():
    env, agent = AutoscalerEnv(DEMO6, seed=0), MOTD3Agent()
    replay = EpisodeReplay(10, seed=0)
    o, i, c, r = episode(env, agent)
    replay.add_episode(o, i, c, r, W, demo=False)
    replay.add_episode(o, i, c, r, W, demo=True)
    assert replay.sample(40, demo_fraction=0.25)["demo"].sum() == 10
    assert replay.sample(8, demo_fraction=0.0)["demo"].sum() == 0
    assert replay.n_transitions == 12


def test_pad_actions_shapes():
    idx, cont = pad_actions([np.array([1, 2]), np.array([3])], [np.ones((2, 2), np.float32), np.ones((1, 2), np.float32)], 3)
    assert idx.shape == (2, 3) and cont.shape == (2, 3, 2) and cont[1, 1:].abs().sum() == 0


# ---------------------------------------------------------------- persistence
@pytest.mark.parametrize("head", HEADS)
def test_save_and_load_preserves_behaviour_and_flags(tmp_path, head):
    agent = MOTD3Agent(use_ehg=False, seed=5, replica_head=head)
    for p in agent.actor.parameters():
        p.data += 0.05 * torch.randn_like(p)
    agent.actor_t.load_state_dict(agent.actor.state_dict())     # the deployed actor is the target actor
    path = str(tmp_path / "m.pt")
    agent.save(path, extra={"note": "x"})
    loaded = MOTD3Agent.load(path)
    obs = AutoscalerEnv(DEMO6, seed=0).reset("burst")
    a, b = agent.act(obs, W), loaded.act(obs, W)
    assert (a[0] == b[0]).all() and np.allclose(a[1], b[1], atol=1e-6)
    assert loaded.flags == {"use_topology": True, "use_ehg": False, "replica_only": False, "replica_head": head}
    assert loaded.meta["note"] == "x"


def test_loading_an_old_ppo_checkpoint_is_refused_with_a_clear_message(tmp_path):
    path = str(tmp_path / "old.pt")
    torch.save({"state_dict": {}, "use_ehg": True}, path)
    with pytest.raises(ValueError, match="not an MO-TD3 checkpoint"):
        MOTD3Agent.load(path)
