"""Trained checkpoints load and behave; evaluation is deterministic; training runs."""

import os

import numpy as np
import pytest

from src import config
from src.evaluation import evaluate, train_arm
from src.baselines.rule_based import HPABaseline, StaticBaseline
from src.simulation.simulator import fixed_topology


@pytest.mark.parametrize("arm", list(train_arm.ARMS))
def test_shipped_checkpoints_load(arm):
    policy = evaluate.load_arm(arm)
    assert policy is not None, f"models/{arm}.pt missing"
    use_topology, use_ehg, _ = train_arm.ARMS[arm]
    assert policy.encoder.use_topology == use_topology and policy.encoder.use_ehg == use_ehg


def test_missing_checkpoint_returns_none(tmp_path):
    assert evaluate.load_policy_file(str(tmp_path / "nope.pt")) is None


def test_same_weights_drive_any_topology_size():
    """Headline claim of the shared policy, checked on the SHIPPED weights."""
    import torch
    policy = evaluate.load_arm("rl_full")
    for n in (3, 6, 12, 40):
        a, _, _ = policy.act(torch.rand(n, config.NODE_FEATURE_DIM), torch.rand(n, n),
                             torch.rand(n, n), torch.rand(3), deterministic=True)
        assert a.shape == (n,)


def test_episode_is_deterministic_given_seed():
    s, e, g = fixed_topology()
    r1 = evaluate.run_episode(HPABaseline(s).decide, 7, "burst", s, e, g)
    r2 = evaluate.run_episode(HPABaseline(s).decide, 7, "burst", s, e, g)
    assert r1 == r2


def test_trained_policy_and_hpa_beat_static_under_burst():
    s, e, g = fixed_topology()
    static = evaluate.run_episode(StaticBaseline(s).decide, 3, "burst", s, e, g)
    hpa = evaluate.run_episode(HPABaseline(s).decide, 3, "burst", s, e, g)
    rl = evaluate.run_episode(
        evaluate.make_rl_decider(evaluate.load_arm("rl_full"), s, e, g), 3, "burst", s, e, g)
    assert hpa["p95_latency"] < static["p95_latency"]
    assert rl["p95_latency"] < static["p95_latency"]
    assert rl["slo_violation_rate"] <= hpa["slo_violation_rate"]


def test_committed_results_match_a_fresh_evaluation():
    """README numbers are reproducible: re-evaluating shipped models reproduces results JSON."""
    import json
    fresh = evaluate.main(n_eval_episodes=20, save=False)
    with open(config.RESULTS_PATH) as f:
        saved = json.load(f)
    for arm, metrics in saved.items():
        for k, v in metrics.items():
            assert fresh[arm][k] == pytest.approx(v, rel=1e-6), (arm, k)


def test_training_runs_and_checkpoint_round_trips(tmp_path):
    policy, history, flags = train_arm.train_policy("rl_full", iterations=2, seed=0, verbose=False)
    assert len(history) == 2 and all(np.isfinite(history))
    path = str(tmp_path / "ckpt.pt")
    train_arm.save_policy(policy, history, flags, path)
    loaded = evaluate.load_policy_file(path)
    for (k1, v1), (k2, v2) in zip(policy.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2 and (v1 == v2).all()


@pytest.mark.parametrize("arm", ["rl_sdg_only", "rl_no_topology", "rl_single_obj"])
def test_every_arm_trains(arm):
    _, history, _ = train_arm.train_policy(arm, iterations=1, seed=1, verbose=False)
    assert len(history) == 1


def test_train_cli_rejects_unknown_arm():
    with pytest.raises(SystemExit):
        train_arm.train_policy("bogus", 1)


def test_warm_start_arm_trains_with_imitation(monkeypatch):
    monkeypatch.setattr(train_arm, "IMITATION_EPISODES", 2)
    monkeypatch.setattr(train_arm, "IMITATION_EPOCHS", 1)
    _, history, flags = train_arm.train_policy("rl_hpa_init", iterations=1, seed=0, verbose=False)
    assert len(history) == 1 and flags == {"use_topology": True, "use_ehg": True}


def test_imitation_learns_the_teacher():
    from src.rl_agent import imitation
    from src.rl_agent.env import AutoscalerEnv
    from src.rl_agent.shared_policy import SharedPolicy
    import torch
    torch.manual_seed(0)
    env = AutoscalerEnv(seed=0)
    demos = imitation.collect_demonstrations(env, 6)
    assert len(demos) == 6 * (config.EPISODE_LENGTH_STEPS - 1)
    obs, actions = demos[0]
    assert actions.shape == (obs[0].shape[0],) and set(actions.tolist()) <= {0, 1, 2}
    policy = SharedPolicy(config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM,
                          config.GCN_EMBED_DIM, config.POLICY_HIDDEN)
    losses = imitation.pretrain(policy, demos, epochs=3, verbose=False)
    assert losses[-1] < losses[0], f"imitation loss did not fall: {losses}"
