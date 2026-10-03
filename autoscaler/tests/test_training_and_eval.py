"""Metrics, the training loop, the evaluation harness, the Pareto maths, and the shipped artefacts."""

import json
import os

import numpy as np
import pytest

from src import config
from src.evaluation import evaluate, pareto, train_agent
from src.evaluation.metrics import adaptation_time, evaluate_controller, run_episode
from src.rl_agent.controllers import RLController, RuleController
from src.rl_agent.mo_td3 import MOTD3Agent
from src.rl_agent.trainer import ARMS, HParams, sample_preference, train_agent as train
from src.topologies import DEMO6, TRAINTICKET14

S = DEMO6.services


# --------------------------------------------------------------------- metrics
def test_adaptation_time():
    base = 10.0
    traffic = [10, 10, 40, 40, 40, 40, 10, 10]
    ok = [100] * 8
    assert adaptation_time(traffic, base, ok, 250.0) == 0.0                              # never violated
    p95 = [100, 100, 300, 300, 100, 100, 100, 100]
    assert adaptation_time(traffic, base, p95, 250.0) == 2 * config.STEP_SECONDS        # 2 windows in breach
    never = [100, 100, 300, 300, 300, 300, 300, 300]
    assert adaptation_time(traffic, base, never, 250.0) == 6 * config.STEP_SECONDS      # still broken at the end
    assert adaptation_time([10] * 8, base, [300] * 8, 250.0) == 0.0                     # no load event


def test_run_episode_is_reproducible_and_seeds_are_matched_across_controllers():
    a = run_episode(RuleController("hpa", S), DEMO6, 5, "burst")
    b = run_episode(RuleController("hpa", S), DEMO6, 5, "burst")
    assert a == b
    c = run_episode(RuleController("static", S), DEMO6, 5, "burst", record=True)
    d = run_episode(RuleController("hpa", S), DEMO6, 5, "burst", record=True)
    assert [t["rps"] for t in c["trace"]] == [t["rps"] for t in d["trace"]], "identical traffic for every controller"
    assert c["p95_latency"] != d["p95_latency"]


def test_episode_metrics_are_consistent():
    r = run_episode(RuleController("static", S), DEMO6, 1, "normal")
    assert r["p50_latency"] < r["p95_latency"] < r["p99_latency"]
    assert 0 <= r["slo_violation_rate"] <= 1 and 0 <= r["mean_util"] <= 1
    assert r["mean_cost"] == pytest.approx(12 * (0.25 + config.MEM_COST_PER_GIB * 128 / 1024))
    w = np.asarray(config.DEFAULT_PREFERENCE)
    assert r["scalarized"] == pytest.approx(w @ [r["obj_slo"], r["obj_cost"], r["obj_balance"]])


def test_evaluate_controller_breaks_results_down_by_pattern():
    r = evaluate_controller(lambda: RuleController("static", S), DEMO6, n_episodes=4)
    assert set(r["by_pattern"]) == {"normal", "burst", "diurnal", "rampup"} and r["n_episodes"] == 4
    assert r["by_pattern"]["burst"]["p95_latency"] > r["by_pattern"]["normal"]["p95_latency"]


def test_baselines_behave_like_their_real_counterparts_on_a_burst():
    run = lambda c, **kw: run_episode(RuleController(c, S, **kw), DEMO6, 3, "burst")
    static, hpa = run("static"), run("hpa")
    tuned = run("hpa", initial={"cpu": 0.5}, target_utilization=0.5)
    # elasticity: HPA absorbs the burst that static collapses under (p95 an order of magnitude lower)
    assert hpa["p95_latency"] < static["p95_latency"] / 3
    # ...but the default 60 %-target HPA on 0.25-core pods runs hot between bursts: CFS throttling inflates
    # p95 at ~60 % utilisation of such small pods, so it breaches the SLO more often than idle static pods do.
    assert hpa["slo_violation_rate"] > 0.3
    # a tuned HPA (bigger pods, lower target) fixes that: this is why the tuned HPA is the baseline that matters
    assert tuned["scalarized"] < hpa["scalarized"] and tuned["slo_violation_rate"] < hpa["slo_violation_rate"]


# ---------------------------------------------------------------- HPA tuning
def test_hpa_tuning_returns_the_best_grid_point():
    best, table = evaluate.tune_hpa(DEMO6, n_episodes=1)
    assert len(table) == len(evaluate.HPA_GRID_CPU) * len(evaluate.HPA_GRID_TARGET)
    assert best == min(table, key=lambda t: t["scalarized"])


# ------------------------------------------------------------------ training
def test_preference_sampling():
    rng = np.random.default_rng(0)
    assert sample_preference(rng, (1, 0, 0)).tolist() == [1, 0, 0]
    ws = np.array([sample_preference(rng) for _ in range(300)])
    assert np.allclose(ws.sum(1), 1, atol=1e-5)
    share_default = np.mean([np.allclose(w, config.DEFAULT_PREFERENCE, atol=1e-5) for w in ws])
    assert 0.2 < share_default < 0.45
    assert (ws.min(0) < 0.05).all(), "the whole simplex is explored"


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_arm_trains_for_a_few_episodes(arm):
    agent, best_state, meta = train(arm, episodes=3, demo_episodes=2, warmup_episodes=1, updates_per_episode=2,
                                    val_every=3, verbose=False, hp=HParams(val_episodes=1, final_val_episodes=1, top_k=1))
    assert best_state["algo"] == "mo_td3" and meta["arm"] == arm and len(meta["history"]) == 1
    flags = ARMS[arm]
    assert agent.flags == {"use_topology": flags["topology"], "use_ehg": flags["ehg"],
                           "replica_only": flags["replica_only"], "replica_head": flags.get("head", "round")}
    assert meta["eval_preference"] == pytest.approx(list(flags["fixed_w"] or config.DEFAULT_PREFERENCE))


def test_cli_writes_checkpoint_and_learning_curve(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    train_agent.main("mo_td3_full", 2, 3, 2, 2, hp=HParams(val_episodes=1, final_val_episodes=1, top_k=1))
    ckpt = tmp_path / "models" / "seeds" / "mo_td3_full_s3.pt"
    assert ckpt.exists() and (tmp_path / "results" / "training" / "mo_td3_full_s3.json").exists()
    assert MOTD3Agent.load(str(ckpt)).flags["use_topology"] is True
    assert train_agent.checkpoint_path("mo_td3_full", 0) == str(tmp_path / "models" / "mo_td3_full.pt")
    hp = train_agent.parse_hparams(["n_step=1", "demo_cpus=0.5,1.0", "bc_floor=0.2"])
    assert hp.n_step == 1 and hp.demo_cpus == (0.5, 1.0) and hp.bc_floor == 0.2
    with pytest.raises(SystemExit):
        train_agent.parse_hparams(["nonsense=1"])


# -------------------------------------------------------------------- pareto
def test_pareto_dominance_front_and_hypervolume():
    a = dict(p95_latency=50, mean_cost=2.0, slo_violation_rate=0.0)
    b = dict(p95_latency=80, mean_cost=2.5, slo_violation_rate=0.1)      # dominated by a
    c = dict(p95_latency=120, mean_cost=1.0, slo_violation_rate=0.05)    # cheaper: non-dominated
    assert pareto.dominates(a, b) and not pareto.dominates(b, a)
    assert not pareto.dominates(a, c) and not pareto.dominates(c, a)
    assert pareto.pareto_front([a, b, c]) == [True, False, True]
    assert pareto.pareto_front([a, dict(a)]) == [True, True]             # ties both stay
    hv = lambda pts: pareto.hypervolume_2d(pts, budget=4.0)
    assert hv([a]) == pytest.approx((1 - 0.0) * (1 - 0.5))
    assert hv([a, c]) > hv([a]) and hv([a, b]) == pytest.approx(hv([a]))      # dominated points add nothing
    assert hv([]) == 0.0


def test_one_agent_sweeps_the_preference_simplex():
    agent = MOTD3Agent(seed=0)
    points = pareto.sweep(agent, DEMO6, episodes=1, grid_step=0.5, include_baselines=False, log=lambda *a: None)
    assert len(points) == 6 and all(p["family"] == "mo_td3" for p in points)
    assert all("preference" in p and abs(sum(p["preference"]) - 1) < 1e-9 for p in points)
    assert any(p["pareto_optimal"] for p in points)


# ------------------------------------------------------------ shipped artefacts
def _results():
    path = config.RESULTS_PATH
    if not os.path.exists(path):
        pytest.skip("no shipped results yet")
    return json.load(open(path))


def test_shipped_checkpoints_load_and_run_on_seen_and_unseen_topologies():
    agent = evaluate.load_arm("mo_td3_full")
    if agent is None:
        pytest.skip("no shipped checkpoint")
    for topo in (DEMO6, TRAINTICKET14):
        r = run_episode(RLController(agent, topo.services), topo, 9, "burst", obs_noise=0.05)
        assert np.isfinite(r["p95_latency"]) and r["mean_cost"] > 0


def test_shipped_results_reproduce_exactly():
    """Re-runs the baselines and the headline agent and asserts the committed JSON still matches."""
    res = _results()
    n = res["meta"]["episodes"]
    rows = res["demo6"]
    check = {"static": lambda: RuleController("static", S), "hpa": lambda: RuleController("hpa", S)}
    for name, make in check.items():
        got = evaluate_controller(make, DEMO6, n_episodes=n)
        for k in ("p95_latency", "slo_violation_rate", "mean_cost", "scalarized"):
            assert got[k] == pytest.approx(rows[name][k], rel=1e-9), (name, k)
    agent = evaluate.load_arm("mo_td3_full")
    if agent is not None and "mo_td3_full" in rows:
        got = evaluate_controller(lambda: RLController(agent, S, name="mo_td3_full"), DEMO6, n_episodes=n)
        for k in ("p95_latency", "slo_violation_rate", "mean_cost", "scalarized"):
            assert got[k] == pytest.approx(rows["mo_td3_full"][k], rel=1e-6), k


def test_shipped_pareto_front_is_internally_consistent():
    path = config.PARETO_PATH
    if not os.path.exists(path):
        pytest.skip("no shipped pareto results")
    blob = json.load(open(path))
    pts = blob["points"]
    flags = pareto.pareto_front(pts)
    assert flags == [p["pareto_optimal"] for p in pts]
    budget = config.BUDGET_PER_SERVICE * 6
    assert blob["families"]["mo_td3"]["hypervolume"] == pytest.approx(
        pareto.hypervolume_2d([p for p in pts if p["family"] == "mo_td3"], budget))
