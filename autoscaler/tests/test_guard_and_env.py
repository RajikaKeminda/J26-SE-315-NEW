"""Safety guard, action encoding, and the RL environment."""

import numpy as np
import pytest

from src import config
from src.rl_agent import rewards
from src.rl_agent.env import (AutoscalerEnv, decode_action, encode_executed, resource_to_u,
                              topology_base_rps, u_to_resource)
from src.safety.guards import SafetyGuard, pod_cost, quantize
from src.topologies import DEMO6, TRAINTICKET14

S = ["a", "b", "c"]


def cur(r=2, cpu=0.5, mem=128.0, mu=0.4, load=0.5):
    return {"replicas": r, "cpu": cpu, "mem": mem, "mem_utilization": mu, "offered_load": load}


def act(dr=0, cpu=0.5, mem=128.0):
    return {"dr": dr, "cpu": cpu, "mem": mem}


def run(guard, proposed, current, step=100, **kw):
    return guard.filter(proposed, current, step, **kw)


# ---------------------------------------------------------------------- guard
def test_bounds_clamp_rather_than_discard():
    g = SafetyGuard(S)
    ap, why = run(g, {"a": act(dr=3)}, {"a": cur(r=7)})
    assert ap["a"]["dr"] == 1 and "clamped" in why["a"]                     # +3 with room for +1 -> +1
    ap, _ = run(g, {"a": act(dr=-3)}, {"a": cur(r=2)}, step=500)
    assert ap["a"]["dr"] == -1                                              # never below MIN_REPLICAS
    ap, _ = run(g, {"a": act(cpu=50, mem=1e6)}, {"a": cur()}, step=900)
    assert ap["a"]["cpu"] == config.CPU_MAX and ap["a"]["mem"] == config.MEM_MAX_MB


def test_scale_down_is_slower_than_scale_up():
    g = SafetyGuard(S)
    assert run(g, {"a": act(dr=-1)}, {"a": cur()}, step=10)[0]["a"]["dr"] == -1
    ap, why = run(g, {"a": act(dr=-1)}, {"a": cur()}, step=11)
    assert ap["a"]["dr"] == 0 and "cooldown" in why["a"]
    assert run(g, {"a": act(dr=+2)}, {"a": cur()}, step=11)[0]["a"]["dr"] == 2          # up is never delayed
    assert run(g, {"a": act(dr=-1)}, {"a": cur()}, step=10 + config.SCALE_DOWN_COOLDOWN_STEPS)[0]["a"]["dr"] == -1


def test_resize_deadband_and_cooldown():
    g = SafetyGuard(S)
    small = run(g, {"a": act(cpu=0.5 * 1.05)}, {"a": cur()}, step=20)[0]["a"]
    assert small["cpu"] == 0.5, "changes inside the deadband are ignored (no pointless restarts)"
    big = run(g, {"a": act(cpu=1.0)}, {"a": cur()}, step=20)[0]["a"]
    assert big["cpu"] == 1.0
    ap, why = run(g, {"a": act(cpu=0.25)}, {"a": cur(cpu=1.0)}, step=21)
    assert ap["a"]["cpu"] == 1.0 and "resize cooldown" in why["a"]
    assert run(g, {"a": act(cpu=0.25)}, {"a": cur(cpu=1.0)},
               step=20 + config.RESIZE_COOLDOWN_STEPS)[0]["a"]["cpu"] == 0.25


def test_memory_floor_protects_against_self_inflicted_oom():
    g = SafetyGuard(S)
    ap, why = run(g, {"a": act(mem=96.0)}, {"a": cur(mem=256.0, mu=0.5)})     # working set 128 MiB
    assert ap["a"]["mem"] >= 160.0 and "memory floor" in why["a"]


def test_budget_blocks_the_least_needy_scale_ups_first():
    per_pod = pod_cost(0.5, 128.0)
    g = SafetyGuard(S, budget=6 * per_pod + 1e-6)
    current = {s: cur(r=2, load=l) for s, l in zip(S, (0.9, 0.5, 0.1))}
    ap, why = run(g, {s: act(dr=+1) for s in S}, current)
    assert all(ap[s]["dr"] == 0 for s in S), "no headroom at all: every scale-up is blocked"
    g2 = SafetyGuard(S, budget=7 * per_pod + 1e-6)                            # room for exactly one more pod
    ap2, why2 = run(g2, {s: act(dr=+1) for s in S}, current)
    assert ap2["a"]["dr"] == 1 and ap2["b"]["dr"] == 0 and ap2["c"]["dr"] == 0
    assert "budget" in why2["c"]


def test_physical_only_skips_policy_but_keeps_bounds_and_budget():
    g = SafetyGuard(S, budget=1e9)
    ap, _ = run(g, {"a": act(dr=-1)}, {"a": cur()}, step=5, physical_only=True)
    ap2, why = run(g, {"a": act(dr=-1)}, {"a": cur()}, step=6, physical_only=True)
    assert ap["a"]["dr"] == ap2["a"]["dr"] == -1 and "a" not in why           # no cooldown applied
    assert run(g, {"a": act(dr=5)}, {"a": cur(r=8)}, physical_only=True)[0]["a"]["dr"] == 0


def test_circuit_breaker_opens_and_recovers():
    g = SafetyGuard(S)
    for i in range(config.CIRCUIT_BREAKER_VIOLATIONS - 1):
        g.observe_slo(999, 100, i)
    assert not g.breaker_open
    g.observe_slo(999, 100, 99)
    assert g.breaker_open
    g.observe_slo(50, 100, 100)
    assert not g.breaker_open
    g.observe_slo(999, 100, 101)
    assert not g.breaker_open, "violation streak restarts after recovery"


def test_staleness_and_readiness():
    import time
    g = SafetyGuard(S)
    assert g.is_stale(time.time() - config.METRIC_STALENESS_SECONDS - 1)
    assert not g.is_stale(time.time()) and not g.is_stale(None)
    assert g.readiness(0)["a"] == (0.0, 0.0)
    run(g, {"a": act(dr=-1, cpu=1.0)}, {"a": cur()}, step=10)
    rz, sd = g.readiness(11)["a"]
    assert 0 < rz <= 1 and 0 < sd <= 1
    assert g.readiness(10 + config.RESIZE_COOLDOWN_STEPS)["a"] == (0.0, 0.0)


def test_quantize():
    assert quantize(0.37, 0.05, 0.1, 1.0) == pytest.approx(0.35)
    assert quantize(5.0, 0.05, 0.1, 1.0) == 1.0 and quantize(0.0, 0.05, 0.1, 1.0) == 0.1


# ------------------------------------------------------------ action encoding
def test_resource_mapping_is_a_log_scale_bijection():
    for lo, hi in ((config.CPU_MIN, config.CPU_MAX), (config.MEM_MIN_MB, config.MEM_MAX_MB)):
        assert u_to_resource(-1, lo, hi) == pytest.approx(lo) and u_to_resource(1, lo, hi) == pytest.approx(hi)
        for v in np.geomspace(lo, hi, 7):
            assert u_to_resource(resource_to_u(v, lo, hi), lo, hi) == pytest.approx(v)
    mid = u_to_resource(0.0, 0.1, 1.0)
    assert mid == pytest.approx(np.sqrt(0.1 * 1.0))


def test_decode_then_encode_roundtrips_what_was_executed():
    env = AutoscalerEnv(DEMO6, seed=0)
    obs = env.reset("normal")
    n = len(DEMO6.services)
    idx = np.array([0, 1, 2, 3, 4, 6])
    cont = np.random.default_rng(0).uniform(-1, 1, (n, 2)).astype(np.float32)
    props = decode_action(DEMO6.services, obs["metrics"], idx, cont)
    assert [props[s]["dr"] for s in DEMO6.services] == [-3, -2, -1, 0, 1, 3]
    assert all(config.CPU_MIN <= p["cpu"] <= config.CPU_MAX for p in props.values())
    ridx, rcont = encode_executed(DEMO6.services, obs["metrics"], props)
    assert list(ridx) == list(idx)
    assert np.allclose(rcont[:, :2], cont, atol=0.12)                # quantisation only
    assert np.allclose(rcont[:, 2], (idx - 3) / 3)                   # replica delta / 3 (what the round head learns)


def test_replica_only_leaves_resources_alone():
    env = AutoscalerEnv(DEMO6, seed=0)
    obs = env.reset("normal")
    props = decode_action(DEMO6.services, obs["metrics"], np.zeros(6, int), np.ones((6, 2)), replica_only=True)
    assert all(p["cpu"] == obs["metrics"][s]["cpu"] and p["mem"] == obs["metrics"][s]["mem"]
               for s, p in props.items())


# ---------------------------------------------------------------- environment
def test_reward_is_a_penalty_vector_with_the_documented_structure():
    env = AutoscalerEnv(DEMO6, seed=0)
    obs = env.reset("normal")
    props = {s: {"dr": 0, "cpu": obs["metrics"][s]["cpu"], "mem": obs["metrics"][s]["mem"]} for s in DEMO6.services}
    obs, r, done, info = env.step_proposals(props)
    assert r.shape == (config.N_OBJECTIVES,) and (r <= 0).all()
    assert info["penalties"]["cost_norm"] == pytest.approx(info["cost"] / info["budget"])
    assert not done and env.t == 1


def test_slo_penalty_has_a_step_at_the_threshold_and_saturates():
    base = dict(metrics={"a": {"utilization": 0.5}}, services=["a"], cost=3.0, budget=6.0, slo_ms=250.0)
    pen = lambda p95, err=0.0: rewards.objective_penalties(
        base["metrics"], base["services"], {"p95_ms": p95, "error_rate": err}, base["cost"], base["budget"], 250.0)[0][0]
    assert pen(200) == 0.0
    assert pen(251) >= config.SLO_VIOLATION_STEP
    assert pen(300) < pen(400) < pen(10_000) == pytest.approx(1.0)
    assert pen(200, err=0.3) == pytest.approx(0.3)


def test_scalarize_and_preferences():
    v = np.array([-0.5, -0.2, -0.1])
    assert rewards.scalarize(v, [1, 0, 0]) == pytest.approx(-0.5)
    assert rewards.scalarize(v, [0.5, 0.3, 0.2]) == pytest.approx(-0.33)
    assert rewards.normalize_preference([2, 2, 0]).tolist() == [0.5, 0.5, 0.0]
    assert rewards.normalize_preference([0, 0, 0]).tolist() == pytest.approx([1 / 3] * 3)
    grid = rewards.preference_grid(0.25)
    assert len(grid) == 15 and all(abs(w.sum() - 1) < 1e-9 for w in grid)


def test_env_runs_a_full_episode_with_a_random_topology_each_time():
    env = AutoscalerEnv(topology=None, seed=4)
    sizes = set()
    for _ in range(5):
        obs = env.reset("burst")
        sizes.add(obs["X"].shape[0])
        done, steps = False, 0
        while not done:
            props = {s: {"dr": 0, "cpu": obs["metrics"][s]["cpu"], "mem": obs["metrics"][s]["mem"]}
                     for s in env.services}
            obs, r, done, info = env.step_proposals(props)
            steps += 1
        assert steps == config.EPISODE_LENGTH_STEPS
    assert len(sizes) > 1, "training must see different topologies"


def test_matched_seeds_give_identical_traffic_and_noise():
    a, b = AutoscalerEnv(DEMO6, seed=11), AutoscalerEnv(DEMO6, seed=11)
    oa, ob = a.reset("burst"), b.reset("burst")
    assert np.allclose(a.traffic, b.traffic) and np.allclose(oa["X"], ob["X"])


def test_guard_is_applied_inside_the_environment():
    env = AutoscalerEnv(DEMO6, seed=0)
    obs = env.reset("normal")
    cpu = {s: obs["metrics"][s]["cpu"] for s in DEMO6.services}
    wild = {s: {"dr": 3, "cpu": cpu[s], "mem": obs["metrics"][s]["mem"]} for s in DEMO6.services}
    for _ in range(4):
        obs, r, done, info = env.step_proposals(wild)
    assert info["total_replicas"] * pod_cost(0.25, 128.0) <= env.budget + 1e-6, "budget must hold"
    assert any("budget" in v or "clamped" in v for v in info["reasons"].values())
    assert all(m["replicas"] <= config.MAX_REPLICAS for m in info["metrics"].values())


def test_executed_actions_not_proposals_are_reported_for_learning():
    env = AutoscalerEnv(DEMO6, seed=0)
    obs = env.reset("normal")
    props = {s: {"dr": -3, "cpu": obs["metrics"][s]["cpu"], "mem": obs["metrics"][s]["mem"]} for s in DEMO6.services}
    _, _, _, info = env.step_proposals(props)
    idx, _ = info["executed"]
    assert (idx == 2).all(), "-3 on 2 replicas was clamped to -1 (index 2) before being executed"


def test_topology_base_rps_is_identical_for_demo6_and_scales_for_bigger_apps():
    assert topology_base_rps(DEMO6) == pytest.approx(config.BASE_RPS)
    assert topology_base_rps(TRAINTICKET14) > config.BASE_RPS      # more budget, cheaper requests



def test_opt_in_stabilisation_window_after_a_scale_up_and_while_the_newest_pod_is_young(monkeypatch):
    from src import config
    from src.safety.guards import SafetyGuard
    monkeypatch.setattr(config, "SCALE_DOWN_AFTER_UP_STEPS", 5)
    monkeypatch.setattr(config, "STEP_SECONDS", 30.0)
    g = SafetyGuard(services=["a"], budget=100.0)
    cur = {"a": {"replicas": 2, "cpu": 0.5, "mem": 256.0, "mem_utilization": 0.3}}
    up, _ = g.filter({"a": {"dr": 1, "cpu": 0.5, "mem": 256.0}}, cur, step=10)
    assert up["a"]["dr"] == 1
    down, why = g.filter({"a": {"dr": -1, "cpu": 0.5, "mem": 256.0}}, cur, step=12)
    assert down["a"]["dr"] == 0 and "after a scale-up" in why["a"]
    down, _ = g.filter({"a": {"dr": -1, "cpu": 0.5, "mem": 256.0}}, cur, step=15)          # the window has passed
    assert down["a"]["dr"] == -1
    # a restarted agent remembers nothing, but the cluster knows the newest pod is 90 s old (window = 150 s)
    fresh = SafetyGuard(services=["a"], budget=100.0)
    young = {"a": dict(cur["a"], youngest_pod_s=90.0)}
    down, why = fresh.filter({"a": {"dr": -1, "cpu": 0.5, "mem": 256.0}}, young, step=0)
    assert down["a"]["dr"] == 0 and "newest pod is only 90 s old" in why["a"]
    old = {"a": dict(cur["a"], youngest_pod_s=900.0)}
    assert SafetyGuard(services=["a"], budget=100.0).filter({"a": {"dr": -1, "cpu": 0.5, "mem": 256.0}}, old, step=0)[0]["a"]["dr"] == -1


def test_stabilisation_window_is_off_by_default_so_shipped_results_stay_valid():
    from src import config
    assert config.SCALE_DOWN_AFTER_UP_STEPS == 0


def test_revoke_restores_the_timestamps_the_last_filter_overwrote(monkeypatch):
    from src import config
    from src.safety.guards import SafetyGuard
    monkeypatch.setattr(config, "SCALE_DOWN_AFTER_UP_STEPS", 5)
    g = SafetyGuard(services=["a"], budget=100.0)
    cur = {"a": {"replicas": 2, "cpu": 0.5, "mem": 256.0, "mem_utilization": 0.3}}
    g.filter({"a": {"dr": 1, "cpu": 0.5, "mem": 256.0}}, cur, step=3)               # a real scale-up at step 3
    g.filter({"a": {"dr": 1, "cpu": 1.0, "mem": 256.0}}, cur, step=9)               # approved at step 9 ...
    g.revoke("a", scale=True, resize=True)                                          # ... but vetoed afterwards
    assert g._last_scale_up_step["a"] == 3 and "a" not in g._last_resize_step      # back to what really happened
    assert g.filter({"a": {"dr": -1, "cpu": 0.5, "mem": 256.0}}, cur, step=9)[0]["a"]["dr"] == -1   # 6 steps after the real scale-up: allowed
    g.revoke("unknown")                                                              # harmless


def test_scale_down_feature_covers_the_longest_blocking_rule(monkeypatch):
    from src import config
    from src.safety.guards import SafetyGuard
    monkeypatch.setattr(config, "SCALE_DOWN_COOLDOWN_STEPS", 3)
    monkeypatch.setattr(config, "SCALE_DOWN_AFTER_UP_STEPS", 10)
    g = SafetyGuard(services=["a"], budget=100.0)
    cur = {"a": {"replicas": 2, "cpu": 0.5, "mem": 256.0, "mem_utilization": 0.3}}
    g.filter({"a": {"dr": 1, "cpu": 0.5, "mem": 256.0}}, cur, step=0)
    assert g.readiness(5)["a"][1] == pytest.approx(0.5)          # still blocked for 5 more steps of the 10, not "ready" after 3


def test_a_locked_memory_is_left_exactly_as_it_is_not_rounded_to_the_grid(monkeypatch):
    """With ALLOW_MEMORY_RESIZE=0 a CPU resize used to carry the memory rounded to the memory grid (on Train-Ticket 800 -> 768 MiB): a change nobody asked for."""
    from src import config
    from src.safety.guards import SafetyGuard
    cur = {"s": {"replicas": 1, "cpu": 0.3, "mem": 200.0, "utilization": 0.9, "mem_utilization": 0.4, "ready_replicas": 1}}
    prop = {"s": {"dr": 0, "cpu": 1.0, "mem": 200.0}}
    monkeypatch.setattr(config, "ALLOW_MEMORY_RESIZE", False)
    approved, _ = SafetyGuard(services=["s"], budget=float("inf")).filter(prop, cur, step=100)
    assert approved["s"]["cpu"] == 1.0 and approved["s"]["mem"] == 200.0
    monkeypatch.setattr(config, "ALLOW_MEMORY_RESIZE", True)
    approved, _ = SafetyGuard(services=["s"], budget=float("inf")).filter(prop, cur, step=100)
    assert approved["s"]["mem"] == 192.0, "unlocked, memory still snaps to its 32 MiB grid (unchanged behaviour)"
