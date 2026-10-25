"""The simulator's physics, and its fit to measurements from a real cluster."""

import numpy as np
import pytest

from src import config
from src.simulation import calibration, service_model as sm
from src.simulation.simulator import Simulator, generate_traffic, random_topology
from src.topologies import DEMO6, TRAINTICKET14


@pytest.fixture(autouse=True)
def default_params():
    sm.set_params(None)
    yield
    sm.set_params(None)


def settle(sim, rps, steps=8):
    for _ in range(steps):
        m = sim.step(rps)
    return m


# ------------------------------------------------------------ service model
def test_capacity_scales_with_cpu_and_saturates_at_one_core():
    c = {cpu: sm.capacity_rps(cpu, 15.0) for cpu in (0.1, 0.25, 0.5, 1.0, 2.0)}
    assert c[0.25] == pytest.approx(2.5 * c[0.1], rel=0.3)      # minus a small idle-CPU floor
    assert c[1.0] == pytest.approx(2.0 * c[0.5], rel=0.1)
    assert c[2.0] == c[1.0], "Python behind the GIL cannot use more than one core"
    assert 55 < c[1.0] < 70, "measured ~62 req/s on the real cluster"


def test_latency_grows_with_load_and_is_finite_at_overload():
    prev = 0.0
    for rho in (0.0, 0.2, 0.5, 0.8, 0.9, 5.0):
        lat = sm.mean_latency_ms(rho, 0.5, 15.0)
        assert lat >= prev and np.isfinite(lat)
        prev = lat


def test_tiny_cpu_limits_are_throttled_even_when_idle():
    """CFS: a 0.1-core limit gives 10 ms of quota per 100 ms; a 16 ms request needs two periods."""
    assert sm.base_latency_ms(0.1, 15.0) > sm.base_latency_ms(0.25, 15.0) + 80
    assert sm.base_latency_ms(0.25, 15.0) == pytest.approx(sm.base_latency_ms(1.0, 15.0))


def test_tail_factor_bounds_and_monotonicity():
    p = sm.params()
    assert sm.tail_factor(0.0) == pytest.approx(p.tail_idle)
    assert sm.tail_factor(10.0) == pytest.approx(p.tail_busy)
    assert sm.tail_factor(0.1) < sm.tail_factor(0.3) <= sm.tail_factor(0.9)


def test_params_roundtrip_and_override(tmp_path):
    p = sm.ModelParams(kappa=1.23)
    sm.set_params(p)
    assert sm.params().kappa == 1.23
    sm.set_params(None)
    assert sm.params().kappa == pytest.approx(sm.load_params().kappa)
    assert sm.load_params(str(tmp_path / "missing.json")).kappa == sm.ModelParams().kappa


# ------------------------------------------------- fit against REAL measurements
def test_fit_predicts_the_real_cluster_measurements():
    """The sim-to-real evidence: the shipped parameters reproduce measurements taken from a
    real Kubernetes deployment of the service (calibration/leaf_cpu*.json)."""
    pts = calibration.load_points()
    assert len(pts) >= 30, "calibration data missing"
    p = sm.params()
    rep = calibration.evaluate(pts, p)
    steady = rep["steady (fitted region)"]
    assert steady["mean_MAPE_%"] < 35 and steady["p95_MAPE_%"] < 40
    assert steady["p95_within_2x"] > 0.9
    for cpu, c in calibration.capacity_check(pts, p).items():
        assert c["model_capacity_rps"] == pytest.approx(c["observed_max_rps"], rel=0.15), cpu


def test_refit_recovers_similar_parameters():
    p = calibration.fit(calibration.load_points())
    shipped = sm.params()
    assert p.kappa == pytest.approx(shipped.kappa, rel=0.25)
    assert p.cpu_overhead_ms == pytest.approx(shipped.cpu_overhead_ms, abs=1.0)


# ---------------------------------------------------------------- simulator
def test_traffic_is_conserved_through_the_call_graph():
    sim = Simulator(DEMO6)
    m = sim.step(40.0)
    leaves = [s for s in DEMO6.services if not DEMO6.call_graph[s]]
    assert sum(m[s]["requests"] for s in leaves) == pytest.approx(40.0)
    t = Simulator(TRAINTICKET14)
    mt = t.step(30.0)
    assert sum(mt[s]["requests"] for s in TRAINTICKET14.services if not TRAINTICKET14.call_graph[s]) \
        == pytest.approx(30.0)


def test_deterministic_given_seed_and_noise_changes_observations_only():
    a, b = Simulator(DEMO6, rng=np.random.default_rng(5), obs_noise=0.1), \
           Simulator(DEMO6, rng=np.random.default_rng(5), obs_noise=0.1)
    assert a.step(20.0)["__e2e__"] == b.step(20.0)["__e2e__"]
    quiet = Simulator(DEMO6)
    noisy = Simulator(DEMO6, rng=np.random.default_rng(1), obs_noise=0.2)
    qm, nm = quiet.step(20.0), noisy.step(20.0)
    assert qm["gateway-service"]["latency_ms"] != nm["gateway-service"]["latency_ms"]
    assert quiet.state["gateway-service"].backlog == noisy.state["gateway-service"].backlog   # state is not noisy


def test_more_replicas_or_cpu_means_lower_latency_and_higher_cost():
    def run(r, c):
        sim = Simulator(DEMO6)
        for st in sim.state.values():
            st.replicas, st.cpu = r, c
        return settle(sim, 24.0)["__e2e__"]["p95_ms"], sim.cost()
    (p_small, c_small), (p_more_pods, c_pods), (p_more_cpu, c_cpu) = run(1, 0.25), run(3, 0.25), run(1, 1.0)
    assert p_more_pods < p_small and p_more_cpu < p_small
    assert c_pods > c_small and c_cpu > c_small


def test_overload_builds_a_backlog_that_outlives_the_burst_and_drops_requests():
    sim = Simulator(DEMO6)
    settle(sim, 10.0, 2)
    hit = settle(sim, 200.0, 6)
    assert hit["gateway-service"]["backlog"] > 0
    assert hit["__e2e__"]["error_rate"] > 0, "requests older than the client timeout are dropped"
    after = sim.step(10.0)
    assert after["gateway-service"]["latency_ms"] > 500, "the queue is still draining"
    for _ in range(40):
        recovered = sim.step(10.0)
    assert recovered["gateway-service"]["latency_ms"] < 150


def test_scale_up_takes_a_pod_startup_time_to_help():
    sim = Simulator(DEMO6)
    for st in sim.state.values():
        st.replicas = 1
    before = settle(sim, 25.0, 3)["gateway-service"]
    sim.apply("gateway-service", replica_delta=+2)
    m = sim.step(25.0)["gateway-service"]
    assert m["replicas"] == 3 and m["pending_replicas"] == 2        # starting, not yet counted as ready
    later = settle(sim, 25.0, 3)["gateway-service"]
    assert later["pending_replicas"] == 0 and later["latency_ms"] < before["latency_ms"]


def test_resize_is_a_rolling_restart_that_takes_effect_late():
    sim = Simulator(DEMO6)
    sim.apply("gateway-service", cpu=1.0)
    assert sim.target_allocation("gateway-service")[0] == 1.0
    assert sim.state["gateway-service"].cpu == 0.25                   # not applied yet
    for _ in range(4):
        sim.step(10.0)
    assert sim.state["gateway-service"].cpu == 1.0 and sim.state["gateway-service"].resize is None
    sim.apply("gateway-service", cpu=0.5)
    sim.apply("gateway-service", cpu=1.0)                              # back to current -> cancels
    assert sim.state["gateway-service"].resize is None


def test_apply_clamps_to_bounds():
    sim = Simulator(DEMO6)
    sim.apply("gateway-service", replica_delta=+99, cpu=99.0, mem=99999.0)
    st = sim.state["gateway-service"]
    assert st.replicas == config.MAX_REPLICAS
    assert st.resize[0] == config.CPU_MAX and st.resize[1] == config.MEM_MAX_MB
    sim.apply("gateway-service", replica_delta=-99, cpu=0.0001, mem=1.0)
    assert sim.state["gateway-service"].replicas == config.MIN_REPLICAS
    assert sim.state["gateway-service"].resize[0] == config.CPU_MIN


def test_too_little_memory_oom_kills_capacity():
    ok, starved = Simulator(DEMO6), Simulator(DEMO6)
    starved.state["gateway-service"].mem = 40.0                      # below the ~55 MiB resident set
    a, b = settle(ok, 20.0)["gateway-service"], settle(starved, 20.0)["gateway-service"]
    assert b["mem_utilization"] > 1.0 and b["latency_ms"] > a["latency_ms"] * 2


def test_e2e_percentiles_are_ordered_and_a_mixture_not_a_single_chain():
    sim = Simulator(DEMO6)
    e = settle(sim, 12.0)["__e2e__"]
    assert 0 < e["p50_ms"] < e["p95_ms"] < e["p99_ms"]
    assert e["mean_ms"] < e["p95_ms"]
    # more traffic on the slow branch shifts the whole distribution
    heavy = Simulator(DEMO6)
    heavy.state["user-service"].replicas = 1
    heavy.state["user-service"].cpu = 0.1
    assert settle(heavy, 30.0)["__e2e__"]["p95_ms"] > settle(Simulator(DEMO6), 30.0)["__e2e__"]["p95_ms"]


def test_inclusive_latency_includes_downstream():
    m = settle(Simulator(DEMO6), 15.0)
    assert m["gateway-service"]["latency_incl_ms"] > m["gateway-service"]["latency_ms"]
    assert m["inventory-service"]["latency_incl_ms"] == pytest.approx(m["inventory-service"]["latency_ms"])
    assert m["gateway-service"]["latency_incl_p95_ms"] > m["gateway-service"]["latency_incl_ms"]


@pytest.mark.parametrize("pattern", ["normal", "burst", "diurnal", "rampup", "random"])
def test_traffic_patterns_shape(pattern):
    rng = np.random.default_rng(0)
    t = generate_traffic(pattern, 40, 12.0, rng)
    assert len(t) == 40 and (t > 0).all()
    if pattern == "burst":
        assert t.max() > 2.0 * 12.0 and t[0] < 2 * 12.0
    if pattern == "rampup":
        assert t[-1] > 2.5 * t[0]
    with pytest.raises(ValueError):
        generate_traffic("nope", 10, 12.0, rng)


def test_works_on_random_topologies_without_blowing_up():
    rng = np.random.default_rng(2)
    for _ in range(15):
        sim = Simulator(random_topology(rng), rng=rng, obs_noise=0.05)
        for rps in (3.0, 60.0, 400.0):
            m = sim.step(rps)
            assert np.isfinite(m["__e2e__"]["p95_ms"])


def test_hot_variant_turns_exactly_one_service_into_the_bottleneck():
    import numpy as np
    from src.topologies import TRAINTICKET_SIM as T, hot_variant, visit_ratios
    ratio = visit_ratios(T)
    assert ratio[T.entry] == 1.0 and ratio["ts-order-service"] > 0.3 and abs(ratio["ts-gateway-service"] - 1.0) < 1e-9
    seen = set()
    for seed in range(40):
        topo, hot = hot_variant(T, np.random.default_rng(seed), 13.0, (0.7, 1.6))
        seen.add(hot)
        changed = [s for s in T.services if abs(topo.work(s) - T.work(s)) > 1e-9]
        assert changed in ([hot], []), "only the hot service changes (or nothing, when it is already that heavy)"
        assert hot != T.entry and topo.name.endswith(hot)
        assert topo.work(hot) >= T.work(hot) and topo.work(hot) <= 8.0 * T.work(hot) + 1e-9
        if changed:
            demand = 13.0 * ratio[hot] * topo.work(hot) / 1000.0
            assert 0.7 - 1e-6 <= demand <= 1.6 + 1e-6 or topo.work(hot) >= 8.0 * T.work(hot) - 1e-9    # needs 0.7-1.6 cores, unless capped
    assert len(seen) >= 6, "a different service is chosen each time"


def test_bottleneck_scenario_makes_one_service_the_only_saturated_one():
    import numpy as np
    from src.rl_agent.env import AutoscalerEnv, topology_base_rps
    from src.topologies import DEMO6, bottleneck_scenario, shift_mix, visit_ratios
    base = topology_base_rps(DEMO6)
    shifted = shift_mix(DEMO6, "pricing-service", 4.0)
    assert visit_ratios(shifted)["pricing-service"] > visit_ratios(DEMO6)["pricing-service"] * 1.5
    assert all(abs(sum(w for _, w in t) - 1.0) < 1e-9 for t in shifted.call_graph.values() if t), "shares still sum to 1"
    for hot in DEMO6.services:
        topo, h = bottleneck_scenario(DEMO6, np.random.default_rng(0), base, hot=hot)
        assert topo is not None and h == hot and topo.initial_cpu == {hot: 0.1} and topo.load_scale >= 1.0
        env = AutoscalerEnv(topology=topo, seed=0)
        obs = env.reset("normal", topology=topo)
        util = {s: m["utilization"] for s, m in obs["metrics"].items()}
        assert util[hot] >= 0.99, f"{hot} must start saturated, got {util[hot]}"
        assert max(v for s, v in util.items() if s != hot) < 0.9, f"no other service may saturate: {util}"
    infeasible, _ = bottleneck_scenario(DEMO6, np.random.default_rng(0), base, hot="pricing-service", max_scale=1.0, demand_range=(50.0, 60.0))
    assert infeasible is None
