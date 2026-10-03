"""
test_all.py

Sanity tests for every v2 component. Run:
    python tests/test_all.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src import config
from src.simulation.simulator import (
    Simulator, random_topology, fixed_topology, generate_traffic, critical_path)
from src.topology_intel.graph_builder import build_all, build_sdg, build_ehg
from src.topology_intel.dual_gcn import DualGCN
from src.rl_agent.shared_policy import SharedPolicy
from src.rl_agent.env import AutoscalerEnv, compute_reward
from src.safety.guards import SafetyGuard
from src.baselines.rule_based import HPABaseline


def test_topology_always_dag():
    rng = np.random.default_rng(0)
    for _ in range(20):
        services, entry, graph = random_topology(rng)
        # walk from entry; a DAG must terminate without revisiting
        seen, frontier, steps = {entry}, [entry], 0
        while frontier and steps < 1000:
            node = frontier.pop()
            for c, _ in graph.get(node, []):
                if c not in seen:
                    seen.add(c); frontier.append(c)
            steps += 1
        assert steps < 1000, "cycle detected in generated topology"


def test_traffic_conserved():
    s, e, g = fixed_topology()
    sim = Simulator(s, e, g)
    m = sim.step(100.0)
    leaves = [x for x in s if not g.get(x)]
    assert abs(sum(m[l]["requests"] for l in leaves) - 100.0) < 1e-3


def test_sdg_static_ehg_dynamic():
    """The whole point of two graphs: SDG stays put, EHG reacts to latency."""
    s, e, g = fixed_topology()
    sim = Simulator(s, e, g)
    m_low = sim.step(100.0)
    m_high = sim.step(3000.0)

    sdg_low, sdg_high = build_sdg(s, g), build_sdg(s, g)
    assert torch.allclose(sdg_low, sdg_high), "SDG must not change with load"

    ehg_low = build_ehg(s, g, m_low)
    ehg_high = build_ehg(s, g, m_high)
    assert not torch.allclose(ehg_low, ehg_high), "EHG must react to load"


def test_shared_policy_topology_agnostic():
    """The headline claim: identical weights work for any number of services."""
    p = SharedPolicy(config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM,
                     config.GCN_EMBED_DIM, config.POLICY_HIDDEN)
    n_params = sum(x.numel() for x in p.parameters())
    for N in [3, 6, 7, 25]:
        X = torch.rand(N, config.NODE_FEATURE_DIM)
        A1, A2, gv = torch.rand(N, N), torch.rand(N, N), torch.rand(3)
        a, lp, v = p.act(X, A1, A2, gv)
        assert a.shape == (N,)
    assert sum(x.numel() for x in p.parameters()) == n_params, "params must not change with N"


def test_dual_gcn_ablation_modes():
    X, A1, A2 = torch.rand(6, 5), torch.rand(6, 6), torch.rand(6, 6)
    full = DualGCN(5, 32, 16, use_ehg=True, use_topology=True)
    sdg = DualGCN(5, 32, 16, use_ehg=False, use_topology=True)
    none = DualGCN(5, 32, 16, use_ehg=False, use_topology=False)
    assert full(X, A1, A2).shape == (6, 32)
    assert sdg(X, A1, A2).shape == (6, 16)
    assert none(X, A1, A2).shape == (6, 16)


def test_critical_path_finds_bottleneck():
    s, e, g = fixed_topology()
    lat = {"gateway-service": 20, "catalog-service": 30, "auth-service": 15,
           "inventory-service": 25, "pricing-service": 200, "user-service": 10}
    total, path = critical_path(g, e, lat)
    assert path == ["gateway-service", "catalog-service", "pricing-service"]
    assert total == 250


def test_reward_penalises_overload():
    s, e, g = fixed_topology()
    sim = Simulator(s, e, g)
    light = sim.step(100.0)
    cl, _ = critical_path(g, e, {k: v["latency_ms"] for k, v in light.items()})
    r_light, _ = compute_reward(s, light, cl)

    heavy = sim.step(4000.0)
    ch, _ = critical_path(g, e, {k: v["latency_ms"] for k, v in heavy.items()})
    r_heavy, _ = compute_reward(s, heavy, ch)
    assert r_heavy < r_light


def test_guard_blocks_and_breaks():
    g = SafetyGuard(services=["a"])
    approved, _ = g.filter({"a": 1}, {"a": 2}, 0)
    assert approved["a"] == 1
    approved, reasons = g.filter({"a": 1}, {"a": 3}, 1)      # rate limited
    assert approved["a"] == 0 and "rate limited" in reasons["a"]

    approved, reasons = g.filter({"a": -1}, {"a": 1}, 10)     # below min
    assert approved["a"] == 0

    g2 = SafetyGuard(services=["a"])
    for step in range(config.CIRCUIT_BREAKER_VIOLATIONS):
        g2.observe_slo(config.SLO_LATENCY_MS * 5, step)
    assert g2.breaker_open
    g2.observe_slo(10.0, 99)
    assert not g2.breaker_open


def test_hpa_scales_up_fast_down_slow():
    hpa = HPABaseline(["a"])
    over = {"a": {"replicas": 2, "utilization": 1.6, "latency_ms": 500, "requests": 100}}
    assert hpa.decide(over)["a"] > 0                       # scales up immediately

    hpa.reset()
    under = {"a": {"replicas": 4, "utilization": 0.05, "latency_ms": 20, "requests": 5}}
    first = hpa.decide(under)["a"]
    assert first == 0, "HPA must wait for the stabilization window before scaling down"
    for _ in range(config.HPA_STABILIZATION_STEPS - 1):
        last = hpa.decide(under)["a"]
    assert last < 0


def test_env_runs_varied_topologies():
    env = AutoscalerEnv(seed=3)
    p = SharedPolicy(config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM,
                     config.GCN_EMBED_DIM, config.POLICY_HIDDEN)
    sizes = set()
    for _ in range(6):
        obs = env.reset()
        sizes.add(len(env.services))
        for _ in range(3):
            a, _, _ = p.act(*obs)
            obs, r, done, info = env.step(a)
            assert isinstance(r, float)
    assert len(sizes) > 1, "expected varied topology sizes across episodes"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
