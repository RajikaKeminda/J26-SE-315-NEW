"""Topologies, critical-path search, and the SDG / EHG graph construction."""

import numpy as np
import pytest

from src import config
from src.simulation.simulator import Simulator, random_topology, topology_depth
from src.topologies import DEMO6, TRAINTICKET14, Topology, get_topology, validate
from src.topology_intel.critical_path import critical_path_dfs, path_distribution, reachable
from src.topology_intel.graph_builder import (
    build_ehg, build_observation, build_sdg, node_features, symmetric_normalize)


# ----------------------------------------------------------------- topologies
@pytest.mark.parametrize("topo", [DEMO6, TRAINTICKET14])
def test_shipped_topologies_are_valid_dags(topo):
    validate(topo)


def test_validate_rejects_bad_topologies():
    base = dict(name="x", services=["a", "b"], entry="a")
    with pytest.raises(ValueError, match="cycle"):
        validate(Topology(**base, call_graph={"a": [("b", 1.0)], "b": [("a", 1.0)]}))
    with pytest.raises(ValueError, match="sum to 1"):
        validate(Topology(**base, call_graph={"a": [("b", 0.5)], "b": []}))
    with pytest.raises(ValueError, match="unreachable"):
        validate(Topology(**base, call_graph={"a": [], "b": []}))
    with pytest.raises(ValueError, match="unknown callee"):
        validate(Topology(**base, call_graph={"a": [("zzz", 1.0)], "b": []}))
    with pytest.raises(KeyError):
        get_topology("nope")


def test_random_topologies_are_valid_and_varied():
    rng = np.random.default_rng(0)
    sizes = set()
    for _ in range(60):
        t = random_topology(rng)
        validate(t)
        sizes.add(len(t.services))
        assert all(8.0 <= t.work(s) <= 26.0 for s in t.services)
    assert sizes == {4, 5, 6, 7, 8, 9}


def test_demo_topology_matches_the_deployed_manifests():
    """The topology the agent reasons about must be the one that is deployed."""
    import re
    import os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for svc in DEMO6.services:
        text = open(os.path.join(root, "k8s", "01-services", f"{svc}.yaml")).read()
        downstream = re.search(r'name: DOWNSTREAM\n\s+value: "([^"]*)"', text).group(1)
        parsed = [(p.split("|")[0].split("//")[1].split(":")[0], float(p.split("|")[1]))
                  for p in downstream.split(",") if p]
        assert parsed == DEMO6.call_graph[svc]


# --------------------------------------------------------------- critical path
def _brute_force(graph, entry, lat):
    best = (-1.0, None)
    for path, _ in path_distribution(graph, entry):
        total = sum(lat[s] for s in path)
        if total > best[0]:
            best = (total, path)
    return best


def test_critical_path_picks_the_heaviest_chain():
    lat = {"gateway-service": 10, "catalog-service": 50, "auth-service": 20,
           "inventory-service": 5, "pricing-service": 80, "user-service": 30}
    total, path = critical_path_dfs(DEMO6.call_graph, DEMO6.entry, lat)
    assert path == ["gateway-service", "catalog-service", "pricing-service"] and total == 140
    lat["user-service"] = 500
    assert critical_path_dfs(DEMO6.call_graph, DEMO6.entry, lat)[1] == \
        ["gateway-service", "auth-service", "user-service"]


def test_critical_path_dfs_equals_brute_force_on_random_dags():
    rng = np.random.default_rng(3)
    for _ in range(40):
        t = random_topology(rng)
        lat = {s: float(rng.uniform(1, 100)) for s in t.services}
        total, path = critical_path_dfs(t.call_graph, t.entry, lat)
        bf_total, bf_path = _brute_force(t.call_graph, t.entry, lat)
        assert total == pytest.approx(bf_total) and path == bf_path


def test_critical_path_handles_shared_callees_and_rejects_cycles():
    total, path = critical_path_dfs(TRAINTICKET14.call_graph, TRAINTICKET14.entry,
                                    {s: 10.0 for s in TRAINTICKET14.services})
    assert path[0] == "ts-gateway" and total == 10.0 * len(path)
    with pytest.raises(ValueError, match="cycle"):
        critical_path_dfs({"a": [("b", 1.0)], "b": [("a", 1.0)]}, "a", {})


def test_path_distribution_is_a_probability_distribution():
    for topo in (DEMO6, TRAINTICKET14):
        paths = path_distribution(topo.call_graph, topo.entry)
        assert sum(p for _, p in paths) == pytest.approx(1.0)
        assert all(path[0] == topo.entry for path, _ in paths)
    # demo6: 4 chains with the product of shares as probability
    probs = {tuple(p): pr for p, pr in path_distribution(DEMO6.call_graph, DEMO6.entry)}
    assert probs[("gateway-service", "catalog-service", "inventory-service")] == pytest.approx(0.3)
    assert probs[("gateway-service", "auth-service", "user-service")] == pytest.approx(0.4)


def test_reachable_and_depth():
    assert reachable(DEMO6.call_graph, DEMO6.entry)[0] == DEMO6.entry
    assert set(reachable(DEMO6.call_graph, DEMO6.entry)) == set(DEMO6.services)
    assert topology_depth(DEMO6) == 3 and topology_depth(TRAINTICKET14) == 5
    assert config.slo_for_depth(3) == pytest.approx(250.0)


# ---------------------------------------------------------------------- graphs
def test_symmetric_normalize_matches_formula():
    A = np.array([[0, 1], [1, 0]], dtype=np.float32)
    out = symmetric_normalize(A)
    assert np.allclose(out, 0.5)                      # D^-1/2 (A+I) D^-1/2 with deg=2
    assert np.allclose(symmetric_normalize(np.zeros((3, 3), np.float32)), np.eye(3))


def test_sdg_is_static_and_ehg_reacts_to_latency():
    """The reason for two graphs: the SDG stays put under load, the EHG moves."""
    sim = Simulator(DEMO6)
    m_low, m_high = sim.step(5.0), None
    sim2 = Simulator(DEMO6)
    for _ in range(3):
        m_high = sim2.step(80.0)
    S = DEMO6.services
    assert np.allclose(build_sdg(S, DEMO6.call_graph), build_sdg(S, DEMO6.call_graph))
    slo = 250.0
    ehg_low = build_ehg(S, DEMO6.call_graph, {s: m_low[s] for s in S}, slo)
    ehg_high = build_ehg(S, DEMO6.call_graph, {s: m_high[s] for s in S}, slo)
    assert not np.allclose(ehg_low, ehg_high)


def test_ehg_uses_edge_percentiles_from_traces_when_given():
    sim = Simulator(DEMO6)
    m = {s: dict(v, edge_p95={}) for s, v in sim.step(10.0).items() if s in DEMO6.services}
    S = DEMO6.services
    base = build_ehg(S, DEMO6.call_graph, m, 250.0)
    m["gateway-service"]["edge_p95"] = {"catalog-service": 5000.0}
    boosted = build_ehg(S, DEMO6.call_graph, m, 250.0)
    assert not np.allclose(base, boosted)


def test_graph_shapes_scale_with_topology_not_with_weights():
    rng = np.random.default_rng(1)
    for topo in (DEMO6, TRAINTICKET14, random_topology(rng)):
        sim = Simulator(topo)
        metrics = sim.step(10.0)
        m = {s: metrics[s] for s in topo.services}
        obs = build_observation(topo.services, topo.call_graph, topo.entry, m, metrics["__e2e__"],
                                sim.cost(), sim.budget, sim.slo_ms)
        n = len(topo.services)
        assert obs["X"].shape == (n, config.NODE_FEATURE_DIM)
        assert obs["A_sdg"].shape == obs["A_ehg"].shape == (n, n)
        assert obs["g"].shape == (config.GLOBAL_FEATURE_DIM,)
        assert np.isfinite(obs["X"]).all() and np.isfinite(obs["g"]).all()
        assert obs["X"][:, 13].sum() == 1.0                      # exactly one entry service
        assert set(np.flatnonzero(obs["X"][:, 14])) == {topo.services.index(s) for s in obs["critical_path"]}


def test_node_features_are_bounded():
    sim = Simulator(DEMO6)
    for rps in (1.0, 60.0, 500.0):
        metrics = sim.step(rps)
        m = {s: metrics[s] for s in DEMO6.services}
        total, path = critical_path_dfs(DEMO6.call_graph, DEMO6.entry,
                                        {s: m[s]["latency_ms"] for s in DEMO6.services})
        X = node_features(DEMO6.services, m, DEMO6.entry, path, total)
        assert X.min() >= -1.0 - 1e-6 and X.max() <= 5.0 + 1e-6
