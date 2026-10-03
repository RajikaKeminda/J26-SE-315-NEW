"""README generation from result files, and the pieces of the live experiment that need no cluster."""

import pytest
import yaml

from src import config
from src.evaluation import live_experiment as live
from src.evaluation import report


def fake_row(p95=200.0, viol=0.1, cost=3.5, adapt=20.0, score=0.25):
    return {"p50_latency": 100, "p95_latency": p95, "p99_latency": 300, "slo_violation_rate": viol,
            "mean_cost": cost, "adaptation_time_s": adapt, "scalarized": score, "mean_util": 0.4}


RESULTS = {
    "meta": {"episodes": 8, "patterns": ["burst"], "preference": [0.5, 0.35, 0.15], "objectives": ["slo", "cost", "balance"],
             "hpa_tuned": {"cpu": 0.5, "target": 0.5, "scalarized": 0.23}},
    "demo6": {"static": fake_row(p95=300, viol=0.2, cost=4.0, adapt=40, score=0.4), "hpa_tuned": fake_row(score=0.23), "mo_td3_full": fake_row(score=0.2, cost=3.0)},
    "demo6_seeds": {"mo_td3_full": {"seeds": [0, 1], "p95_latency": [100, 120], "slo_violation_rate": [0.0, 0.1],
                                    "mean_cost": [3.0, 3.2], "scalarized": [0.2, 0.22]}},
    "trainticket14": {"hpa_tuned": fake_row(), "mo_td3_full": fake_row(score=0.18)},
}


def test_comparison_table_bolds_the_best_per_column():
    table = report.comparison_table(RESULTS["demo6"])
    lines = table.splitlines()
    assert lines[0].startswith("| controller") and len(lines) == 2 + 3
    agent_row = next(l for l in lines if "MO-TD3 full" in l)
    assert "**0.200**" in agent_row and "**3.00**" in agent_row
    assert "**" not in next(l for l in lines if "static" in l).replace("static (no scaling)", "")


def test_results_block_has_all_sections():
    text = report.block_results(RESULTS)
    assert "tuned HPA" in text and "0.5 cores, 50% target" in text
    seeds = text.split("Across training seeds")[1].split("14-service")[0]
    assert "| 2 |" in seeds and "0.210 ± 0.010" in seeds
    assert "never seen in training" in text and "0.180" in text
    assert report.block_results(None).startswith("_(no results")


def test_pareto_and_calibration_and_live_blocks():
    pareto = {"points": [{"name": "a", "family": "mo_td3", "pareto_optimal": True, "p95_latency": 100, "slo_violation_rate": 0.0, "mean_cost": 3.0},
                         {"name": "b", "family": "hpa", "pareto_optimal": False, "p95_latency": 150, "slo_violation_rate": 0.1, "mean_cost": 4.0}],
              "families": {"mo_td3": {"hypervolume": 0.5, "n_points": 1, "n_pareto_optimal": 1},
                           "hpa": {"hypervolume": 0.4, "n_points": 1, "n_pareto_optimal": 0}}}
    text = report.block_pareto(pareto)
    assert "| a |" in text and "| b |" not in text and "0.500" in text and "0.400" in text
    cal = report._load(config.SIM_PARAMS_PATH)
    assert "measured saturation" in report.block_calibration(cal)
    liveblob = {"meta": {"steps": 40, "step_seconds": 10, "slo_ms": 250},
                "runs": [{"pattern": "burst", "controller": "static", "live": fake_row(), "sim": fake_row(p95=210)}]}
    assert "| burst | static (no scaling) | 200 | 210 |" in report.block_live(liveblob)


def test_apply_replaces_only_the_marked_blocks_and_requires_them():
    src = "intro\n<!-- BEGIN:results -->\nold\n<!-- END:results -->\noutro"
    assert report.apply(src, {"results": "NEW"}) == "intro\n<!-- BEGIN:results -->\nNEW\n<!-- END:results -->\noutro"
    with pytest.raises(SystemExit):
        report.apply("no markers here", {"results": "x"})


def test_readme_has_every_generated_block():
    text = open(report.README).read()
    for name in report.render():
        assert f"<!-- BEGIN:{name} -->" in text and f"<!-- END:{name} -->" in text


def test_readme_matches_committed_results():
    """Docs cannot drift from data: regenerate and compare."""
    text = open(report.README).read()
    if "_(generated)_" in text:
        pytest.skip("README tables not generated yet")
    assert report.apply(text, report.render()) == text, "run: python -m src.evaluation.report"


# ----------------------------------------------------------- live experiment bits
def test_live_hpa_manifest_matches_the_simulated_hpa(monkeypatch):
    captured = {}
    monkeypatch.setattr(live, "kubectl", lambda *a, **k: captured.update(args=a, input=k.get("input")))
    live.create_hpas(target_pct=60)
    docs = [d for d in yaml.safe_load_all(captured["input"]) if d]
    assert len(docs) == len(config.SERVICES) and captured["args"][:2] == ("apply", "-f")
    d = docs[0]
    assert d["spec"]["minReplicas"] == config.MIN_REPLICAS and d["spec"]["maxReplicas"] == config.MAX_REPLICAS
    assert d["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 60
    assert d["spec"]["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == \
        config.HPA_SCALE_DOWN_STABILIZATION_STEPS * config.STEP_SECONDS
    assert d["metadata"]["namespace"] == config.NAMESPACE


def test_live_initial_cpu_and_sim_pairing():
    best = {"cpu": 0.5, "target": 0.45}
    assert live.initial_cpu("hpa_tuned", best) == 0.5 and live.initial_cpu("static", best) == config.CPU_DEFAULT
    sim = live.run_sim("hpa_tuned", "burst", 1000, None, best, config.DEFAULT_PREFERENCE)
    again = live.run_sim("hpa_tuned", "burst", 1000, None, best, config.DEFAULT_PREFERENCE)
    assert sim == again and 0 <= sim["slo_violation_rate"] <= 1
    assert set(live.ALL_CONTROLLERS) >= {"static", "hpa_tuned", "mo_td3", "k8s_hpa"}


def test_live_traffic_curve_equals_the_environments_curve():
    """The live run and its simulated twin must see the same traffic."""
    import numpy as np
    from src.rl_agent.env import AutoscalerEnv, topology_base_rps
    from src.simulation.simulator import generate_traffic
    from src.topologies import DEMO6
    env = AutoscalerEnv(DEMO6, seed=1234)
    env.reset("burst")
    curve = generate_traffic("burst", config.EPISODE_LENGTH_STEPS + 1, topology_base_rps(DEMO6), np.random.default_rng(1234))
    assert list(env.traffic) == pytest.approx(list(curve))
