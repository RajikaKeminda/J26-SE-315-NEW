"""The conventional autoscalers must behave like the real ones, or the comparison means nothing."""

import math

import pytest

from src import config
from src.baselines.rule_based import HPABaseline, StaticBaseline, ThresholdBaseline, VPABaseline


def m(util=0.5, replicas=2, cpu=0.5, mem=128.0, mem_util=0.4):
    return {"utilization": util, "replicas": replicas, "cpu": cpu, "mem": mem, "mem_utilization": mem_util}


def test_hpa_formula_desired_equals_ceil_current_times_ratio():
    h = HPABaseline(["a"], target_utilization=0.5)
    out = h.decide({"a": m(util=0.9, replicas=3)})["a"]
    assert out["dr"] == math.ceil(3 * 0.9 / 0.5) - 3 == 3


def test_hpa_tolerance_band_means_no_change():
    h = HPABaseline(["a"], target_utilization=0.5)
    assert h.decide({"a": m(util=0.54)})["a"]["dr"] == 0                 # within +-10 %
    assert h.decide({"a": m(util=0.46)})["a"]["dr"] == 0


def test_hpa_scale_up_is_capped_like_the_default_policy():
    h = HPABaseline(["a"], target_utilization=0.1)
    out = h.decide({"a": m(util=1.0, replicas=2)})["a"]
    assert out["dr"] == max(config.HPA_SCALE_UP_MAX_PODS, 2)             # max(4 pods, 100 %)
    big = h.decide({"a": m(util=1.0, replicas=6)})["a"]["dr"]
    assert big == config.MAX_REPLICAS - 6, "bounded by MAX_REPLICAS"


def test_hpa_scale_down_uses_the_stabilisation_window():
    h = HPABaseline(["a"], target_utilization=0.6, stabilization_steps=4)
    h.decide({"a": m(util=1.0, replicas=2)})                              # recommends 4: remembered
    for _ in range(3):
        assert h.decide({"a": m(util=0.05, replicas=4)})["a"]["dr"] == 0  # window still holds the high rec
    out = h.decide({"a": m(util=0.05, replicas=4)})["a"]["dr"]
    assert out < 0, "after the window passes, scale down"


def test_hpa_never_leaves_bounds_and_never_touches_resources():
    h = HPABaseline(["a"])
    for util in (0.0, 0.3, 1.0):
        for r in (config.MIN_REPLICAS, config.MAX_REPLICAS):
            o = h.decide({"a": m(util=util, replicas=r)})["a"]
            assert config.MIN_REPLICAS <= r + o["dr"] <= config.MAX_REPLICAS or o["dr"] > 0
            assert o["cpu"] == 0.5 and o["mem"] == 128.0


def test_vpa_recommends_p90_usage_plus_margin_and_never_changes_replicas():
    v = VPABaseline(["a"])
    out = None
    for _ in range(6):
        out = v.decide({"a": m(util=0.4, cpu=1.0)})["a"]                 # usage = 0.4 cores of a 1.0 limit
    assert out["dr"] == 0
    assert out["cpu"] == pytest.approx(0.4 * (1 + config.VPA_SAFETY_MARGIN))


def test_vpa_ignores_small_differences():
    v = VPABaseline(["a"])
    for _ in range(6):
        out = v.decide({"a": m(util=0.87, cpu=0.5)})["a"]                # rec ~0.50: inside 10 %
    assert out["cpu"] == 0.5


def test_vpa_cannot_exceed_the_limit_it_is_throttled_at():
    """Usage is capped by the CPU limit, so a saturated pod only ever grows by one margin per cycle."""
    v = VPABaseline(["a"])
    for _ in range(6):
        out = v.decide({"a": m(util=1.0, cpu=0.25)})["a"]
    assert out["cpu"] == pytest.approx(0.25 * (1 + config.VPA_SAFETY_MARGIN))


def test_threshold_rule_and_cooldown():
    t = ThresholdBaseline(["a"], cooldown_steps=3)
    assert t.decide({"a": m(util=0.9)})["a"]["dr"] == 1                   # > 80 %: up, immediately
    assert t.decide({"a": m(util=0.1)})["a"]["dr"] == 0                   # < 30 % but inside the cooldown
    assert t.decide({"a": m(util=0.1)})["a"]["dr"] == 0
    assert t.decide({"a": m(util=0.1)})["a"]["dr"] == -1                  # cooldown over: down
    assert t.decide({"a": m(util=0.55)})["a"]["dr"] == 0                  # inside the band


def test_static_does_nothing_and_all_expose_reset():
    s = StaticBaseline(["a"])
    assert s.decide({"a": m()})["a"] == {"dr": 0, "cpu": 0.5, "mem": 128.0}
    for b in (s, HPABaseline(["a"]), VPABaseline(["a"]), ThresholdBaseline(["a"])):
        b.reset()
