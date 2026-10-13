"""CRISP-style critical-path analysis: per-request blame aggregated into critical shares, and its held-out comparison with the
graph-level depth-first search on real Train-Ticket traces."""

import json
import os

import pytest

from src.topology_intel import crisp
from tests.test_trainticket import _span, _trace, booking

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "trainticket_traces.json")


def test_shares_are_per_request_fractions_of_downstream_time():
    reqs = crisp.requests_from_traces([booking()] * 4, "gw")
    assert len(reqs) == 4 and "gw" not in reqs[0].blame
    sh = crisp.shares(reqs)
    assert sum(sh.values()) == pytest.approx(1.0)
    assert sh["seat"] > 0.2 and "basic" not in sh        # of two parallel calls only the one that finished last is on the critical path


def test_requests_that_never_leave_the_entry_service_carry_no_signal():
    only_gw = _trace([_span("g", "gw", "server", 0, 50)])
    assert crisp.requests_from_traces([only_gw, booking()], "gw")[0].downstream_ms > 0
    assert len(crisp.requests_from_traces([only_gw], "gw")) == 0


def test_profile_views_and_critical_set():
    fast = _trace([_span("g", "gw", "server", 0, 100, op="POST fast"), _span("gc", "gw", "client", 1, 90, parent="g"),
                   _span("a", "svc-a", "server", 2, 88, parent="gc")])
    slow = _trace([_span("g", "gw", "server", 0, 900, op="POST slow"), _span("gc", "gw", "client", 1, 890, parent="g"),
                   _span("b", "svc-b", "server", 2, 888, parent="gc")])
    reqs = crisp.requests_from_traces([fast] * 8 + [slow] * 2, "gw")
    prof = crisp.CriticalProfile(reqs, tail_quantile=0.8, min_type_requests=2)
    assert prof.overall["svc-a"] > prof.overall["svc-b"]                  # most requests wait for a ...
    assert next(iter(prof.tail)) == "svc-b"                               # ... but the slow tail waits for b
    assert prof.top(1, op="POST slow") == ["svc-b"] and prof.top(1, op="POST fast") == ["svc-a"]
    assert prof.critical_set(coverage=0.5) == ["svc-a"] and set(prof.critical_set(coverage=0.99)) == {"svc-a", "svc-b"}


def test_critical_set_beats_the_depth_first_chain_on_held_out_real_requests():
    from src.evaluation import critical_path_eval as E
    res = E.cross_validate(json.load(open(FIXTURE)), folds=5, repeats=3)
    cur, per_type = res["current: DFS on mean own latency"], res["top-k critical share per request type"]
    rnd = res["random chain (expected)"]
    assert per_type["coverage"][0] > cur["coverage"][0] + 0.05           # held-out: learned on training traces only
    assert per_type["top1"][0] > cur["top1"][0] and per_type["tail_coverage"][0] > cur["tail_coverage"][0]
    assert cur["coverage"][0] > rnd["coverage"][0] + 0.2
    # changing only the WEIGHTS of the chain search does not help: the gain comes from the set / per-type representation
    assert abs(res["DFS weighted by critical share"]["coverage"][0] - cur["coverage"][0]) < 0.03
