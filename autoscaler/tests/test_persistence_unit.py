"""Persistence without a database: reason categories, the row builder, the writer thread (with the database calls replaced), the engine hook."""

import json
import os
import time
from datetime import datetime, timezone

import pytest

from src import config, persistence
from src.persistence import db, identity, rows, writer as writer_mod
from src.persistence.categorize import categorize
from src.persistence.sink import NullSink, PostgresSink
from tests.test_engine_and_api import FakeJaeger, FakeK8s, FakeProm


# ------------------------------------------------------------------ reason codes
@pytest.mark.parametrize("text,codes", [
    ("clamped replicas to [1,2]", ["clamp"]),
    ("memory floor 256 MiB (working set 200)", ["memory_floor"]),
    ("scale-down cooldown (1 of 3)", ["cooldown"]),
    ("scale-down blocked after a scale-up (2 of 10 steps)", ["stabilisation"]),
    ("scale-down blocked: newest pod is only 40 s old (window 300 s)", ["young_pod"]),
    ("resize cooldown (1 of 6)", ["resize_cooldown"]),
    ("blocked: would exceed cluster budget (14.0 cores-eq)", ["budget"]),
    ("scale-up blocked: node memory headroom (100 MiB left, need 400)", ["node_memory"]),
    ("scale-up blocked: node memory unknown (boom)", ["node_memory"]),
    ("resize disabled (ALLOW_RESIZE=0)", ["resize_disabled"]),
    ("observed only (not in CONTROLLED_SERVICES)", ["observed_only"]),
    ("k8s error: api down", ["k8s_error"]),
    ("HOLD: cluster read failed", ["hold"]),
    ("scale-down cooldown (1 of 3); blocked: would exceed cluster budget (9.0 cores-eq)", ["cooldown", "budget"]),
    ("something new", ["other"]),
    (None, []), ("", []),
])
def test_categorize(text, codes):
    assert categorize(text) == codes


def test_every_reason_the_code_base_can_produce_is_recognised():
    """The messages are literals in guards.py / engines.py: a new one without a rule would silently become 'other'."""
    import re
    src = open("src/safety/guards.py").read() + open("src/inference/engines.py").read()
    texts = re.findall(r'(?:notes\.append|reasons\[[^\]]+\]\s*=|reasons\.setdefault\([^,]+,)\s*\(?f?"([^"{]+)', src)
    assert texts, "found no reason literals - the pattern needs updating"
    for t in texts:
        assert categorize(t) != ["other"], f"no category for {t!r}"


# ------------------------------------------------------------------ the row builder
def _payload(step=0, ts=1_700_000_000.0, **kw):
    node = {"id": "svc-a", "replicas": 2, "ready": 2, "pending": 0, "cpu": 0.5, "mem": 256, "latency_ms": 20.0, "latency_p95_ms": 40.0,
            "requests": 10.0, "utilization": 0.8, "mem_utilization": 0.4, "on_critical_path": True, "is_entry": False, "controlled": True,
            "proposed_delta": 0, "applied_delta": 0, "proposed_cpu": 0.5, "applied_cpu": 0.5, "proposed_mem": 256, "applied_mem": 256,
            "blocked_reason": None}
    entry = dict(node, id="gw", is_entry=True)
    p = {"step": step, "timestamp": ts, "mode": "live", "external_rps": 10.0, "slo_ms": 500.0,
         "e2e": {"mean_ms": 100.0, "p50_ms": 90.0, "p95_ms": 200.0, "p99_ms": 300.0, "error_rate": 0.0}, "slo_breached": False,
         "critical_path": ["gw", "svc-a"], "critical_path_latency_ms": 60.0, "breaker_open": False, "total_replicas": 4, "cost": 2.0,
         "budget": 8.0, "nodes": [entry, node], "edges": [{"source": "gw", "target": "svc-a", "traffic_share": 1.0}],
         "stale": False, "shadow": False, "controller": "hpa", "policy": "none",
         "critical_path_evidence": {"slowest_trace": "abc", "share": {"svc-a": 1.0}, "coverage": 0.9}}
    p.update(kw)
    return p


def _node(p, svc="svc-a"):
    return next(n for n in p["nodes"] if n["id"] == svc)


def test_every_step_stores_a_step_row_and_a_row_per_service():
    r = rows.StepRecorder().rows(_payload())
    assert len(r["steps"]) == 1 and len(r["service_metrics"]) == 2
    assert len(r["steps"][0]) == len(rows.COLUMNS["steps"])
    assert all(len(x) == len(rows.COLUMNS["service_metrics"]) for x in r["service_metrics"])
    assert "actions" not in r, "no proposal and nothing blocked: no action row"
    assert r["steps"][0][1] == datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)


def test_a_scale_up_is_an_applied_action_with_before_and_after():
    p = _payload()
    n = _node(p)
    n.update(proposed_delta=1, applied_delta=1)
    r = rows.StepRecorder().rows(p, {"guard_approved": {"svc-a": 1}})
    a = dict(zip(rows.COLUMNS["actions"], r["actions"][0]))
    assert a["outcome"] == "applied" and a["replicas_before"] == 2 and a["replicas_after"] == 3 and a["guard_dr"] == 1
    assert a["shadow"] is False and a["reason_codes"] == []


def test_shadow_decisions_are_stored_but_do_not_change_replicas():
    p = _payload(shadow=True)
    _node(p).update(proposed_delta=1, applied_delta=1)
    a = dict(zip(rows.COLUMNS["actions"], rows.StepRecorder().rows(p)["actions"][0]))
    assert a["shadow"] is True and a["outcome"] == "applied" and a["replicas_after"] == a["replicas_before"] == 2


def test_a_vetoed_action_keeps_the_reason_and_its_codes():
    p = _payload()
    _node(p).update(proposed_delta=1, applied_delta=0, blocked_reason="scale-up blocked: node memory headroom (10 MiB left, need 400)")
    a = dict(zip(rows.COLUMNS["actions"], rows.StepRecorder().rows(p, {"guard_approved": {"svc-a": 1}})["actions"][0]))
    assert a["outcome"] == "vetoed" and a["reason_codes"] == ["node_memory"] and a["guard_dr"] == 1 and a["applied_dr"] == 0


def test_a_partly_applied_action_is_modified_and_a_cluster_error_is_an_event():
    p = _payload()
    _node(p).update(proposed_delta=2, applied_delta=1, blocked_reason="clamped replicas to [1,3]")
    assert dict(zip(rows.COLUMNS["actions"], rows.StepRecorder().rows(p)["actions"][0]))["outcome"] == "modified"
    p = _payload()
    _node(p).update(proposed_delta=1, applied_delta=0, blocked_reason="k8s error: api down")
    r = rows.StepRecorder().rows(p)
    assert r["actions"][0][4] == "k8s_error" and any(e[2] == "k8s_error" for e in r["events"])


def test_untouched_uncontrolled_services_make_no_action_rows():
    p = _payload()
    _node(p).update(controlled=False, blocked_reason="observed only (not in CONTROLLED_SERVICES)")
    assert "actions" not in rows.StepRecorder().rows(p)
    _node(p).update(proposed_delta=1)                         # the agent wanted to act on a service it may not touch
    assert rows.StepRecorder().rows(p)["actions"][0][4] == "vetoed"


def test_critical_path_is_stored_on_change_and_every_nth_step():
    rec = rows.StepRecorder(snapshot_every=5)
    stored = [bool(rec.rows(_payload(step=i)).get("critical_paths")) for i in range(1, 12)]
    assert stored[0] is True and stored[1:4] == [False] * 3 and stored[4] is True       # first, then step 5, step 10
    changed = rec.rows(_payload(step=12, critical_path=["gw", "svc-b"]))["critical_paths"][0]
    assert changed[5] is True and changed[3] == ["gw", "svc-b"]


def test_topology_is_stored_only_when_the_edges_change():
    rec = rows.StepRecorder()
    assert rec.rows(_payload(step=0))["topology_snapshots"]
    assert "topology_snapshots" not in rec.rows(_payload(step=1))
    assert rec.rows(_payload(step=2, edges=[{"source": "gw", "target": "svc-b", "traffic_share": 1.0}]))["topology_snapshots"]


def test_violation_episode_opens_updates_and_closes_after_two_clean_steps():
    rec = rows.StepRecorder()
    bad = lambda i, p95: _payload(step=i, slo_breached=True, e2e={"p95_ms": p95, "mean_ms": 0, "p50_ms": 0, "p99_ms": 0, "error_rate": 0})
    ok = lambda i: _payload(step=i)
    assert "violations" not in rec.rows(ok(0))
    first = rec.rows(bad(1, 700.0))["violations"][0]
    assert first[0] == 1 and first[2] is None and first[4] == 1                     # start step 1, open, one breached step
    second = rec.rows(bad(2, 900.0))["violations"][0]
    assert second[0] == 1 and second[4] == 2 and second[5] == 900.0 and second[7][0]["service"] == "svc-a"
    assert "violations" not in rec.rows(ok(3)), "one clean step does not close it"
    closed = rec.rows(ok(4))["violations"][0]
    assert closed[3] == 2 and closed[2] is not None                                 # ended at the last breached step
    assert rec.rows(bad(5, 800.0))["violations"][0][0] == 5, "a new breach is a new episode"


def test_breaker_and_hold_transitions_are_events_once():
    rec = rows.StepRecorder()
    kinds = lambda p: [e[2] for e in rec.rows(p).get("events", [])]
    assert kinds(_payload(step=0)) == []
    assert kinds(_payload(step=1, breaker_open=True)) == ["breaker_open"]
    assert kinds(_payload(step=2, breaker_open=True)) == []
    assert kinds(_payload(step=3)) == ["breaker_closed"]
    assert kinds(_payload(step=4, stale=True, stale_reason="cluster read failed")) == ["hold"]
    assert kinds(_payload(step=5, stale=True, stale_reason="cluster read failed")) == []
    assert kinds(_payload(step=6)) == ["recovered"]


def test_a_hold_step_stores_the_step_but_no_service_data():
    r = rows.StepRecorder().rows(_payload(stale=True, stale_reason="x", slo_breached=True))
    assert len(r["steps"]) == 1 and r["steps"][0][9] is False       # stale data never counts as an SLO breach
    assert "service_metrics" not in r and "actions" not in r and "violations" not in r


def test_frames_only_when_asked_for():
    assert "frames" not in rows.StepRecorder().rows(_payload())
    assert len(rows.StepRecorder(store_frames=True).rows(_payload())["frames"]) == 1


# ------------------------------------------------------------------ the writer thread (database calls replaced)
class FakeConn:
    def close(self):
        pass


@pytest.fixture
def fake_db(monkeypatch):
    log = {"batches": [], "fail": 0, "registered": [], "ended": False, "conflict": None}

    def insert(conn, cluster_id, run_id, batch):
        if log["fail"] > 0:
            log["fail"] -= 1
            raise RuntimeError("connection lost")
        log["batches"].append((cluster_id, run_id, batch))

    def register(conn, name, namespace, uid=None, api=None, entry=None):
        if log["conflict"]:
            raise db.IdentityConflict(log["conflict"])
        log["registered"].append((name, namespace, uid))
        return "cluster-1"

    monkeypatch.setattr(db, "migrate", lambda c: None)
    monkeypatch.setattr(db, "register_cluster", register)
    monkeypatch.setattr(db, "start_run", lambda *a, **k: "run-1")
    monkeypatch.setattr(db, "end_run", lambda c, r: log.update(ended=True))
    monkeypatch.setattr(db, "insert_rows", insert)
    monkeypatch.setattr(db, "purge", lambda *a: {"steps": 0})
    real_sleep = time.sleep
    monkeypatch.setattr(writer_mod.time, "sleep", lambda s: real_sleep(0.001))       # no real back-off in tests
    OPEN_WRITERS.clear()
    yield log
    log["fail"], log["conflict"] = 0, None                  # a writer left running would call the next test's patched functions
    for w in OPEN_WRITERS:
        w.close(timeout=2.0)


OPEN_WRITERS = []


def _writer(**kw):
    w = writer_mod.Writer("x", lambda: {"name": "c1", "namespace": "ns"},
                          {"mode": "live", "shadow": True}, (24, 90), connect_fn=FakeConn, **kw)
    OPEN_WRITERS.append(w)
    return w


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.01)
    return cond()


def test_writer_registers_the_cluster_then_writes_batches_with_its_id(fake_db):
    w = _writer()
    assert w.wait_ready()
    w.put({"steps": [(1,)]})
    w.close()
    assert fake_db["registered"] == [("c1", "ns", None)] and fake_db["ended"]
    assert all(c == "cluster-1" and r == "run-1" for c, r, _ in fake_db["batches"])
    assert any("steps" in b for _, _, b in fake_db["batches"])
    assert w.status()["written_steps"] == 1


def test_writer_keeps_the_batch_and_retries_when_the_database_fails(fake_db):
    w = _writer()
    assert w.wait_ready()
    fake_db["fail"] = 3
    w.put({"steps": [(7,)]})
    assert _wait(lambda: any(b.get("steps") == [(7,)] for _, _, b in fake_db["batches"])), w.status()
    w.close()
    assert w.status()["state"] == "stopped"


def test_put_never_blocks_and_counts_drops_when_the_queue_is_full(fake_db):
    fake_db["fail"] = 10**6                                            # database permanently down: the thread never drains
    w = _writer(maxsize=5)
    t0 = time.time()
    for i in range(200):
        w.put({"steps": [(i,)]})
    assert time.time() - t0 < 1.0, "put() must not wait for the database"
    assert w.status()["dropped"] > 0 and w.status()["queued"] <= 5


def test_close_does_not_hang_when_the_database_is_down_and_the_queue_is_full(fake_db):
    fake_db["fail"] = 10**6
    w = _writer(maxsize=5)
    for i in range(50):
        w.put({"steps": [(i,)]})
    t0 = time.time()
    w.close(timeout=3.0)
    assert time.time() - t0 < 3.5 and w.status()["state"] == "stopped"


def test_an_identity_conflict_disables_the_writer_and_drops_everything_quietly(fake_db):
    fake_db["conflict"] = "name already used by another cluster"
    w = _writer()
    assert _wait(lambda: w.state == "disabled")
    assert "another cluster" in w.status()["last_error"]
    assert w.put({"steps": [(1,)]}) is False
    assert fake_db["batches"] == []


# ------------------------------------------------------------------ identity
class IdentityK8s:
    def __init__(self, uid):
        self.uid = uid

    def cluster_identity(self):
        return {"api_server": "https://127.0.0.1:1", "kube_uid": self.uid}


def test_identity_uses_the_configured_name_and_the_uid_when_readable(monkeypatch):
    monkeypatch.setattr(config, "CLUSTER_NAME", "tt-pc")
    r = identity.resolve(IdentityK8s("uid-123"), "gw")
    assert (r["name"], r["kube_uid"], r["entry"]) == ("tt-pc", "uid-123", "gw")


def test_identity_falls_back_to_the_uid_and_never_to_the_api_url(monkeypatch):
    monkeypatch.setattr(config, "CLUSTER_NAME", "")
    assert identity.resolve(IdentityK8s("abcdef123456"))["name"] == "cluster-abcdef12"
    with pytest.raises(db.IdentityError, match="AUTOSCALER_CLUSTER_NAME"):
        identity.resolve(IdentityK8s(None))


# ------------------------------------------------------------------ factory and engine hook
def test_no_database_url_means_a_null_sink_and_unchanged_behaviour(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "")
    s = persistence.make_sink("live", False, "hpa", "none", None)
    assert isinstance(s, NullSink) and s.status()["enabled"] is False


def test_only_the_configured_modes_are_stored(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://x")
    for mode in ("demo", "replay"):
        s = persistence.make_sink(mode, False, "hpa", "none", None)
        assert isinstance(s, NullSink) and mode in s.status()["reason"]


class RecordingSink:
    enabled = True

    def __init__(self):
        self.calls = []

    def record_step(self, payload, extras=None):
        self.calls.append((payload, extras))

    def status(self):
        return {"enabled": True}

    def close(self):
        pass


@pytest.fixture
def live_engine():
    from src.inference.engines import LiveEngine
    e = LiveEngine(None, None, controller="hpa")
    e._prom, e._jaeger, e._k8s = FakeProm(e.services), FakeJaeger(), FakeK8s(e.services)
    return e


def test_the_engine_records_every_step_including_holds_with_the_guard_decision(live_engine):
    live_engine.sink = RecordingSink()
    live_engine._prom.set(rps=60.0, own=400.0, cpu=0.5)
    ok = live_engine.tick()
    live_engine._prom.fail = True
    held = live_engine.tick()
    (p1, x1), (p2, x2) = live_engine.sink.calls
    assert p1 is ok and p2 is held and held["stale"]
    assert "guard_approved" in x1 and all(isinstance(v, int) for v in x1["guard_approved"].values())
    assert x2 == {}


def test_a_failing_sink_cannot_stop_the_control_step(live_engine):
    class Broken(PostgresSink):
        def __init__(self):
            self.errors = 0

            class W:
                last_error = None

                def put(self, b):
                    raise RuntimeError("queue exploded")
            self.writer, self.recorder = W(), rows.StepRecorder()
    live_engine.sink = Broken()
    p = live_engine.tick()
    assert not p["stale"] and live_engine.sink.errors == 1
    assert "queue exploded" in live_engine.sink.writer.last_error


def test_a_real_live_payload_produces_rows_for_every_table(live_engine):
    live_engine._prom.set(rps=60.0, own=400.0, cpu=0.5)
    p = live_engine.tick()
    r = rows.StepRecorder().rows(p, live_engine._extras)
    assert {"steps", "service_metrics", "actions", "critical_paths", "topology_snapshots"} <= set(r)
    assert len(r["service_metrics"]) == len(live_engine.services)
    assert any(a[4] == "applied" for a in r["actions"])


def test_the_state_endpoint_reports_the_persistence_status(live_engine):
    assert live_engine.persistence_status()["enabled"] is False
    live_engine.sink = RecordingSink()
    assert live_engine.persistence_status() == {"enabled": True}


def test_every_query_over_history_is_scoped_by_cluster_id():
    """A new query that forgets the cluster filter would show one cluster's history inside another's view."""
    import re
    src = open("src/persistence/queries.py").read()
    sql = re.findall(r'"((?:SELECT|INSERT|UPDATE|DELETE)[^"]*(?:"\s*\n\s*"[^"]*)*)"', src)
    texts = [m for m in re.findall(r'_rows\(\s*conn,\s*((?:"[^"]*"\s*)+)', src)]
    assert len(texts) >= 12, "the pattern found too few queries - update it"
    unscoped = [t for t in texts if "cluster_id" not in t and "FROM clusters" not in t]
    assert not unscoped, f"queries without a cluster_id filter: {unscoped}"
    assert sql  # the pattern above is a sanity net only


def test_the_replay_engine_accepts_frames_loaded_from_the_database():
    from src.inference.engines import ReplayEngine
    e = ReplayEngine(None, frames=[_payload(step=0), _payload(step=1)])
    assert e.tick()["replay_length"] == 2
    with pytest.raises(ValueError, match="no recorded payloads"):
        ReplayEngine(None, frames=[])


def test_memory_can_be_locked_while_cpu_still_changes(monkeypatch):
    """ALLOW_MEMORY_RESIZE=0: the agent's CPU and replica outputs act, its memory output is ignored (in training and live alike)."""
    import numpy as np
    from src.rl_agent.env import decode_action
    metrics = {"a": {"cpu": 0.3, "mem": 800.0}}
    idx, cont = np.array([4]), np.array([[0.9, 0.9]], dtype=np.float32)
    free = decode_action(["a"], metrics, idx, cont)["a"]
    assert free["cpu"] > 0.3 and free["mem"] != 800.0
    monkeypatch.setattr(config, "ALLOW_MEMORY_RESIZE", False)
    locked = decode_action(["a"], metrics, idx, cont)["a"]
    assert locked["cpu"] == free["cpu"] and locked["mem"] == 800.0 and locked["dr"] == free["dr"] == 1


def test_the_stored_regime_holds_no_absolute_path_of_this_machine(monkeypatch):
    from src.persistence import training
    monkeypatch.setattr(config, "SIM_PARAMS_PATH", os.path.join(config.BASE_DIR, "src", "simulation", "sim_params_trainticket.json"))
    r = training.regime()
    assert r["SIM_PARAMS_PATH"] == os.path.join("src", "simulation", "sim_params_trainticket.json")
    assert "/Users/" not in json.dumps(r) and r["STEP_SECONDS"] == config.STEP_SECONDS
