"""Persistence against a real PostgreSQL. Skipped unless TEST_DATABASE_URL is set, e.g.

    bash scripts/22_db_up.sh
    TEST_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler pytest tests/test_persistence_pg.py

Each test works in its own schema (search_path), dropped afterwards, so nothing else in that database is touched."""

import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from src.persistence import db, queries, rows                      # noqa: E402
from src.persistence.sink import PostgresSink                      # noqa: E402
from src.persistence.writer import Writer                          # noqa: E402
from tests.test_engine_and_api import FakeJaeger, FakeK8s, FakeProm   # noqa: E402
from tests.test_persistence_unit import _payload, _node            # noqa: E402

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def connect():
    schema = "t_" + uuid.uuid4().hex[:10]
    admin = psycopg.connect(URL, autocommit=True)
    admin.execute(f"CREATE SCHEMA {schema}")

    def make(autocommit=False):
        return psycopg.connect(URL, options=f"-c search_path={schema}", connect_timeout=5, autocommit=autocommit)
    yield make
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def make_writer(connect, name="cluster-a", uid=None, ns="default"):
    return Writer(URL, lambda: {"name": name, "namespace": ns, "kube_uid": uid, "api_server": "https://x", "entry": "gw"},
                  {"mode": "live", "shadow": False, "controller": "hpa", "policy": "none", "slo_ms": 500.0, "settings": {"A": 1}},
                  (24, 90), connect_fn=connect)


def reader(connect):
    return connect(autocommit=True)


@pytest.fixture
def live_engine():
    from src.inference.engines import LiveEngine
    e = LiveEngine(None, None, controller="hpa")
    e._prom, e._jaeger, e._k8s = FakeProm(e.services), FakeJaeger(), FakeK8s(e.services)
    return e


def only_cluster(conn):
    cs = queries.clusters(conn)
    assert len(cs) == 1
    return cs[0]["id"]


# ----------------------------------------------------------------------------------------------
def test_schema_migrates_twice_and_every_fact_table_has_cluster_and_run(connect):
    conn = connect()
    db.migrate(conn)
    db.migrate(conn)
    conn.commit()
    tables = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")}
    assert {"clusters", "runs", "steps", "service_metrics", "actions", "critical_paths", "topology_snapshots", "violations", "events",
            "frames", "schema_version"} <= tables
    for t in tables - {"clusters", "schema_version"}:
        cols = {r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
                                           "AND table_name = %s", (t,))}
        assert "cluster_id" in cols, f"{t} has no cluster_id"
        if t != "runs":
            assert "run_id" in cols, f"{t} has no run_id"
    assert [r[0] for r in conn.execute("SELECT version FROM schema_version")] == [1]


def test_live_ticks_are_stored_in_every_table_and_can_be_read_back(connect, live_engine):
    w = make_writer(connect)
    live_engine.sink = PostgresSink(w, rows.StepRecorder("dfs", 1, store_frames=True))
    assert w.wait_ready()
    live_engine._prom.set(rps=60.0, own=400.0, cpu=0.5)
    n = 6
    for _ in range(n):
        live_engine.tick()
    live_engine.close()
    assert w.status()["state"] == "stopped" and w.status()["last_error"] is None

    c = reader(connect)
    cid = only_cluster(c)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    assert len(queries.steps(c, cid, since)) == n
    assert len(queries.metrics(c, cid, since)) == n * len(live_engine.services)
    acts = queries.actions(c, cid, since)
    assert any(a["outcome"] in ("applied", "modified") and a["replicas_after"] is not None for a in acts)
    assert all(a["reason_codes"] for a in acts if a["outcome"] == "vetoed"), "a veto always says why"
    assert queries.critical_paths(c, cid, since)[0]["chain"][0] == live_engine.entry
    assert queries.topology(c, cid, since)
    ev = {e["type"] for e in queries.events(c, cid, since)}
    assert {"agent_start", "agent_stop"} <= ev
    run = queries.runs(c, cid)[0]
    assert run["shadow"] is False and run["settings"] == {"A": 1} and run["ended_at"] is not None
    frames = queries.frames(c, cid, since)
    assert len(frames) == n and frames[0]["nodes"] and frames[0]["step"] == 0
    s = queries.summary(c, cid, since)
    assert s["steps"] == n and sum(s["actions_by_outcome"].values()) == len(acts)


def test_two_clusters_in_one_database_never_see_each_other(connect):
    conn = reader(connect)
    ids = {}
    for name in ("cluster-a", "cluster-b"):
        w = make_writer(connect, name)
        rec = rows.StepRecorder("dfs", 1)
        sink = PostgresSink(w, rec)
        assert w.wait_ready()
        for i in range(3):
            p = _payload(step=i, ts=time.time(), slo_breached=True,
                         e2e={"p95_ms": 900.0, "mean_ms": 1, "p50_ms": 1, "p99_ms": 1, "error_rate": 0})
            _node(p).update(id=f"{name}-svc", proposed_delta=1, applied_delta=1)
            p["nodes"][1]["id"] = f"{name}-svc"
            p["edges"] = [{"source": "gw", "target": f"{name}-svc", "traffic_share": 1.0}]
            sink.record_step(p)
        sink.close()
        ids[name] = queries.resolve_cluster(conn, name)
    assert len({*ids.values()}) == 2 and None not in ids.values()
    for mine, other in (("cluster-a", "cluster-b"), ("cluster-b", "cluster-a")):
        cid = ids[mine]
        everything = {
            "steps": queries.steps(conn, cid), "metrics": queries.metrics(conn, cid), "actions": queries.actions(conn, cid),
            "violations": queries.violations(conn, cid), "critical": queries.critical_paths(conn, cid),
            "topology": queries.topology(conn, cid), "events": queries.events(conn, cid)}
        assert all(rows_ for rows_ in everything.values()), {k: len(v) for k, v in everything.items()}
        text = repr(everything)
        assert f"{other}-svc" not in text, "another cluster's data leaked into the result"
        assert f"{mine}-svc" in text
        assert queries.summary(conn, cid)["steps"] == 3
    # the database itself refuses a row whose run belongs to another cluster
    run_a = queries.runs(conn, ids["cluster-a"])[0]["id"]
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute("INSERT INTO steps (cluster_id, run_id, step, ts, slo_breached, breaker_open, stale, shadow) "
                     "VALUES (%s, %s, 99, now(), false, false, false, false)", (ids["cluster-b"], run_a))


def test_same_name_other_cluster_is_refused_and_same_cluster_other_name_is_recognised(connect):
    c = connect()
    db.migrate(c)
    first = db.register_cluster(c, "tt", "default", kube_uid="uid-aaaaaaaa")
    assert db.register_cluster(c, "tt", "default", kube_uid="uid-aaaaaaaa") == first
    with pytest.raises(db.IdentityConflict, match="another cluster"):
        db.register_cluster(c, "tt", "default", kube_uid="uid-bbbbbbbb")
    assert db.register_cluster(c, "renamed", "default", kube_uid="uid-aaaaaaaa") == first
    assert db.register_cluster(c, "tt", "other-namespace", kube_uid="uid-aaaaaaaa") != first, "another namespace is another target"
    with pytest.raises(db.IdentityConflict, match="cannot be read now"):
        db.register_cluster(c, "tt", "default")                      # the name alone cannot prove it is the same cluster any more
    other = db.register_cluster(c, "plain", "default")               # a cluster that was never fingerprinted is matched by name
    assert db.register_cluster(c, "plain", "default") == other


def test_a_conflicting_cluster_is_disabled_and_nothing_is_written(connect):
    c = connect()
    db.migrate(c)
    db.register_cluster(c, "tt", "default", kube_uid="uid-aaaaaaaa")
    c.commit()
    w = make_writer(connect, "tt", uid="uid-bbbbbbbb")
    end = time.time() + 5
    while w.state == "starting" and time.time() < end:
        time.sleep(0.02)
    assert w.state == "disabled" and "another cluster" in w.last_error
    assert w.put({"steps": [(0,)]}) is False
    assert reader(connect).execute("SELECT count(*) FROM steps").fetchone()[0] == 0


def test_retention_deletes_metrics_after_a_day_but_keeps_actions_and_violations(connect):
    w = make_writer(connect)
    sink = PostgresSink(w, rows.StepRecorder("dfs", 1, store_frames=True))
    assert w.wait_ready()
    old = time.time() - 3 * 86400
    for i, ts in enumerate((old, time.time())):
        p = _payload(step=i, ts=ts, slo_breached=True, e2e={"p95_ms": 900.0, "mean_ms": 1, "p50_ms": 1, "p99_ms": 1, "error_rate": 0})
        _node(p).update(proposed_delta=1, applied_delta=1)
        sink.record_step(p)
    sink.close()
    c = connect(autocommit=True)
    counts = lambda: {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                      for t in ("service_metrics", "frames", "steps", "actions", "violations", "critical_paths", "events")}
    before = counts()
    deleted = db.purge(c, 24, 90)
    after = counts()
    assert deleted["service_metrics"] == 2 and after["service_metrics"] == before["service_metrics"] - 2      # 2 services, the 3-day-old step
    assert after["frames"] == before["frames"] - 1
    for t in ("steps", "actions", "violations", "critical_paths", "events"):
        assert after[t] == before[t], f"{t} is kept for 90 days"
    deleted = db.purge(c, 24, 1)
    assert counts()["steps"] == 1 and counts()["actions"] == 1, "history older than its own retention goes too"


def test_writer_recovers_when_the_database_connection_is_killed(connect):
    pids = []

    def tracked():
        conn = connect()
        pids.append(conn.info.backend_pid)
        return conn

    w = make_writer(tracked)
    sink = PostgresSink(w, rows.StepRecorder("dfs", 1))
    assert w.wait_ready()
    sink.record_step(_payload(step=0, ts=time.time()))
    end = time.time() + 5
    while w.written < 1 and time.time() < end:
        time.sleep(0.02)
    assert w.written == 1
    killer = psycopg.connect(URL, autocommit=True)
    killer.execute("SELECT pg_terminate_backend(%s)", (pids[-1],))
    killer.close()
    sink.record_step(_payload(step=1, ts=time.time()))
    sink.record_step(_payload(step=2, ts=time.time()))
    end = time.time() + 20
    while w.written < 3 and time.time() < end:
        time.sleep(0.1)
    assert w.written == 3, w.status()
    assert len(pids) >= 2, "it reconnected"
    sink.close()
    c = reader(connect)
    assert c.execute("SELECT count(*) FROM runs").fetchone()[0] == 1, "an outage does not start a new run"
    assert c.execute("SELECT count(*) FROM events WHERE type = 'agent_start'").fetchone()[0] == 1
    assert reader(connect).execute("SELECT count(*) FROM steps").fetchone()[0] == 3


def test_repeated_steps_are_idempotent_and_an_open_violation_is_updated_in_place(connect):
    w = make_writer(connect)
    sink = PostgresSink(w, rows.StepRecorder("dfs", 1))
    assert w.wait_ready()
    for i in range(3):
        sink.record_step(_payload(step=i, ts=time.time(), slo_breached=True,
                                  e2e={"p95_ms": 600.0 + 100 * i, "mean_ms": 1, "p50_ms": 1, "p99_ms": 1, "error_rate": 0}))
    sink.record_step(_payload(step=3, ts=time.time()))
    sink.record_step(_payload(step=4, ts=time.time()))
    sink.close()
    c = reader(connect)
    cid = only_cluster(c)
    v = queries.violations(c, cid)
    assert len(v) == 1 and v[0]["steps_breached"] == 3 and v[0]["peak_p95_ms"] == 800.0 and v[0]["ended_at"] is not None
    assert v[0]["top_services"][0]["service"] == "svc-a"


# ---------------------------------------------------------------------------------------------- HTTP API
@pytest.fixture
def api(connect, monkeypatch):
    from fastapi.testclient import TestClient
    from src import config
    from src.dashboard import server
    schema = connect().execute("SHOW search_path").fetchone()[0].split(",")[0].strip().strip('"')
    sep = "&" if "?" in URL else "?"
    monkeypatch.setattr(config, "DATABASE_URL", f"{URL}{sep}options=-csearch_path%3D{schema}")
    server.configure("demo", interval=0.02, seed=1, controller="hpa")
    return TestClient(server.app)


def _store(connect, name, svc, n=4):
    w = make_writer(connect, name)
    sink = PostgresSink(w, rows.StepRecorder("dfs", 1))
    assert w.wait_ready()
    for i in range(n):
        p = _payload(step=i, ts=time.time(), slo_breached=i < 2,
                     e2e={"p95_ms": 900.0 if i < 2 else 100.0, "mean_ms": 1, "p50_ms": 1, "p99_ms": 1, "error_rate": 0})
        p["nodes"][1]["id"] = svc
        _node(p, svc).update(proposed_delta=1, applied_delta=1)
        p["edges"] = [{"source": "gw", "target": svc, "traffic_share": 1.0}]
        sink.record_step(p)
    sink.close()


def test_history_api_is_scoped_to_one_cluster(connect, api):
    _store(connect, "cluster-a", "svc-of-a")
    _store(connect, "cluster-b", "svc-of-b")
    names = {c["name"] for c in api.get("/api/clusters").json()["clusters"]}
    assert names == {"cluster-a", "cluster-b"}
    for mine, other in (("cluster-a", "svc-of-b"), ("cluster-b", "svc-of-a")):
        for what in ("steps", "metrics", "actions", "violations", "critical-path", "topology", "events", "summary", "runs"):
            r = api.get(f"/api/history/{what}", params={"cluster": mine})
            assert r.status_code == 200, (what, r.text)
            assert other not in r.text, f"{what} of {mine} leaked {other}"
    s = api.get("/api/history/summary", params={"cluster": "cluster-a"}).json()["data"]
    assert s["steps"] == 4 and s["breached_steps"] == 2 and s["violation_episodes"] == 1
    a = api.get("/api/history/actions", params={"cluster": "cluster-a", "service": "svc-of-a"}).json()["data"]
    assert a and all(x["service"] == "svc-of-a" for x in a)


def test_history_api_errors_are_explicit(connect, api, monkeypatch):
    from src import config
    assert api.get("/api/history/steps", params={"cluster": "nope"}).status_code == 404
    assert api.get("/api/history/steps").status_code == 404, "no cluster is being controlled and none was named"
    assert api.get("/api/history/bogus", params={"cluster": "x"}).status_code == 404
    assert api.get("/api/history/steps", params={"cluster": "x", "hours": 0}).status_code == 422
    monkeypatch.setattr(config, "DATABASE_URL", "")
    r = api.get("/api/clusters")
    assert r.status_code == 503 and "AUTOSCALER_DATABASE_URL" in r.text
    assert api.get("/api/state").json()["persistence"]["enabled"] is False


def test_frames_replay_through_the_replay_engine(connect):
    from src.inference.engines import ReplayEngine
    w = make_writer(connect)
    sink = PostgresSink(w, rows.StepRecorder("dfs", 1, store_frames=True))
    assert w.wait_ready()
    for i in range(3):
        sink.record_step(_payload(step=i, ts=time.time()))
    sink.close()
    c = reader(connect)
    frames = queries.frames(c, only_cluster(c))
    e = ReplayEngine(None, frames=frames)
    p = e.tick()
    assert p["mode"] == "replay" and p["replay_length"] == 3 and [n["id"] for n in p["nodes"]] == ["gw", "svc-a"]


def test_real_train_ticket_traces_end_up_as_stored_critical_path_analysis(connect, monkeypatch):
    """The recorded Train-Ticket requests through the live engine into the database: the critical path, its evidence against the real
    requests, the CRISP-style shares and the learned topology must all be readable back."""
    import json
    from src import config
    from src.inference.engines import LiveEngine
    from src.topologies import TRAINTICKET_LIVE
    from tests.test_trainticket_replay import FIXTURE, Cluster, ReplayProm
    traces = json.load(open(FIXTURE))
    monkeypatch.setattr(config, "METRICS_SOURCE", "trainticket")
    monkeypatch.setattr(config, "TOPOLOGY", TRAINTICKET_LIVE)
    monkeypatch.setattr(config, "SERVICES", list(TRAINTICKET_LIVE.services))
    monkeypatch.setattr(config, "ENTRY_SERVICE", TRAINTICKET_LIVE.entry)
    monkeypatch.setattr(config, "CALL_GRAPH", {k: list(v) for k, v in TRAINTICKET_LIVE.call_graph.items()})
    monkeypatch.setattr(config, "CONTROLLED_SERVICES", ["ts-order-service"])
    e = LiveEngine(None, None, controller="hpa", shadow=True)
    e._prom = ReplayProm(traces)
    e._k8s = Cluster(sorted({p["serviceName"] for t in traces for p in t["processes"].values()} - {"ts-station-service"}))
    w = make_writer(connect, "train-ticket")
    e.sink = PostgresSink(w, rows.StepRecorder("dfs", 1))
    assert w.wait_ready()
    for _ in range(3):
        e.tick()
    e.close()
    c = reader(connect)
    cid = only_cluster(c)
    cp = queries.critical_paths(c, cid)
    assert cp and cp[-1]["chain"][0] == "ts-gateway-service" and "ts-order-service" in cp[-1]["chain"]
    ev = cp[0]["evidence"]
    assert ev["coverage"] > 0.7 and ev["top1"] > 0.8 and ev["random"]["coverage"] < ev["coverage"] and ev["slowest_trace"]
    assert cp[0]["profile"]["overall"] and cp[0]["profile"]["set"], "CRISP-style critical shares are stored"
    topo = queries.topology(c, cid)
    assert len(topo) == 1 and len(topo[0]["edges"]) >= 15, "the learned graph is stored once, not at every step"
    steps = queries.steps(c, cid)
    assert len(steps) == 3 and all(s["shadow"] for s in steps)
    assert {m["service"] for m in queries.metrics(c, cid)} == set(e.services)
