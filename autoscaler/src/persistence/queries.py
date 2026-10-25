"""
queries.py

Reads for the dashboard's History panel, the export tool and replay. Every query that touches data is scoped by ONE cluster_id (a
test checks that no SQL here forgets it); the only unscoped read is the list of cluster names. Times are timestamptz; `since` / `until`
default to the last 24 hours.
"""

import uuid
from datetime import datetime, timedelta, timezone

from psycopg.rows import dict_row

DEFAULT_WINDOW = timedelta(hours=24)
MAX_LIMIT = 20000


def _window(since, until):
    until = until or datetime.now(timezone.utc)
    return (since or until - DEFAULT_WINDOW), until


def _rows(conn, sql, params):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _limit(n):
    return max(1, min(int(n), MAX_LIMIT))


def clusters(conn):
    """The monitored clusters (names only - no history)."""
    return _rows(conn, "SELECT c.id::text AS id, c.name, c.namespace, c.api_server, c.entry_service, c.created_at, c.last_seen_at, "
                       "(SELECT count(*) FROM runs r WHERE r.cluster_id = c.id) AS runs FROM clusters c ORDER BY c.last_seen_at DESC", ())


def resolve_cluster(conn, ref):
    """A cluster id or a name -> id (str), or None. A name used in several namespaces is ambiguous and also returns None."""
    if not ref:
        return None
    try:
        ref = str(uuid.UUID(str(ref)))
        rows = _rows(conn, "SELECT id::text AS id FROM clusters WHERE id = %s", (ref,))
    except ValueError:
        rows = _rows(conn, "SELECT id::text AS id FROM clusters WHERE name = %s", (ref,))
    return rows[0]["id"] if len(rows) == 1 else None


def runs(conn, cluster_id, limit=50):
    return _rows(conn, "SELECT id::text AS id, started_at, ended_at, mode, shadow, controller, policy, git_sha, settings FROM runs "
                       "WHERE cluster_id = %s ORDER BY started_at DESC LIMIT %s", (cluster_id, _limit(limit)))


def steps(conn, cluster_id, since=None, until=None, limit=5000):
    since, until = _window(since, until)
    return _rows(conn, "SELECT run_id::text AS run_id, step, ts, rps, p50_ms, p95_ms, p99_ms, error_rate, slo_ms, slo_breached, breaker_open, "
                       "total_replicas, cost, budget, stale, stale_reason, shadow, controller FROM steps "
                       "WHERE cluster_id = %s AND ts >= %s AND ts <= %s ORDER BY ts LIMIT %s", (cluster_id, since, until, _limit(limit)))


def metrics(conn, cluster_id, since=None, until=None, service=None, limit=5000):
    since, until = _window(since, until)
    return _rows(conn, "SELECT run_id::text AS run_id, step, ts, service, replicas, ready, pending, cpu_limit, mem_limit_mb, cpu_util, mem_util, "
                       "latency_ms, p95_ms, rps, on_critical_path, controlled FROM service_metrics "
                       "WHERE cluster_id = %s AND ts >= %s AND ts <= %s AND (%s::text IS NULL OR service = %s) ORDER BY ts, service LIMIT %s",
                 (cluster_id, since, until, service, service, _limit(limit)))


def actions(conn, cluster_id, since=None, until=None, service=None, outcome=None, changed_only=False, limit=2000):
    """`changed_only`: only actions that really changed something in the cluster - not shadow decisions - a replica count, or a CPU limit different from the one the service had at that step."""
    since, until = _window(since, until)
    return _rows(conn, "SELECT a.run_id::text AS run_id, a.step, a.ts, a.service, a.shadow, a.outcome, a.proposed_dr, a.guard_dr, a.applied_dr, a.proposed_cpu, "
                       "a.applied_cpu, m.cpu_limit AS cpu_before, a.proposed_mem_mb, a.applied_mem_mb, a.replicas_before, a.replicas_after, a.reason, a.reason_codes, "
                       "a.cpu_util, a.p95_ms FROM actions a LEFT JOIN service_metrics m ON m.run_id = a.run_id AND m.step = a.step AND m.service = a.service "
                       "WHERE a.cluster_id = %s AND a.ts >= %s AND a.ts <= %s AND (%s::text IS NULL OR a.service = %s) "
                       "AND (%s::text IS NULL OR a.outcome = %s) "
                       "AND (NOT %s OR (NOT a.shadow AND a.outcome IN ('applied', 'modified') AND (a.applied_dr <> 0 OR a.replicas_after <> a.replicas_before "
                       "OR (m.cpu_limit IS NOT NULL AND abs(a.applied_cpu - m.cpu_limit) > 0.0001)))) ORDER BY a.ts DESC LIMIT %s",
                 (cluster_id, since, until, service, service, outcome, outcome, bool(changed_only), _limit(limit)))


def violations(conn, cluster_id, since=None, until=None, limit=500):
    since, until = _window(since, until)
    return _rows(conn, "SELECT run_id::text AS run_id, start_step, started_at, ended_at, end_step, steps_breached, peak_p95_ms, slo_ms, top_services, "
                       "actions, breaker_opened, slowest_trace FROM violations WHERE cluster_id = %s AND started_at >= %s AND started_at <= %s "
                       "ORDER BY started_at DESC LIMIT %s", (cluster_id, since, until, _limit(limit)))


def critical_paths(conn, cluster_id, since=None, until=None, changed_only=False, limit=1000):
    since, until = _window(since, until)
    return _rows(conn, "SELECT run_id::text AS run_id, step, ts, source, chain, chain_latency_ms, changed, evidence, profile, slowest_trace "
                       "FROM critical_paths WHERE cluster_id = %s AND ts >= %s AND ts <= %s AND (NOT %s OR changed) ORDER BY ts DESC LIMIT %s",
                 (cluster_id, since, until, bool(changed_only), _limit(limit)))


def topology(conn, cluster_id, since=None, until=None, limit=50):
    since, until = _window(since, until)
    return _rows(conn, "SELECT run_id::text AS run_id, step, ts, edges_hash, n_services, edges FROM topology_snapshots "
                       "WHERE cluster_id = %s AND ts >= %s AND ts <= %s ORDER BY ts DESC LIMIT %s", (cluster_id, since, until, _limit(limit)))


def events(conn, cluster_id, since=None, until=None, limit=500):
    since, until = _window(since, until)
    return _rows(conn, "SELECT run_id::text AS run_id, step, ts, type, severity, detail FROM events "
                       "WHERE cluster_id = %s AND ts >= %s AND ts <= %s ORDER BY ts DESC LIMIT %s", (cluster_id, since, until, _limit(limit)))


def frames(conn, cluster_id, since=None, until=None, limit=MAX_LIMIT):
    since, until = _window(since, until)
    return [r["payload"] for r in _rows(
        conn, "SELECT payload FROM frames WHERE cluster_id = %s AND ts >= %s AND ts <= %s ORDER BY ts LIMIT %s",
        (cluster_id, since, until, _limit(limit)))]


def summary(conn, cluster_id, since=None, until=None):
    """Counts for the History header: steps, SLO-violation steps and episodes, actions by outcome, vetoes by reason code."""
    since, until = _window(since, until)
    s = _rows(conn, "SELECT count(*) AS steps, count(*) FILTER (WHERE slo_breached) AS breached_steps, count(*) FILTER (WHERE stale) AS stale_steps, "
                    "max(p95_ms) AS max_p95_ms FROM steps WHERE cluster_id = %s AND ts >= %s AND ts <= %s", (cluster_id, since, until))[0]
    s["violation_episodes"] = _rows(conn, "SELECT count(*) AS n FROM violations WHERE cluster_id = %s AND started_at >= %s AND started_at <= %s",
                                    (cluster_id, since, until))[0]["n"]
    s["actions_by_outcome"] = {r["outcome"]: r["n"] for r in _rows(
        conn, "SELECT outcome, count(*) AS n FROM actions WHERE cluster_id = %s AND ts >= %s AND ts <= %s GROUP BY outcome",
        (cluster_id, since, until))}
    s["reasons"] = {r["code"]: r["n"] for r in _rows(
        conn, "SELECT code, count(*) AS n FROM actions, unnest(reason_codes) AS code WHERE cluster_id = %s AND ts >= %s AND ts <= %s "
              "GROUP BY code ORDER BY n DESC", (cluster_id, since, until))}
    return s
