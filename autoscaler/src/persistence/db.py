"""
db.py

Everything that talks to PostgreSQL: connect, migrate, register the cluster and the run, insert rows, purge old ones. Plain psycopg 3,
no ORM; the schema is the numbered files in schema/. Used by the writer thread (writes) and by the read API (queries.py) with its own connections.
"""

import os
import re
import subprocess

import psycopg
from psycopg.types.json import Jsonb

from src.persistence import rows as rows_mod

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schema")


class IdentityError(Exception):
    """The cluster cannot be identified, so nothing may be stored for it."""


class IdentityConflict(IdentityError):
    """The cluster name is already stored for a different cluster (its kube-system UID differs)."""


def connect(url, **kw):
    return psycopg.connect(url, connect_timeout=5, **kw)


def migrate(conn):
    """Applies schema/NNN_*.sql files not applied yet. Safe to call from several processes at once (advisory lock)."""
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(727001)")
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
        done = {r[0] for r in conn.execute("SELECT version FROM schema_version")}
        for name in sorted(os.listdir(SCHEMA_DIR)):
            m = re.match(r"(\d+)_.*\.sql$", name)
            if not m or int(m.group(1)) in done:
                continue
            with open(os.path.join(SCHEMA_DIR, name)) as f:
                conn.execute(f.read())
            conn.execute("INSERT INTO schema_version (version) VALUES (%s)", (int(m.group(1)),))


def register_cluster(conn, name, namespace, kube_uid=None, api_server=None, entry=None):
    """-> cluster_id. A cluster is (name, namespace); when the kube-system UID is known it is stored and later checked: the same
    name with another UID is a different cluster and raises IdentityConflict instead of mixing the two. The same UID under a new
    name is the same cluster (found by UID)."""
    with conn.transaction():
        row = None
        if kube_uid:
            row = conn.execute("SELECT id, name FROM clusters WHERE kube_uid = %s AND namespace = %s", (kube_uid, namespace)).fetchone()
        if row is None:
            row = conn.execute("SELECT id, kube_uid FROM clusters WHERE name = %s AND namespace = %s", (name, namespace)).fetchone()
            if row is not None and row[1] and kube_uid and row[1] != kube_uid:
                raise IdentityConflict(f"cluster name {name!r} (namespace {namespace!r}) is already used by another cluster "
                                       f"(kube-system UID {row[1][:8]}... vs {kube_uid[:8]}...); choose another AUTOSCALER_CLUSTER_NAME")
            if row is not None and row[1] and not kube_uid:
                raise IdentityConflict(f"cluster {name!r} (namespace {namespace!r}) was stored with its kube-system UID, which cannot be read now, "
                                       f"so this cannot be told apart from another cluster of the same name; allow reading the kube-system "
                                       f"namespace again (k8s/03-autoscaler/rbac-cluster-identity.yaml) or choose another AUTOSCALER_CLUSTER_NAME")
        if row is None:
            return conn.execute(
                "INSERT INTO clusters (name, namespace, kube_uid, api_server, entry_service) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (name, namespace, kube_uid, api_server, entry)).fetchone()[0]
        conn.execute("UPDATE clusters SET last_seen_at = now(), kube_uid = COALESCE(kube_uid, %s), api_server = COALESCE(%s, api_server),"
                     " entry_service = COALESCE(%s, entry_service) WHERE id = %s", (kube_uid, api_server, entry, row[0]))
        return row[0]


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3,
                              cwd=os.path.dirname(__file__)).stdout.strip() or None
    except Exception:
        return None


def start_run(conn, cluster_id, mode, shadow, controller, policy, slo_ms, settings):
    with conn.transaction():
        return conn.execute(
            "INSERT INTO runs (cluster_id, mode, shadow, controller, policy, slo_ms, git_sha, settings) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (cluster_id, mode, shadow, controller, policy, slo_ms, git_sha(), Jsonb(settings))).fetchone()[0]


def end_run(conn, run_id):
    with conn.transaction():
        conn.execute("UPDATE runs SET ended_at = now() WHERE id = %s", (run_id,))


def _adapt(table, row):
    cols = rows_mod.COLUMNS[table]
    return tuple(Jsonb(v) if c in rows_mod.JSON_COLUMNS and v is not None else v for c, v in zip(cols, row))


def insert_rows(conn, cluster_id, run_id, batch):
    """batch: {table: [row tuples in COLUMNS order]}; one transaction. Every row gets this cluster and run."""
    with conn.transaction():
        with conn.cursor() as cur:
            for table, rows in batch.items():
                if not rows:
                    continue
                cols = rows_mod.COLUMNS[table]
                names = ", ".join(["cluster_id", "run_id"] + cols)
                marks = ", ".join(["%s"] * (len(cols) + 2))
                key = rows_mod.KEYS.get(table)
                if table == "violations":
                    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("start_step", "started_at"))
                    tail = f" ON CONFLICT {key} DO UPDATE SET {sets}"
                elif key:
                    tail = f" ON CONFLICT {key} DO NOTHING"
                else:
                    tail = ""
                cur.executemany(f"INSERT INTO {table} ({names}) VALUES ({marks}){tail}",
                                [(cluster_id, run_id) + _adapt(table, r) for r in rows])


# table -> (timestamp column, which retention it follows)
RETENTION = {"service_metrics": ("ts", "metrics"), "frames": ("ts", "metrics"),
             "steps": ("ts", "history"), "actions": ("ts", "history"), "critical_paths": ("ts", "history"),
             "topology_snapshots": ("ts", "history"), "violations": ("started_at", "history"), "events": ("ts", "history")}


def purge(conn, metrics_hours, history_days):
    """Deletes rows older than the retention of their table (all clusters alike - retention is a property of the data, not of a cluster).
    Returns {table: rows deleted}."""
    deleted = {}
    for table, (col, kind) in RETENTION.items():
        interval = f"{metrics_hours} hours" if kind == "metrics" else f"{history_days} days"
        with conn.transaction():
            cur = conn.execute(f"DELETE FROM {table} WHERE {col} < now() - %s::interval", (interval,))
            deleted[table] = cur.rowcount
    return deleted
