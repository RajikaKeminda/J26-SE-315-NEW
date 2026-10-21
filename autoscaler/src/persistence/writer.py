"""
writer.py

A bounded queue and one daemon thread. The control loop only ever calls put(): it never waits for the database, never raises, and
if the database is slow or down the oldest unwritten data is simply not kept (counted in `dropped`) - history is optional, the
control of the cluster is not.

The thread connects, migrates, registers the cluster and the run (retrying with a growing pause while the database is unreachable),
then writes batches, one transaction each, and purges expired rows every `purge_every_s`.
"""

import queue
import threading
import time

from src.persistence import db

STOP = object()


class Writer:
    def __init__(self, url, identity, run_info, retention, maxsize=2000, purge_every_s=600.0, batch_max=50, connect_fn=None):
        """identity: callable -> dict(name, namespace, kube_uid, api_server, entry) (called in the thread, may do I/O);
        run_info: dict(mode, shadow, controller, policy, slo_ms, settings); retention: (metrics_hours, history_days)."""
        self.url, self.identity, self.run_info, self.retention = url, identity, run_info, retention
        self.q = queue.Queue(maxsize=maxsize)
        self.batch_max, self.purge_every_s = batch_max, purge_every_s
        self._connect = connect_fn or (lambda: db.connect(self.url))
        self.cluster_id = self.run_id = None
        self.cluster_name = None
        self._stopping = False
        self.state = "starting"            # starting | ready | disabled | stopped
        self.last_error = None
        self.written = self.dropped = self.purged = 0
        self._thread = threading.Thread(target=self._run, name="persistence-writer", daemon=True)
        self._thread.start()

    # -- called from the control loop ------------------------------------
    def put(self, batch):
        if self.state in ("disabled", "stopped"):
            return False
        try:
            self.q.put_nowait(batch)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def status(self):
        return {"enabled": True, "state": self.state, "cluster": self.cluster_name, "cluster_id": str(self.cluster_id) if self.cluster_id else None,
                "run_id": str(self.run_id) if self.run_id else None, "written_steps": self.written, "dropped": self.dropped,
                "queued": self.q.qsize(), "last_error": self.last_error}

    def close(self, timeout=5.0):
        """Flush what is queued and stop. Never hangs: with the database down and the queue full, the oldest rows are discarded to
        make room for the stop signal and the retry loop gives up."""
        self._stopping = True
        while True:
            try:
                self.q.put_nowait(STOP)
                break
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
        self._thread.join(timeout)

    def wait_ready(self, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end and self.state == "starting":
            time.sleep(0.02)
        return self.state == "ready"

    # -- thread ----------------------------------------------------------
    def _setup(self, conn):
        """First connection: migrate, register the cluster, start the run. A reconnect after an outage continues the same run."""
        if self.run_id is not None:
            return
        ident = self.identity()
        db.migrate(conn)
        self.cluster_id = db.register_cluster(conn, ident["name"], ident["namespace"], ident.get("kube_uid"),
                                              ident.get("api_server"), ident.get("entry"))
        self.cluster_name = ident["name"]
        r = self.run_info
        self.run_id = db.start_run(conn, self.cluster_id, r["mode"], r["shadow"], r.get("controller"), r.get("policy"),
                                   r.get("slo_ms"), r.get("settings", {}))
        db.insert_rows(conn, self.cluster_id, self.run_id, {"events": [(None, _now(), "agent_start", "info", {"mode": r["mode"], "shadow": r["shadow"]})]})

    def _run(self):
        conn, pause, last_purge, pending = None, 1.0, 0.0, []
        while True:
            try:
                if conn is None:
                    conn = self._connect()
                    self._setup(conn)
                    self.state, pause, self.last_error = "ready", 1.0, None
                if not pending:
                    try:
                        item = self.q.get(timeout=1.0)
                    except queue.Empty:
                        item = None
                    if item is STOP:
                        self._finish(conn)
                        return
                    if item is not None:
                        pending.append(item)
                        while len(pending) < self.batch_max:
                            try:
                                nxt = self.q.get_nowait()
                            except queue.Empty:
                                break
                            if nxt is STOP:
                                self._flush(conn, pending)
                                self._finish(conn)
                                return
                            pending.append(nxt)
                if pending:
                    self._flush(conn, pending)
                    pending = []
                if time.time() - last_purge > self.purge_every_s:
                    last_purge = time.time()
                    self.purged += sum(db.purge(conn, *self.retention).values())
            except db.IdentityError as e:
                self.state, self.last_error = "disabled", str(e)
                self._drain()
                return
            except Exception as e:                     # database down / restarted / disk full: keep the batch, retry later
                self.last_error = f"{type(e).__name__}: {e}"
                if self.state == "ready":
                    self.state = "starting"
                try:
                    conn and conn.close()
                except Exception:
                    pass
                conn = None
                if self._stopping:                     # closing while the database is unreachable: give up instead of retrying
                    self.state = "stopped"
                    return
                time.sleep(pause)
                pause = min(pause * 2, 30.0)

    def _flush(self, conn, pending):
        merged = {}
        for b in pending:
            for table, rows in b.items():
                merged.setdefault(table, []).extend(rows)
        db.insert_rows(conn, self.cluster_id, self.run_id, merged)
        self.written += len(merged.get("steps", []))

    def _finish(self, conn):
        try:
            db.insert_rows(conn, self.cluster_id, self.run_id, {"events": [(None, _now(), "agent_stop", "info", {})]})
            db.end_run(conn, self.run_id)
            conn.close()
        except Exception:
            pass
        self.state = "stopped"

    def _drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
