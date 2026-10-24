"""
training.py

Stores what a model was trained on, in the same PostgreSQL as the live history, so the dataset can be shown and audited:

  training_inputs        the real measurements the simulator is built from (traces, calibration, simulator parameters, the target topology, the regime)
  training_runs          one row per training run: arm, starting checkpoint, seed, hyper-parameters, the regime, the checkpoint's sha256
  training_curve         the validation curve
  training_episodes      a summary of EVERY simulated episode (topology, traffic peak, preference, p95, violations, cost, replicas, actions, penalties)
  training_trajectories  the full step-by-step, service-by-service record of a sample of episodes (what the agent saw, did and was scored)

Used by `python -m src.evaluation.train_agent ... --record` and by the helpers below:

    python -m src.persistence.training register --cluster-name train-ticket      # real traces, calibration, profile -> training_inputs
    python -m src.persistence.training attach --run <id> --file results/<evaluation>.json     # keep the evidence with the run
    python -m src.persistence.training summary                                    # what is stored
    python -m src.persistence.training export --run <id> --out results/dataset     # CSV / JSON for the panel

Recording never stops a training run: a database error is printed once and recording is switched off.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from psycopg.types.json import Jsonb

from src import config
from src.persistence import db

REGIME_KEYS = ["STEP_SECONDS", "POD_STARTUP_SECONDS", "ROLLING_SECONDS_PER_REPLICA", "EPISODE_LENGTH_STEPS", "MIN_REPLICAS", "MAX_REPLICAS",
               "INITIAL_REPLICAS", "CPU_MIN", "CPU_MAX", "CPU_QUANTUM", "CPU_DEFAULT", "MEM_MIN_MB", "MEM_MAX_MB", "MEM_QUANTUM_MB",
               "MEM_DEFAULT_MB", "BUDGET_PER_SERVICE", "SCALE_DOWN_COOLDOWN_STEPS", "SCALE_DOWN_AFTER_UP_STEPS", "RESIZE_COOLDOWN_STEPS",
               "ALLOW_RESIZE", "ALLOW_MEMORY_RESIZE", "CIRCUIT_BREAKER_VIOLATIONS", "CRITICAL_PATH_SOURCE", "SIM_PARAMS_PATH"]


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def regime():
    return {k: getattr(config, k) for k in REGIME_KEYS if hasattr(config, k)}


def topology_payload(t):
    return {"name": t.name, "services": list(t.services), "entry": t.entry,
            "call_graph": {k: [[c, w] for c, w in v] for k, v in t.call_graph.items()}, "work_ms": dict(t.work_ms)}


def add_input(conn, kind, name, payload, cluster_id=None):
    """Stores a training input once (identified by its content) and returns its id."""
    sha = _sha(payload)
    with conn.transaction():
        row = conn.execute("SELECT id FROM training_inputs WHERE kind = %s AND sha256 = %s", (kind, sha)).fetchone()
        if row:
            return row[0]
        return conn.execute("INSERT INTO training_inputs (source_cluster_id, kind, name, sha256, payload) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                            (cluster_id, kind, name, sha, Jsonb(payload))).fetchone()[0]


class TrainingRecorder:
    def __init__(self, url, arm, seed, episodes, init_from=None, hparams=None, topo_options=None, cluster_name=None, namespace="default",
                 trajectory_every=25, note=None, target_topologies=()):
        self.url, self.every, self.disabled = url, max(1, trajectory_every), False
        self._eps, self._traj, self._curve, self.run_id = [], [], [], None
        try:
            self.conn = db.connect(url)
            db.migrate(self.conn)
            cid = db.register_cluster(self.conn, cluster_name, namespace) if cluster_name else None
            inputs = [add_input(self.conn, "regime", "training regime (environment settings)", regime(), cid)]
            sp = config.SIM_PARAMS_PATH
            if os.path.exists(sp):
                inputs.append(add_input(self.conn, "sim_params", os.path.basename(sp), json.load(open(sp)), cid))
            for t in target_topologies:
                inputs.append(add_input(self.conn, "topology", t.name, topology_payload(t), cid))
            if cid:                                   # the real measurements registered for this cluster (python -m src.persistence.training register)
                with self.conn.transaction():         # (a bare query would leave a transaction open, and closing the connection would roll everything back)
                    inputs += [r[0] for r in self.conn.execute(
                        "SELECT id FROM training_inputs WHERE source_cluster_id = %s AND kind IN ('traces', 'calibration', 'profile')", (cid,)).fetchall()]
            with self.conn.transaction():
                self.run_id = self.conn.execute(
                    "INSERT INTO training_runs (source_cluster_id, arm, init_from, seed, episodes, hparams, regime, topology_options, input_ids, git_sha, note) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (cid, arm, init_from, seed, episodes, Jsonb(hparams or {}), Jsonb(regime()), Jsonb(topo_options or {}), inputs,
                     db.git_sha(), note)).fetchone()[0]
            print(f"[record] training run {self.run_id} is being stored in the database", flush=True)
        except Exception as e:
            self._fail(e)

    def _fail(self, e):
        if not self.disabled:
            print(f"[record] database unavailable, recording switched off: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        self.disabled = True

    # -- per episode ------------------------------------------------------
    def on_episode(self, ep, env, w, obs_list, idx_list, cont_list, rew):
        if self.disabled:
            return
        try:
            steps = obs_list[1:]
            if not steps:
                return
            p95 = [o["e2e"]["p95_ms"] for o in steps]
            slo = float(env.slo_ms)
            pen = -np.asarray(rew, dtype=float)                                 # reward = -penalty
            deltas = [np.asarray(config.REPLICA_DELTA_OPTIONS)[np.asarray(i)] for i in idx_list]
            self._eps.append((ep, env.topology.name, len(env.services), float(max(o["external_rps"] for o in steps)),
                              [float(x) for x in np.asarray(w)], len(steps), float(np.mean(p95)), float(np.max(p95)), slo,
                              int(sum(x > slo for x in p95)), float(np.mean([o["cost"] for o in steps])),
                              float(np.mean([sum(m["replicas"] for m in o["metrics"].values()) for o in steps])),
                              int(sum(int((d != 0).sum()) for d in deltas)), [float(x) for x in pen.mean(axis=0)]))
            if ep % self.every == 0:
                traj = []
                for t, o in enumerate(steps):
                    traj.append({"step": t + 1, "rps": round(float(o["external_rps"]), 2), "p95_ms": round(float(o["e2e"]["p95_ms"]), 1),
                                 "cost": round(float(o["cost"]), 3), "penalty": [round(float(x), 4) for x in pen[t]],
                                 "services": {s: {"replicas": int(o["metrics"][s]["replicas"]), "cpu": round(float(o["metrics"][s]["cpu"]), 3),
                                                  "mem": round(float(o["metrics"][s]["mem"])), "util": round(float(o["metrics"][s]["utilization"]), 3),
                                                  "p95_ms": round(float(o["metrics"][s]["latency_p95_ms"]), 1),
                                                  "rps": round(float(o["metrics"][s]["requests"]), 2), "delta": int(deltas[t][i])}
                                              for i, s in enumerate(env.services)}})
                self._traj.append((ep, env.topology.name, traj))
            if len(self._eps) >= 25:
                self._flush()
        except Exception as e:
            self._fail(e)

    def on_validation(self, entry):
        if self.disabled:
            return
        losses = {k: v for k, v in entry.items() if k not in ("episode", "val_scalarized", "val_p95_ms", "val_cost", "val_slo_violation", "seconds")}
        self._curve.append((entry["episode"], entry["val_scalarized"], entry["val_p95_ms"], entry["val_cost"], entry["val_slo_violation"],
                            entry.get("seconds"), losses))
        self._flush()

    def _flush(self):
        if self.disabled or self.run_id is None:
            return
        try:
            with self.conn.transaction():
                with self.conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO training_episodes (run_id, episode, topology, n_services, pattern_rps_peak, preference, steps, mean_p95_ms, max_p95_ms, "
                        "slo_ms, violation_steps, mean_cost, mean_replicas, scale_actions, mean_penalty) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                        "%s, %s, %s, %s) ON CONFLICT DO NOTHING", [(self.run_id,) + r for r in self._eps])
                    cur.executemany("INSERT INTO training_trajectories (run_id, episode, topology, steps) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                                    [(self.run_id, e, t, Jsonb(s)) for e, t, s in self._traj])
                    cur.executemany(
                        "INSERT INTO training_curve (run_id, episode, val_score, val_p95_ms, val_cost, val_violation, seconds, losses) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        [(self.run_id,) + c[:6] + (Jsonb(c[6]),) for c in self._curve])
            self._eps, self._traj, self._curve = [], [], []
        except Exception as e:
            self._fail(e)

    def finish(self, checkpoint_path=None, meta=None, status="finished"):
        if self.disabled or self.run_id is None:
            return
        self._flush()
        try:
            meta = meta or {}
            sha = file_sha256(checkpoint_path) if checkpoint_path and os.path.exists(checkpoint_path) else None
            with self.conn.transaction():
                self.conn.execute("UPDATE training_runs SET ended_at = now(), status = %s, checkpoint_path = %s, checkpoint_sha256 = %s, best_val = %s, "
                                  "final_val = %s WHERE id = %s",
                                  (status, checkpoint_path, sha, meta.get("best_val_scalarized"), meta.get("selected_fresh_val_scalarized"), self.run_id))
            self.conn.close()
            print(f"[record] training run {self.run_id} stored ({status})", flush=True)
        except Exception as e:
            self._fail(e)


# ---------------------------------------------------------------------------------------------------- CLI
def _register(a):
    here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    repo = os.path.dirname(here)
    with db.connect(config.DATABASE_URL) as conn:
        db.migrate(conn)
        cid = db.register_cluster(conn, a.cluster_name, a.namespace)
        items = [("traces", "tests/fixtures/trainticket_traces.json", os.path.join(here, "tests", "fixtures", "trainticket_traces.json")),
                 ("calibration", "results/trainticket_calibration.json", os.path.join(here, "results", "trainticket_calibration.json")),
                 ("profile", "profiles/trainticket.env", os.path.join(repo, "profiles", "trainticket.env"))]
        for kind, name, path in items:
            if not os.path.exists(path):
                print(f"skip {name}: not found")
                continue
            payload = {"text": open(path).read()} if kind == "profile" else json.load(open(path))
            iid = add_input(conn, kind, name, payload, cid)
            n = len(payload) if isinstance(payload, list) else ""
            print(f"{kind:12} {name} {('(' + str(n) + ' items)') if n != '' else ''} -> {iid}")


def _attach(a):
    """Store a result file (an evaluation, a comparison) with a training run, so the dataset and the evidence stay together."""
    with db.connect(config.DATABASE_URL) as conn:
        payload = json.load(open(a.file))
        iid = add_input(conn, a.kind, a.name or os.path.basename(a.file), payload)
        with conn.transaction():
            conn.execute("UPDATE training_runs SET input_ids = array_append(input_ids, %s) WHERE id = %s AND NOT (%s = ANY(input_ids))", (iid, a.run, iid))
        print(f"{a.kind} {a.file} -> {iid} (attached to run {a.run})")


def _summary(a):
    with db.connect(config.DATABASE_URL, autocommit=True) as conn:
        for r in conn.execute("SELECT kind, name, pg_column_size(payload) AS bytes, created_at FROM training_inputs ORDER BY created_at").fetchall():
            print(f"input   {r[0]:12} {r[1]:45} {r[2]:>9} bytes")
        for r in conn.execute(
                "SELECT r.id, r.started_at, r.arm, r.init_from, r.seed, r.episodes, r.status, r.best_val, "
                "(SELECT count(*) FROM training_episodes e WHERE e.run_id = r.id), (SELECT count(*) FROM training_trajectories t WHERE t.run_id = r.id), "
                "(SELECT count(*) FROM training_curve c WHERE c.run_id = r.id) FROM training_runs r ORDER BY r.started_at").fetchall():
            print(f"run     {r[0]}  {r[1]:%Y-%m-%d %H:%M}  {r[2]}  from={r[3]}  seed={r[4]}  episodes={r[5]}  {r[6]}  best_val={r[7]}  "
                  f"stored: {r[8]} episodes, {r[9]} trajectories, {r[10]} validations")


def _export(a):
    from src.persistence.export import _write
    os.makedirs(a.out, exist_ok=True)
    with db.connect(config.DATABASE_URL, autocommit=True) as conn:
        from psycopg.rows import dict_row
        for table, order in (("training_runs", "started_at"), ("training_curve", "episode"), ("training_episodes", "episode"), ("training_trajectories", "episode")):
            col = "id" if table == "training_runs" else "run_id"
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SELECT * FROM {table} WHERE {col} = %s ORDER BY {order}", (a.run,))
                rows = cur.fetchall()
            fmt = "jsonl" if table == "training_trajectories" else "csv"
            path = os.path.join(a.out, f"{table}.{fmt}")
            with open(path, "w", newline="") as fh:
                _write(rows, fmt, fh)
            print(f"{path}: {len(rows)} rows")
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT i.kind, i.name, i.sha256, i.payload FROM training_inputs i WHERE i.id IN (SELECT unnest(input_ids) FROM training_runs WHERE id = %s)", (a.run,))
            for r in cur.fetchall():
                path = os.path.join(a.out, f"input_{r['kind']}_{r['sha256'][:8]}.json")
                json.dump(r["payload"], open(path, "w"), indent=1, default=str)
                print(f"{path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="store the real measurements (traces, calibration, profile) as training inputs")
    r.add_argument("--cluster-name", required=True)
    r.add_argument("--namespace", default="default")
    sub.add_parser("summary", help="list stored inputs and training runs")
    t = sub.add_parser("attach", help="store a result file (e.g. results/trainticket_sim_cpu_cap2.json) with a training run")
    t.add_argument("--run", required=True)
    t.add_argument("--file", required=True)
    t.add_argument("--kind", default="evaluation")
    t.add_argument("--name", default=None)
    e = sub.add_parser("export", help="write one run and its inputs to files")
    e.add_argument("--run", required=True)
    e.add_argument("--out", default="results/dataset")
    a = ap.parse_args(argv)
    if not config.DATABASE_URL:
        ap.error("set AUTOSCALER_DATABASE_URL")
    {"register": _register, "summary": _summary, "export": _export, "attach": _attach}[a.cmd](a)


if __name__ == "__main__":
    main()
