# History: what the agent saw and did, kept in PostgreSQL

Until now the control room showed the last 80 steps and forgot everything on restart. With a database configured the agent keeps
its own history: the actions it took, the cluster's metrics, the critical-path analysis and every SLO violation. The history of each
cluster is stored apart, so plugging the agent into another cluster never mixes data.

It is **off unless `AUTOSCALER_DATABASE_URL` is set**; without it nothing changes.

## Start it

```bash
bash scripts/22_db_up.sh                       # PostgreSQL 16 in Docker, bound to 127.0.0.1, data in a named volume
export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler     # printed by the script
export AUTOSCALER_CLUSTER_NAME=train-ticket-pc                                                           # what you call this cluster
bash scripts/19_trainticket_live.sh                                                                      # or scripts/20_autoscale_any_cluster.sh
```

`scripts/20_autoscale_any_cluster.sh` also takes `--database-url` and `--cluster-name`. If no name is given it uses the cluster name from the
kubeconfig. Open the control room and look at the **History** panel, or:

```bash
docker exec -it autoscaler-postgres psql -U autoscaler -d autoscaler
curl 'localhost:8080/api/history/summary?hours=24'          # the cluster being controlled
python -m src.persistence.export --cluster train-ticket-pc --out results/export --hours 72      # CSV per table, for the research
```

The credentials in `docker-compose.db.yml` are for a local database on `127.0.0.1`. For anything shared set `AUTOSCALER_DB_PASSWORD` before the
first start, or point `AUTOSCALER_DATABASE_URL` at a database you manage.

## What is stored, and for how long

| Table | One row per | Kept |
|---|---|---|
| `clusters` | monitored cluster and namespace (name, kube-system UID, API server, entry service) | for ever |
| `runs` | agent start: mode, shadow or act, controller, policy, git commit, **the settings it ran with** | for ever |
| `steps` | control step: end-to-end p50/p95/p99, error rate, rps, SLO, breached?, breaker, cost, hold? | 90 days |
| `service_metrics` | step x service: replicas, ready, pending, CPU/memory limit and utilisation, p95, rps, on the critical path? | **24 hours** |
| `actions` | step x service where something was proposed or blocked: proposed, guard-approved and applied replicas / CPU / memory, outcome, reason text and codes, replicas before and after | 90 days |
| `critical_paths` | every 10th step and every change: the chain, its latency, the evidence against real requests (coverage, top-1, random baseline, slowest trace), the CRISP-style critical shares | 90 days |
| `topology_snapshots` | change of the learned dependency graph | 90 days |
| `violations` | episode in which the p95 stayed above the SLO: start, end, peak, the services most often on the critical path during it, scaling actions taken, breaker opened? | 90 days |
| `events` | breaker opened / closed, hold (cluster unreadable) and recovery, scaling errors, agent start / stop | 90 days |
| `frames` | step, **only with `AUTOSCALER_STORE_FRAMES=1`**: the whole payload, for replay | 24 hours |

`outcome` is `applied` (what was proposed went through), `modified` (the guard changed it), `vetoed` (nothing went through) or `k8s_error`; `shadow` says whether
the action was only decided. Reason codes: `clamp`, `memory_floor`, `cooldown`, `stabilisation`, `young_pod`, `resize_cooldown`, `budget`, `node_memory`,
`resize_disabled`, `observed_only`, `k8s_error`, `hold`, `other` (`persistence/categorize.py`; a test fails if the code base gains a reason without a code).

Retention (`RETENTION_METRICS_HOURS=24`, `RETENTION_HISTORY_DAYS=90`) is applied every 10 minutes by the writer.

## One database, many clusters, nothing mixed

* Every fact row carries `cluster_id` and `run_id`, and `(cluster_id, run_id)` is a composite foreign key to `runs`: the database itself refuses a row whose run belongs
  to another cluster. A cluster is a (name, namespace) pair.
* The name comes from `AUTOSCALER_CLUSTER_NAME`. When the agent may read the `kube-system` namespace (optional:
  `kubectl apply -f k8s/03-autoscaler/rbac-cluster-identity.yaml`, which allows `get` on that one object) its UID is stored as a fingerprint, and then
  **the same name with another UID is refused** (history is switched off for that run, with the reason in the control room) the same UID under another name is
  recognised as the same cluster, and a name stored with a UID is also refused when the UID can no longer be read. Without that permission the name alone separates the clusters, so give each cluster its own.
* If neither a name nor the UID is available nothing is stored. The API server URL is not used: over an SSH tunnel it is `127.0.0.1:<port>` and the port changes at every
  restart of the cluster.
* The API and the page read **one** cluster at a time (`?cluster=<name or id>`, default the one being controlled); `/api/clusters` lists names only. A test writes two
  clusters into one database and checks that every route, and the summary, never returns the other one's rows; another test checks that every query in `queries.py` is filtered by `cluster_id`.
* Demo and replay modes are never stored (`AUTOSCALER_PERSIST_MODES=live`).

## It cannot hurt the control loop

The engine hands each step's rows to a bounded queue and a writer thread does the inserts in batches. If the database is slow, down or full the control
loop does not wait and does not fail: rows are retried with a growing pause, and when the queue is full the newest are dropped and counted. The control room shows
the state (`/api/state` -> `persistence`: written, dropped, queued, last error). When the database comes back the writer reconnects by itself (tested by killing its connection).
Inserts are idempotent, and `step` starts at 0 on every run, so a run is the unit of replay.

## Replay and export

```bash
AUTOSCALER_STORE_FRAMES=1 bash scripts/19_trainticket_live.sh        # also keep full payloads (about 20 KB a step, 24 h)
python -m src.dashboard.server --mode replay --replay-db train-ticket-pc --replay-hours 6     # play a stored run back, labelled as a replay
python -m src.persistence.export --cluster train-ticket-pc --table actions --format jsonl --out -
```

## Settings

| Variable | Default | |
|---|---|---|
| `AUTOSCALER_DATABASE_URL` | empty (off) | libpq URL |
| `AUTOSCALER_CLUSTER_NAME` | kubeconfig cluster name (launch scripts) | required unless the kube-system UID is readable |
| `AUTOSCALER_PERSIST_MODES` | `live` | |
| `AUTOSCALER_STORE_FRAMES` | `0` | full payloads for replay |
| `RETENTION_METRICS_HOURS` / `RETENTION_HISTORY_DAYS` | `24` / `90` | |
| `CRITICAL_PATH_SNAPSHOT_EVERY` | `10` | steps between stored critical paths (a change is always stored) |

## Tests

`tests/test_persistence_unit.py` (no database: reason codes, the row builder, violation episodes, the writer with the database calls replaced, identity, the engine hook)
and `tests/test_persistence_pg.py` (real PostgreSQL, in a private schema per test; skipped unless `TEST_DATABASE_URL` is set - CI starts a PostgreSQL service for it):

```bash
bash scripts/22_db_up.sh
TEST_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler python -m pytest tests/test_persistence_pg.py
```

They cover: migration twice, every table carrying the cluster and run, real Train-Ticket traces stored as critical-path analysis, two clusters never seen in each other's
queries (and the foreign key refusing a cross-cluster row), the name/UID conflict, retention, a killed connection, the HTTP routes, and replay from stored frames.

## Limits and next steps

* The guard explains vetoes in free text; the codes are derived from its wording. Structured reasons from `safety/guards.py` would be sturdier.
* A violation's "top services" are the ones most often on the critical path during it, a pointer to where to look and not a proof of cause.
* The database runs next to the agent. A StatefulSet for running it inside a cluster is not provided.
* Worth doing next: hourly roll-ups kept for 90 days (so purging the 24 h metrics loses no trend), an "action effect" view (p95 before and after each scale action), keeping
  the learned per-service request costs across restarts, a Grafana data source on this database, and row-level security keyed on `cluster_id` as a second layer.
