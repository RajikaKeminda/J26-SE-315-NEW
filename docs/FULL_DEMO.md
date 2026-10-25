# Full demonstration - step by step (about 60 minutes, or 25 with the short route)

Every command was run during this project. Run from the repository root unless a block says `cd autoscaler`. `$K` = `kubectl --kubeconfig ~/.kube/tt-agent-config`.
Times are suggestions. **Short route (25 min): parts 1, 3, 4, 5 and 7.**

## Part 0 - the day before and 30 minutes before

**Windows PC (the cluster).** Docker Desktop and minikube running, all Train-Ticket pods Ready, SkyWalking still stopped (the node needs its 850 MiB), nothing else heavy running.
Sync the clock once (admin PowerShell: `w32tm /resync`; it was 12 s ahead). Find the Kubernetes API port, because it changes after every Docker/minikube restart:
```
kubectl config view --minify | findstr server
```
If the other Claude session is used, tell it: read-only checks only, and wait for "GO" before starting any load. Otherwise run the load yourself (below).

**Mac.**
```bash
# 1. tunnels: Prometheus, Jaeger and the API port from above (one command; never pkill the tunnels)
ssh -fN -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
  -L 9090:127.0.0.1:9090 -L 16686:127.0.0.1:16686 -L <api-port>:127.0.0.1:<api-port> "<user>"@<tailscale-ip>
# 2. the API port in the kubeconfig must match; the 24 h token must be valid
grep server: ~/.kube/tt-agent-config
kubectl --kubeconfig ~/.kube/tt-agent-config auth can-i patch deployments            # must print: yes
#    expired?  on the PC:  kubectl create token autoscaler-agent -n default --duration=24h   -> then on the Mac:
#    kubectl --kubeconfig ~/.kube/tt-agent-config config set-credentials autoscaler-agent --token=<new>
# 3. history database
bash scripts/22_db_up.sh
export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler
export AUTOSCALER_CLUSTER_NAME=train-ticket
# 4. preflight: prints what is wrong (tunnel, token, traces, clock); Ctrl+C when the checks pass
bash scripts/19_trainticket_live.sh
```
**Baseline of the demo service** - it must read exactly this, or fix it before starting:
```bash
$K get deploy ts-order-service -o jsonpath='{.spec.replicas} {.status.readyReplicas} {.spec.template.spec.containers[0].resources}{"\n"}'
# 1 1 {"limits":{"cpu":"1","memory":"800Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}
```
**Fallback recording** (so there is something to show if the network fails): run a shadow session with load for 10 minutes while
`cd autoscaler && python -m src.dashboard.record --out results/replays/trainticket_run.jsonl --seconds 600`. A recorded run is already in the repository.

## Part 1 - the idea (3 min, no commands)

"A reinforcement-learning agent decides, for every service at once, how many pods and how much CPU. It reads two graphs built from live traces - who calls whom, and how much each call costs right now - so it works for any number of services.
It was trained in a simulator fitted to real measurements, and a safety layer always sits between it and the cluster."

## Part 2 - the simulator control room (4 min)
```bash
bash scripts/10_run_dashboard.sh            # http://localhost:8080 (stop it with Ctrl+C before part 5)
```
* **Live tab:** the dependency graph (scroll to zoom, drag to move), the red chain is the critical path, the decision feed shows every action *and every blocked action with the reason*.
* Drag the **SLO** slider to 100 and the others to 0: the same network adds capacity. Drag **cost** up: it saves money. "One network, any trade-off."
* Switch the controller MO-TD3 -> HPA -> static during the burst scenario: the baselines' p95 climbs.
* **Analysis tab:** controller comparison, cost-versus-latency Pareto front, simulator calibration (capacity within 3 % of a real pod), training curves.

## Part 3 - the models and the training data (6 min)
```bash
ls autoscaler/models/*.pt | head -20
cd autoscaler
python -m src.persistence.training summary                    # every stored training input and run, with counts
docker exec -it autoscaler-postgres psql -U autoscaler -d autoscaler
```
Say: `mo_td3_full` (4-12 services) -> fine-tuned to `mo_td3_large` (4-64 services) -> fine-tuned to `mo_td3_trainticket_cpu` (replicas + CPU on the real Train-Ticket's timing, memory locked).
Fine-tuning = warm-starting a checkpoint and continuing to train in the simulator under the target cluster's settings.
In psql:
```sql
select arm, init_from, seed, episodes, status, best_val, left(checkpoint_sha256, 12) sha from training_runs;
select kind, name, pg_column_size(payload) bytes from training_inputs;                       -- the real measurements the simulator is built from
select episode, topology, n_services, round(mean_p95_ms) p95, violation_steps, scale_actions from training_episodes order by episode limit 8;
select episode, val_score, val_p95_ms, val_violation from training_curve order by episode;   -- the learning curve
select jsonb_array_length(steps) steps, steps->0->'services'->'ts-order-service' from training_trajectories limit 1;   -- one step, one service: what it saw and did
```
Training itself (16 minutes, offline): `bash scripts/23_train_trainticket_cpu.sh`. Files for the panel: `autoscaler/results/dataset/`.
Simulator result (2-replica cap): the new model equals the tuned HPA on SLO violations (48.2 % vs 48.1 %) at lower cost (9.7 vs 13.1); `docs/RUN_LOG_TRAINTICKET_CPU.md` section 1.

## Part 4 - plug into a cluster it has never seen (4 min)
```bash
bash scripts/20_autoscale_any_cluster.sh --namespace default --kubeconfig ~/.kube/tt-agent-config --discover-only
```
Read it aloud: the entry service it found from the traces, which services it can see and control, action bounds from the real limits and measured memory, the replica cap from node headroom, the pod start-up measured from real pods, shadow mode by default.
Nothing was told to it except the namespace and the two URLs.

## Part 5 - the live demonstration on Train-Ticket (about 25 min)

**5.1 Stage the bottleneck** (the service gets a 0.3-core limit, one pod; undone in 5.8):
```bash
bash scripts/21_tt_demo_bottleneck.sh create 300m          # waits for the new pod; the pod must show 1/1 before you go on
```
**5.2 Start the agent in SHADOW mode** (it decides, nothing changes; the database is on - run this in the terminal where you did the two `export` lines of part 0, or the history is not stored):
```bash
ARM=mo_td3_trainticket_cpu RESIZE=cpu SLO_MS=3000 CONTROLLED_SERVICES=ts-order-service NODE_RESERVE_MB=600 \
  bash scripts/19_trainticket_live.sh                      # http://localhost:8080
```
Open also Jaeger (http://localhost:16686) and, in a second terminal, `$K get pods -w`.

**5.3 Start the load on the PC** (3 users; 75 s warm-up then 18 minutes). Either ask the PC session for "phase 3: load 1080 s", or in PowerShell on the PC:
```
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\train ticket\ts-calibrate.ps1" -LevelList 3 -Seconds 1080 -Think 1.0 -CoolDownSeconds 0 -MaxCalmWaitSeconds 90 -AllowChurnPattern "/ts-order-service-" -Json "D:\train ticket\demo-results.json" -Samples "D:\train ticket\demo-samples.csv"
```
(simple form: `python "D:\train ticket\ts-load.py" http://127.0.0.1:8080 --users 3 --seconds 1080 --think 1.0`; do not exceed 3 users - the node stalls.)

**5.4 While the load runs, 5-6 minutes in SHADOW (the "before"):** show on the Live tab the graph filling in from traces ("nothing is declared"), then the Analysis tab:
* **"Critical path - verified against real requests":** how much of the downstream time the red chain explains, against a random chain; click the slowest-trace link and show the same services on Jaeger's waterfall.
* **Critical shares (CRISP-style):** per service and per request type, which no single chain can show.
* The p95 is about 3 s, order's CPU utilisation about 0.8-0.9, and the decision feed shows what the agent *would* do: "+2 pods, CPU 1.0 - blocked by the replica cap".

**5.5 Switch to ACT** (stop the shadow agent with Ctrl+C or `kill $(lsof -nP -iTCP:8080 -sTCP:LISTEN -t)`, then the same command with `act`):
```bash
ARM=mo_td3_trainticket_cpu RESIZE=cpu SLO_MS=3000 CONTROLLED_SERVICES=ts-order-service NODE_RESERVE_MB=600 \
  bash scripts/19_trainticket_live.sh act
```
**5.6 What you will see (be ready to narrate):**
* Within one step (30 s): the order Deployment is patched - CPU limit 300m -> 1 core, replicas 1 -> 2. In `$K get pods -w` the old pod disappears and two new ones start (`Recreate`).
* **About 50-80 seconds of outage for that service**: some requests fail (2 % of calls in the recorded run). Say it plainly: a CPU change restarts the pod; the simulator assumed a rolling restart with no downtime.
* After both pods are Ready (about 90 s): the gateway p95 falls from about 3.3 s to 0.9 s, throughput rises from about 3 to 7 req/s, order's CPU utilisation drops to about 0.3.
* The memory limit and requests are untouched. Prove it:
```bash
$K get deploy ts-order-service -o jsonpath='{.spec.replicas} {.spec.template.spec.containers[0].resources}{"\n"}'
# 2 {"limits":{"cpu":"1","memory":"800Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}
```
**5.7 Show it was stored** (psql, or the History panel on the Analysis tab):
```sql
select step, ts::time(0), outcome, proposed_dr, applied_dr, round(applied_cpu::numeric,2) cpu, replicas_before rb, replicas_after ra, reason_codes
  from actions where service='ts-order-service' and outcome in ('applied','modified') order by ts desc limit 5;
select ts::time(0), cpu_limit, replicas, ready, round(cpu_util::numeric,2) util, round(p95_ms::numeric) p95 from service_metrics
  where service='ts-order-service' order by ts desc limit 30;                             -- the resize, step by step
select ts::time(0), round(rps::numeric,1) rps, round(p95_ms::numeric) p95, slo_breached from steps order by ts desc limit 30;
select * from violations order by started_at desc limit 3;  select type, ts::time(0) from events order by ts desc limit 5;
```
**5.8 End and restore the cluster** (always):
```bash
kill $(lsof -nP -iTCP:8080 -sTCP:LISTEN -t)                 # stop the agent; its run is closed in the database
bash scripts/21_tt_demo_bottleneck.sh restore               # CPU limit back to 1
$K scale deploy ts-order-service --replicas=1
$K get deploy ts-order-service -o jsonpath='{.spec.replicas} {.status.readyReplicas} {.spec.template.spec.containers[0].resources}{"\n"}'   # = the baseline of part 0
```
**Say out loud what this does not prove:** the agent proposes the maximum for this service at *every* load and never scales down - it is not a load-following controller; only one service was controlled; the test account's order history keeps growing and slows
`order/refresh` and `preserve` by itself, so absolute latencies drift between days.

## Part 6 - history and the dataset (4 min)
Analysis tab -> **History**: cluster selector, p95 against the SLO with violation bands and scale markers, the actions table, critical-path changes. Then:
```bash
cd autoscaler
python -m src.persistence.export --cluster train-ticket --hours 24 --out results/dataset/live_history     # one CSV per table
curl 'localhost:8080/api/history/summary?hours=24'                                                          # only while an agent is running
```
Say: every row carries its cluster, so plugging into another cluster never mixes data; metrics are kept 24 h, actions, violations and critical-path analysis 90 days; a database outage never stops the control loop.

## Part 7 - many services and the honest answers (4 min)
```bash
cd autoscaler && python -m src.evaluation.scale_eval --sizes 8 14 25 50 100 --topologies 3 --episodes 8     # simulation, a few minutes; or show results/scale_results.json
python -m src.evaluation.critical_path_eval                                                                   # 253 real requests: the path explains 80 % of downstream time, a random chain 48 %
```
Decision time is 0.4 ms at 8 services and 0.9 ms at 100; at 25 services the agent has 13-15 % SLO violations against 28 % for the tuned HPA (simulation). Live, the cluster has 47 Deployments; one API call reads all of them per step.
For questions use the table at the end of `docs/INTERVIEW_DEMO.md`; the short truth: on Train-Ticket the agent is on par with a tuned HPA, not better; the value is the critical-path analysis, multi-objective control, the safety layer and the stored evidence.

## If something fails during the demo

| Symptom | Do |
|---|---|
| Preflight `Unauthorized` | new token on the PC (`kubectl create token autoscaler-agent -n default --duration=24h`), `set-credentials` on the Mac |
| Preflight `EOF` / timeout on the API | the API port changed: new port in the tunnel and in `server:` of `~/.kube/tt-agent-config` |
| Dashboard: "Holding - not acting" | a tunnel is down; the agent never acts on missing data. Re-open the tunnel |
| `no entry service` / empty graph | the load is not running (no traces) |
| Agent proposes `+1` but nothing changes | it is in shadow mode, or a guard vetoed it (the feed and History say why) |
| History panel says "history is off" | `AUTOSCALER_DATABASE_URL` not set in that terminal, or `bash scripts/22_db_up.sh` was not run |
| Node stalls / pods restart | stop the load first; the agent only holds |
| Everything is down | `python -m src.dashboard.server --mode replay --replay autoscaler/results/replays/trainticket_run.jsonl` (labelled as a replay), and show `autoscaler/results/dataset/` and `docs/RUN_LOG_TRAINTICKET_CPU.md` |

More commands: `docs/USER_GUIDE.md`. The full record of the live runs: `docs/RUN_LOG_TRAINTICKET_CPU.md`.
