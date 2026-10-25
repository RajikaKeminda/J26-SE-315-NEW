# Run log - a model that scales replicas AND CPU on the real Train-Ticket (2026-10-24)

What was done, in order, with the exact commands, what was stored in the database, and what came out. Everything ran from the repository root on the Mac; the cluster
(Train-Ticket on minikube) is on a separate Windows PC that was reached over SSH tunnels, with a second Claude session on that PC generating the load and watching the node.

## 1. Result in one table

Same load every time (3 users, think 1.0 s); `ts-order-service` is the staged bottleneck (CPU limit lowered to 300m, one pod).

| | before (shadow, 300m, 1 pod) | after (agent acted: 1 core, 2 pods) |
|---|---|---|
| gateway p95 (database) | 3281 ms | 897 ms |
| requests per second at the gateway (database) | 2.9 | 7.3 |
| `ts-order-service` own p95 | 2073 ms | 322 ms |
| `ts-order-service` CPU utilisation | 0.79 | 0.32 |
| load generator, `order/refresh` p50 / p95 | 1073 / 1878 ms | 324 / 571 ms (phase 2, steady state) |
| load generator, `preserve` p50 / p95 | 2534 / 3694 ms | 816 / 1176 ms (phase 2, steady state) |

**The cost of the change (phase 3, the agent acted in the middle of the load):** the pod was recreated, `ts-order-service` was unavailable for at most 80 s, and **125 of 6377 calls (2.0 %)
failed** between 22:09:29 and 22:11:07 on the PC's clock (98 s; the last 19 s were the new JVMs being Ready but cold). Nothing failed before or after.

**Honest reading:**
* The agent did what was wanted - more CPU and a second replica on the saturated service, with the memory untouched - and latency dropped about 3.7x. But it did this **at every step, idle or loaded**: it
  proposed +3 replicas and CPU 1.0 for `ts-order-service` all the time (the guard clamped replicas to the 2-replica cap) and never proposed a scale-down. It acts like "keep the order service at maximum", not
  like a load-following controller. In the simulator the same model also keeps the order service large (mean replica change +1.05 per step).
* In the simulator (2-replica cap, 48 episodes) the new model has the same SLO violations as the tuned HPA (48.2 % vs 48.1 %) at a lower cost (9.7 vs 13.1; score 0.423 vs 0.509), while `mo_td3_large` is worse
  (69.5 % violations) and `mo_td3_trainticket_cap6` equals the HPA (48.6 %). The cheaper cost comes partly from resizing CPU, which the simulator models as a rolling restart without downtime, so it is
  optimistic: on the real cluster every resize is an outage of about a minute.
* The test account's order history keeps growing (5194 -> 5408+ orders) and slows `order/refresh` and `preserve` between runs by itself; the before/after above used the same data volume within minutes of each other, but
  absolute latencies cannot be compared with the October calibration.

## 2. Steps and commands

### 2.1 Setup (once)
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r autoscaler/requirements-dev.txt
bash scripts/22_db_up.sh                                   # PostgreSQL 16 in Docker; prints the URL
export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler
export AUTOSCALER_CLUSTER_NAME=train-ticket
```
Tunnels (Prometheus 9090, Jaeger 16686, the Kubernetes API port that minikube picked: 52783) and `~/.kube/tt-agent-config` with a valid 24 h token for the service account `autoscaler-agent`:
`kubectl --kubeconfig ~/.kube/tt-agent-config auth can-i patch deployments` must print `yes`.

### 2.2 Code prepared for this run (all in the repository, tested)
* `ALLOW_MEMORY_RESIZE=0` - the agent may change CPU but never memory (`config.py`, `rl_agent/env.py::decode_action`, `safety/guards.py`, `k8s_action/k8s_client.py::_set_cpu_only`).
* `src/persistence/training.py`, `schema/002_training.sql`, `train_agent --record` - the training dataset in the database.
* `RESIZE`, `ARM`, `SLO_MS`, `NODE_RESERVE_MB` options for `scripts/19_trainticket_live.sh`; `--resize cpu|all` for `scripts/20_autoscale_any_cluster.sh`.
* `scripts/23_train_trainticket_cpu.sh` - the training run below.

### 2.3 Store the real inputs the simulator is built from
```bash
cd autoscaler
python -m src.persistence.training register --cluster-name train-ticket     # 253 real traces, calibration measurements, the profile -> training_inputs
```

### 2.4 Train (about 16 minutes, offline)
```bash
bash scripts/23_train_trainticket_cpu.sh        # from the repository root
```
which is, after loading `profiles/trainticket.env` (30 s step, 95 s pod start-up, 1-2 replicas, CPU 0.25-1.0) and setting `ALLOW_MEMORY_RESIZE=0` and `SIM_PARAMS_PATH=src/simulation/sim_params_trainticket.json`:
```bash
python -m src.evaluation.train_agent mo_td3_full --episodes 800 --seed 11 --demo-episodes 0 \
  --init-from models/mo_td3_large.pt --sizes 4 14 --target trainticket-sim --target-prob 0.7 --val-sizes 14 \
  --hp actor_lr=5e-5 critic_lr=2e-4 randomize_initial=1 \
  --out models/mo_td3_trainticket_cpu.pt --curve-out results/training/mo_td3_trainticket_cpu.json \
  --record --cluster-name train-ticket --note "replicas + CPU (memory fixed) for the Train-Ticket regime, from mo_td3_large"
```
Output: `models/mo_td3_trainticket_cpu.pt` (run id `711e665a-d44a-43c0-a991-c575f9537087`; best validation 0.300 at episode 150, selected after re-scoring on 48 fresh episodes: 0.321).
Stored in the database: the run's settings and regime, a summary of all 800 simulated episodes, 32 step-by-step trajectories, the 16-point validation curve, the checkpoint's sha256 and the real inputs.

### 2.5 Evaluate in the simulator (same regime, 2-replica cap)
```bash
cd autoscaler
set -a; source ../profiles/trainticket.env; set +a
export ALLOW_MEMORY_RESIZE=0 SIM_PARAMS_PATH=$PWD/src/simulation/sim_params_trainticket.json
python -m src.evaluation.regime_eval --topology trainticket-sim --episodes 48 \
  --agent mo_td3_large=models/mo_td3_large.pt --agent mo_td3_trainticket_cap6=models/mo_td3_trainticket_cap6.pt \
  --agent mo_td3_trainticket_cpu=models/mo_td3_trainticket_cpu.pt --out results/trainticket_sim_cpu_cap2.json
python -m src.persistence.training attach --run 711e665a-d44a-43c0-a991-c575f9537087 --file results/trainticket_sim_cpu_cap2.json
```

### 2.6 Live: phase 1 - baseline, agent in SHADOW (database on)
```bash
AUTOSCALER_DATABASE_URL=... AUTOSCALER_CLUSTER_NAME=train-ticket ARM=mo_td3_trainticket_cpu RESIZE=cpu SLO_MS=3000 \
CONTROLLED_SERVICES=ts-order-service NODE_RESERVE_MB=600 bash scripts/19_trainticket_live.sh          # shadow: decides, changes nothing
```
The PC session ran the load with its watchdog: 540 s, 3 users, think 1.0 (4.6 req/s, 0 errors). Stored: 24 steps, 792 per-service metric rows, 758 action rows, 6 critical-path analyses, 3 topology snapshots.

### 2.7 Live: phase 2 - the same agent in ACT mode
```bash
... same variables ... bash scripts/19_trainticket_live.sh act
```
At its first step it raised `ts-order-service` from 1 to 2 replicas and from 300m to 1 core. Load 1200 s (7.9 req/s, 0 failed calls; the action had happened before the load). **Bug found here:** the memory limit became 768 MiB and the
requests 1 core / 768 MiB (the "locked" memory was rounded to its 64 MiB grid and the resize rewrote requests = limits). Fixed in `safety/guards.py` and `k8s_action/k8s_client.py::_set_cpu_only` with tests.

### 2.8 Live: phase 3 - load first, the agent acts in the middle (verifies the fix)
```bash
bash scripts/21_tt_demo_bottleneck.sh create 300m          # stage the bottleneck: 1 pod, 300m
# agent in shadow (baseline) -> the PC starts the load (75 s warm-up, then 1080 s) -> after 6 minutes: stop the shadow agent, start it with `act`
```
Result: section 1. The memory limit stayed 800 MiB and the requests 50m / 128Mi (checked on the cluster by both sessions). Stored: 15 + 26 steps, 481 + 804 action rows, 495 + 858 metric rows, SLO-violation episodes, events.

### 2.9 Restore the cluster to its baseline
```bash
bash scripts/21_tt_demo_bottleneck.sh restore               # CPU limit back to 1
kubectl --kubeconfig ~/.kube/tt-agent-config scale deploy ts-order-service --replicas=1
kubectl --kubeconfig ~/.kube/tt-agent-config get deploy ts-order-service -o jsonpath='{.spec.replicas} {.spec.template.spec.containers[0].resources}'
# -> 1 {"limits":{"cpu":"1","memory":"800Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}
```
(`restore` only handles the CPU limit; if a run ever changes memory or requests, patch them back by hand as was done once after phase 2.)

### 2.10 Look at and export the data
```bash
docker exec -it autoscaler-postgres psql -U autoscaler -d autoscaler
cd autoscaler
python -m src.persistence.training summary
python -m src.persistence.export --cluster train-ticket --hours 24 --out results/dataset/live_history       # the live history, one CSV per table
python -m src.persistence.training export --run 711e665a-d44a-43c0-a991-c575f9537087 --out results/dataset/training_run
```
Committed: `autoscaler/results/dataset/` (live history of the four agent runs; the training run with all 800 episodes, 32 trajectories, curve and the real inputs),
`autoscaler/results/trainticket_live_cpu_runs.json` (the numbers of this document), `autoscaler/results/trainticket_sim_cpu_cap2.json`, `autoscaler/models/mo_td3_trainticket_cpu.pt`.

## 3. The dataset, for the panel

| Where | What |
|---|---|
| `training_inputs` (7 rows) | the real inputs: 253 Train-Ticket traces, calibration measurements, the Train-Ticket profile, the regime (step, start-up, bounds), the simulator parameters, the call graph with the measured CPU cost per request, the simulator comparison |
| `training_runs` | the run: arm, starting checkpoint, seed, hyper-parameters, regime, git commit, checkpoint sha256, validation scores |
| `training_episodes` (800 rows) | every simulated episode: topology, traffic peak, preference, mean and max p95, violations, cost, replicas, scale actions, penalties |
| `training_trajectories` (32) | every 25th episode step by step and service by service: what the agent saw, did and was scored |
| `training_curve` (16) | the validation curve |
| `steps`, `service_metrics`, `actions`, `critical_paths`, `topology_snapshots`, `violations`, `events` | the live runs on the real cluster, per cluster |

## 4. What this does not show
* A load-following controller: this model keeps `ts-order-service` large at any load. The 3.7x latency gain is the effect of a bigger service, which any controller that raises CPU and replicas would also get.
* A cost saving: scale-down was never proposed. The simulator's lower cost is not seen live.
* Resizing without an outage: on this cluster a CPU change costs about 80 s of unavailability (`Recreate`); with a rolling update and a spare pod it would be near zero but the node's memory does not allow it.
* More than one service: only `ts-order-service` was controlled.
