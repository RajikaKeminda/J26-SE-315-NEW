# User guide - models, training, fine-tuning, demos, and running on a real cluster

Every command here was taken from the scripts and tools in this repository. Run them from the **repository root** unless a block says `cd autoscaler`.
Commands marked *(needs a cluster)* change or read a real Kubernetes cluster. Everything else runs on a laptop.

Contents
1. [Setup](#1-setup)
2. [What the models are](#2-what-the-models-are)
3. [Train a model from scratch](#3-train-a-model-from-scratch)
4. [Evaluate and compare models](#4-evaluate-and-compare-models)
5. [Fine-tune a model](#5-fine-tune-a-model)
6. [Run demos](#6-run-demos)
7. [Connect to a cluster, run, fine-tune, demonstrate](#7-connect-to-a-cluster-run-fine-tune-demonstrate)
8. [Settings you will actually change](#8-settings-you-will-actually-change)
9. [History database](#9-history-database)
10. [Tests](#10-tests)
11. [When something goes wrong](#11-when-something-goes-wrong)
12. [Cheat sheet](#12-cheat-sheet)
13. [Honest limits](#13-honest-limits)

---

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r autoscaler/requirements-dev.txt        # torch, numpy, fastapi, kubernetes, psycopg, pytest ...
make test                                             # the whole test suite (no cluster needed)
```

| Tool | Needed for |
|---|---|
| Python 3.11+ | everything |
| Docker (OrbStack on a Mac) | the local cluster, `docker compose`, the history database |
| kind, kubectl | the local demo cluster (`brew install kind kubectl`) |
| ssh / Tailscale | reaching a cluster on another machine |

Where things are:

| Path | What |
|---|---|
| `autoscaler/src/` | the agent, simulator, safety guard, control room, evaluation tools |
| `autoscaler/models/` | trained checkpoints (`*.pt`) |
| `autoscaler/results/` | evaluation results (JSON), training curves, recorded replays |
| `scripts/` | numbered scripts, run in order for the local cluster; 19-23 are for real clusters, the database and the Train-Ticket training run |
| `profiles/` | settings for a real application (`trainticket.env`; `auto-<ns>.env` is generated) |
| `docs/` | `TRAINTICKET.md`, `INTERVIEW_DEMO.md`, `DEMO_GUIDE.md`, `PERSISTENCE.md`, `TAF_TRACEABILITY.md` |

---

## 2. What the models are

### The agent: MO-TD3

One neural network that decides, for **every service at once**, a replica change (-3..+3), a CPU size and a memory size.

* **Input per service:** 18 numbers (latency, p95, CPU and memory utilisation, offered load, rps and trend, replicas, pending pods, limits,
  CPU cost per request, entry flag, critical-path flags and share, cooldowns). Names are never an input.
* **Two graphs:** the dependency graph (traffic shares) and the latency graph (what each call costs right now), both rebuilt from live data.
* **Network:** a dual graph network (GCN) shared by all services, then a shared policy head. The same weights work for any number of services.
* **MO-TD3:** a twin-critic (TD3) agent with **one value estimate per objective** (SLO, cost, balance). The **preference vector** `[slo, cost, balance]`
  is an input, so the same network can be steered at run time (the sliders in the control room).
* **Safety guard** (not learned, always on): bounds, memory floor, cooldowns, cost budget, circuit breaker (5 SLO breaches in a row hand control to HPA-style rules).

### The arms (what `train_agent` can train)

| Arm | What it is |
|---|---|
| `mo_td3_full` | **the headline agent**: dual graph, hybrid actions (replicas + CPU + memory), three objectives, pure reinforcement learning |
| `mo_td3_sdg_only` | ablation: dependency graph only (no latency graph) |
| `mo_td3_no_topology` | ablation: no graph at all, each service looked at alone |
| `mo_td3_single_obj` | ablation: SLO objective only |
| `mo_td3_replica_only` | replica changes only (CPU and memory actions switched off) - what a node with no spare memory needs |
| `mo_td3_categorical` | ablation: replica change as a discrete choice instead of a rounded number |
| `mo_td3_demo`, `mo_td3_demo_*` | the same models seeded with HPA demonstrations (the study found this makes them worse) |

### The checkpoints in `autoscaler/models/`

| File | Use it for |
|---|---|
| `mo_td3_full.pt` | default; trained on 4-12 random services; the 6-service demo app |
| **`mo_td3_large.pt`** | **general clusters up to about 64 services**: `mo_td3_full` fine-tuned on random 4-64 service topologies |
| `mo_td3_trainticket_cap6.pt` | Train-Ticket's regime: replica-only, Java pod start-up (about 95 s), up to 6 replicas. **This was the model used live** |
| `mo_td3_trainticket.pt` | the same, trained for the node's 2-replica cap |
| `mo_td3_replica_only.pt` | replica-only starting point |
| `mo_td3_ft_control.pt`, `mo_td3_ft_crisp.pt` | experiment artifacts (critical-set inputs experiment). Do not deploy |
| the rest (`*_sdg_only`, `*_no_topology`, `*_single_obj`, `*_categorical`, `*_demo*`, `seeds/`) | ablation study |

The control room, scripts and `--arm` take a checkpoint **name without `.pt`**, e.g. `--arm mo_td3_large`.
With no `--arm`, `models/shared_policy.pt` is used if present, otherwise `mo_td3_full`.

### The baselines it is compared against

`static` (never scale), `threshold` rules, `hpa` (Kubernetes-style, default 60 % target), `hpa_tuned` (the same HPA, grid-searched on held-out traffic), `vpa`
(a re-implementation of the recommender and updater). On a real cluster `k8s_hpa` is the real Kubernetes HPA controller.

---

## 3. Train a model from scratch

Training happens in the **simulator**, which is fitted to measurements of a real cluster (`src/simulation/sim_params.json`). No cluster is needed.

### One arm

```bash
cd autoscaler
python -m src.evaluation.train_agent mo_td3_full --episodes 1000 --seed 0
```

Outputs `models/mo_td3_full.pt` (the best-validation checkpoint) and the learning curve `results/training/mo_td3_full_s0.json`.
A seed other than 0 writes to `models/seeds/<arm>_s<seed>.pt`.

Useful options:

| Option | Meaning |
|---|---|
| `--episodes N` | training episodes (default 1000) |
| `--seed S` | random seed |
| `--demo-episodes N` | episodes of HPA demonstrations used before RL (default 400; `0` for pure RL) |
| `--updates-per-episode N` | gradient updates per episode (default 30) |
| `--out FILE`, `--curve-out FILE` | where the checkpoint and curve go |
| `--hp KEY=VALUE ...` | hyper-parameters: `actor_lr`, `critic_lr`, `n_step`, `explore_noise`, `val_episodes`, `randomize_initial`, ... (list: the `HParams` class in `src/rl_agent/trainer.py`) |
| `--sizes MIN MAX` | number of services in the random training topologies (default 4 12) |
| `--init-from FILE` | warm start: this is fine-tuning (section 5) |

A 2-minute smoke run to check that everything works:

```bash
python -m src.evaluation.train_agent mo_td3_full --episodes 6 --demo-episodes 4 --updates-per-episode 3 --out /tmp/smoke.pt --hp n_step=2 val_episodes=2
```

### Every arm, several seeds

```bash
bash scripts/06_train_all_arms.sh                                     # 1 seed, 1000 episodes, all arms
SEEDS="0 1 2" JOBS=4 bash scripts/06_train_all_arms.sh                # the multi-seed study, 4 in parallel
ARMS="mo_td3_full" EPISODES=1500 bash scripts/06_train_all_arms.sh    # just the headline agent
```

Each training process holds 150-300 MB of replay buffer; on an 8 GB laptop `JOBS=4` is the practical ceiling.

### Everything at once (reproduce all results)

```bash
JOBS=4 bash scripts/17_reproduce_results.sh                           # about 2-3 h on a laptop (the script's own estimate)
EPISODES=600 SEEDS_FULL="0" bash scripts/17_reproduce_results.sh      # a quick smoke version
```

It trains the arms, evaluates every controller, traces the Pareto fronts, redraws the figures and regenerates the README tables.

### Re-fit the simulator to your cluster *(needs a cluster)*

The simulator's service model (capacity, queueing, tail latency) comes from load tests of the demo service. To re-measure it on your cluster:

```bash
make kind-up docker-build kind-load deploy            # the local cluster with the demo app (section 6B)
bash scripts/13_calibrate.sh                          # about 10 min; writes calibration/ and src/simulation/sim_params.json
```

---

## 4. Evaluate and compare models

```bash
bash scripts/09_run_ablation.sh [episodes]            # every controller on identical traffic, 6-service app and the unseen 14-service app -> results/ablation_results.json
bash scripts/12_run_pareto.sh [--episodes 32]         # one network swept over 15 preferences vs the HPA tuning sweep -> results/pareto_results.json
cd autoscaler
python -m src.evaluation.scale_eval --sizes 8 14 25 50 100 --topologies 3 --episodes 8                              # many-service benchmark
python -m src.evaluation.scale_eval --ckpt models/mo_td3_large.pt --out results/scale_results_size_trained.json     # a specific checkpoint
python -m src.evaluation.plots && python -m src.evaluation.report                                                   # figures and README tables
python -m src.evaluation.report --check                                                                             # fails if the README disagrees with results/*.json
```

What each shows: the ablation table (SLO violation rate, p95, cost, adaptation steps, score, lower is better), the Pareto fronts (hypervolume, higher is
better), and the scale table (violations and decision time from 8 to 100 services).

Checking the critical path against real requests (offline, 253 recorded Train-Ticket requests):

```bash
cd autoscaler && python -m src.evaluation.critical_path_eval        # coverage and top-1 hit of the agent's path, a random chain, the deepest chain, plus a held-out comparison
```

---

## 5. Fine-tune a model

Fine-tuning **extends an existing checkpoint**: `--init-from` loads its weights (actor, both critics and their targets) and training continues in the
simulator under the settings you choose. The architecture does not change, so the result is a drop-in `--arm`. Use a small learning rate so it adapts instead of forgetting.

`mo_td3_large` and `mo_td3_trainticket_cap6` were made exactly this way.

### Recipe A - more services (this made `mo_td3_large`)

```bash
cd autoscaler
python -m src.evaluation.train_agent mo_td3_full --episodes 500 --seed 7 --demo-episodes 0 \
  --init-from models/mo_td3_full.pt --sizes 4 64 --val-sizes 25 50 \
  --hp actor_lr=5e-5 critic_lr=2e-4 --out models/mo_td3_large.pt --curve-out results/training/mo_td3_large.json
```

### Recipe B - one specific application and its timing (this made `mo_td3_trainticket_cap6`)

```bash
cd autoscaler
set -a; source ../profiles/trainticket.env; set +a                     # the simulator now runs with the cluster's step length, pod start-up, bounds
export SIM_PARAMS_PATH=$PWD/src/simulation/sim_params_trainticket.json # memory, idle CPU, latency floor measured on the real cluster
python -m src.evaluation.train_agent mo_td3_replica_only --episodes 800 --seed 9 --demo-episodes 0 \
  --init-from models/mo_td3_replica_only.pt --sizes 4 14 --target trainticket-sim --target-prob 0.7 --val-sizes 14 \
  --hp actor_lr=5e-5 critic_lr=2e-4 randomize_initial=1 \
  --out models/mo_td3_trainticket_cap6.pt --curve-out results/training/mo_td3_trainticket_cap6.json
```

* `--target NAME` mixes a named topology (from `src/topologies.py`) into training and validation; `--target-prob` is the share of episodes on it. The rest stay random, so the model does not forget.
* `randomize_initial=1` starts each service at its own CPU limit so the agent does not learn only from one starting point.
* Replica cap: set `MAX_REPLICAS` before training (the profile does) so the simulator allows what the cluster allows.

### Recipe D - replicas AND CPU for Train-Ticket, with the whole run stored in the database (this made `mo_td3_trainticket_cpu`)

```bash
bash scripts/22_db_up.sh                                         # once
export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler
bash scripts/23_train_trainticket_cpu.sh                         # about 16 minutes: 800 episodes from mo_td3_large, regime of profiles/trainticket.env
python -m src.persistence.training summary                       # the stored run, with counts (run from autoscaler/)
```

The script sets `ALLOW_MEMORY_RESIZE=0` (the agent learns replicas and CPU, never memory), starts from `models/mo_td3_large.pt`, mixes the measured Train-Ticket topology into
70 % of the episodes, and stores the run (`--record`, section 9). Compare it afterwards with `python -m src.evaluation.regime_eval` as in Recipe C step 5.

### Recipe E - bottleneck-aware (the agent must find WHERE the load is)

```bash
bash scripts/24_train_bottleneck_aware.sh                                          # demo regime, from mo_td3_full -> models/mo_td3_bn.pt (about 30 min)
REGIME=trainticket FROM=models/mo_td3_bn.pt OUT=mo_td3_bn_tt SCALE_UP_MIN_UTIL=0.5 SCALE_DOWN_MAX_UTIL=0.5 bash scripts/24_train_bottleneck_aware.sh
cd autoscaler && SCALE_UP_MIN_UTIL=0.5 SCALE_DOWN_MAX_UTIL=0.5 ALLOW_MEMORY_RESIZE=0 python -m src.evaluation.bottleneck_eval --topology demo6 --agent bn=models/mo_td3_bn.pt --agent full=models/mo_td3_full.pt
```
In `hot_prob` (default 0.7 in the script) of the episodes one random service is made the only bottleneck (`topologies.bottleneck_scenario`); validation uses a scenario per service (`--hot-val`). Results: `docs/BOTTLENECK_RESULTS.md`.

### Recipe C - for your own cluster

1. **Run the agent in shadow mode with load** (section 7) to produce `profiles/auto-<ns>.env`: pod start-up time, step length, bounds, replica cap, budget.
2. **Record the shape of the cluster**: save a window of real traces and note the learned call graph and CPU cost per request. `src/topologies.py::TRAINTICKET_SIM` is the model to copy:
   a `Topology(...)` with the services, the call graph with traffic shares, and `work_ms` per service. Add it to the `TOPOLOGIES` dict. **This step is manual.**
3. **Optional**: copy `src/simulation/sim_params.json` to `sim_params_<name>.json` and change memory, idle CPU and the latency floor to what you measured; point `SIM_PARAMS_PATH` at it.
4. **Train** (as Recipe B, with your topology name and `--init-from models/mo_td3_large.pt`, arm `mo_td3_full`):
   ```bash
   cd autoscaler
   set -a; source ../profiles/auto-<ns>.env; set +a
   python -m src.evaluation.train_agent mo_td3_full --episodes 500 --demo-episodes 0 \
     --init-from models/mo_td3_large.pt --sizes 4 30 --target mycluster-sim --target-prob 0.5 --val-sizes 25 \
     --hp actor_lr=5e-5 critic_lr=2e-4 randomize_initial=1 --out models/mo_td3_mycluster.pt
   ```
   Warm-starting a *different* arm (for example `mo_td3_replica_only` from `mo_td3_large`) has not been tried here.
5. **Compare against a tuned HPA in the same regime**:
   ```bash
   python -m src.evaluation.regime_eval --topology mycluster-sim \
     --agent original=models/mo_td3_large.pt --agent tuned=models/mo_td3_mycluster.pt
   python -m src.evaluation.pareto_regime --topology mycluster-sim --ckpt models/mo_td3_mycluster.pt
   ```
   Keep the fine-tuned model only if it clearly beats the original. Fine-tuning can make a model worse on the general benchmark (both experiment checkpoints did).
6. **Deploy it** with `--arm mo_td3_mycluster` (section 7), in shadow mode first.

Training uses one seed by default; differences of about 0.03 in score are inside seed noise.

---

## 6. Run demos

### 6A. No cluster at all (2 minutes) - the simulator control room

```bash
bash scripts/10_run_dashboard.sh                           # http://localhost:8080
bash scripts/10_run_dashboard.sh demo 8080 --scenario diurnal
```

The page has two tabs, **Live** and **Analysis**. Things to do:

* Drag the **SLO** slider to 100 and the others to 0: the same network moves to more pods and lower latency. Drag **cost** up and it saves money.
* Switch the controller (MO-TD3, HPA, VPA, threshold, static) on the burst scenario and watch the baselines' p95 climb.
* Scenarios: burst, diurnal, rampup, normal. The burst starts after about 2 minutes of demo time.
* The Analysis tab has the controller comparison, the cost-versus-latency Pareto, the simulator calibration and the training curves.

### 6B. The local Kubernetes cluster (10-15 minutes) *(needs Docker)*

```bash
make kind-up                         # a kind cluster
make docker-build kind-load          # the service image and the autoscaler image
make deploy                          # namespace, Prometheus, Jaeger, 6 services, the in-cluster autoscaler
make e2e                             # self-checking smoke test: traffic -> metrics and traces -> the agent changes pods
```

Then, in separate terminals:

```bash
bash scripts/05_port_forward.sh                 # dashboard :8080, Prometheus :9090, Jaeger :16686, gateway :8000 (leave running)
bash scripts/18_demo_traffic.sh burst           # 12 rps -> 40 rps -> 12 rps, about 7 minutes, generated inside the cluster
kubectl get pods -n autoscaler-poc -w           # pods appearing and disappearing
```

Other loads: `bash scripts/18_demo_traffic.sh diurnal`, `bash scripts/18_demo_traffic.sh steady 12 300`. Run the agent from your laptop instead of in the cluster:

```bash
WITH_AUTOSCALER=0 bash scripts/03_deploy_all.sh
bash scripts/05_port_forward.sh                                 # separate terminal
bash scripts/08_run_autoscaler.sh --shadow --iterations 20      # observe only
bash scripts/08_run_autoscaler.sh --iterations 60               # act for real
bash scripts/08_run_autoscaler.sh --controller hpa              # run a baseline instead
```

Real head-to-head on the cluster (about 8 minutes per run):

```bash
CONTROLLERS="static hpa_tuned mo_td3" PATTERNS="burst diurnal" bash scripts/15_live_experiment.sh
bash scripts/16_install_metrics_server.sh                       # only for the real-Kubernetes-HPA arm (k8s_hpa)
```

Expected on a burst: the 3x spike pushes p95 well above the SLO for a moment, the circuit breaker opens after 5 breached steps and the HPA rules take over until it recovers.
That is the safety layer working; `docs/DEMO_GUIDE.md` has a talking script and likely questions. Remove everything with `make kind-down`.

### 6C. Docker only

```bash
docker compose up --build            # 6 services + Prometheus + Jaeger + the control room (demo mode)
docker compose down
```

### 6D. The real Train-Ticket (the interview demo)

Full script with timings: `docs/INTERVIEW_DEMO.md`. The short form, with the tunnels and the kubeconfig from section 7:

```bash
bash scripts/19_trainticket_live.sh                                       # preflight, then SHADOW mode; open http://localhost:8080
bash scripts/20_autoscale_any_cluster.sh --namespace default --kubeconfig ~/.kube/tt-agent-config --discover-only    # what it finds on its own
```

Demo steps:

1. **Plug into any cluster:** the `--discover-only` run prints the entry service (the root of most traces), the controllable services, the bounds from the real limits, the replica cap from node headroom, and the pod start-up measured from real pods.
2. **How the critical path is found:** open the control room next to Jaeger (http://localhost:16686). The graph starts with one node and fills in as traces arrive. In the Analysis tab, "Critical path - verified against real requests" shows how much of a request's downstream time the path explains, against a random chain, and links to the slowest real trace.
3. **How it scales (stage a bottleneck first, then undo it):**
   ```bash
   bash scripts/21_tt_demo_bottleneck.sh create 300m            # ts-order-service: 1 core -> 0.3 core (rolling restart; wait for the pod to show 1/1)
   bash scripts/20_autoscale_any_cluster.sh --namespace default --kubeconfig ~/.kube/tt-agent-config \
        --controlled ts-order-service --slo 1400 --arm mo_td3_trainticket_cap6 --reserve 600 --act
   kubectl --kubeconfig ~/.kube/tt-agent-config get pods -w     # second terminal
   bash scripts/21_tt_demo_bottleneck.sh restore                # afterwards
   ```
   Start the load on the cluster's PC first. Without traffic there are no traces and nothing to scale.
4. **Scale:** `python -m src.evaluation.scale_eval` (simulation, 8 to 100 services).
5. **History:** the Analysis tab's History panel (section 9).

### 6E. If the network fails - replay

```bash
# record a good run beforehand, while the control room runs:
cd autoscaler && python -m src.dashboard.record --out results/replays/trainticket_run.jsonl --seconds 600
# later, with no cluster:
python -m src.dashboard.server --mode replay --replay results/replays/trainticket_run.jsonl     # labelled as a replay on the page
python -m src.dashboard.server --mode replay --replay-db <cluster-name> --replay-hours 6          # from the history database (needs AUTOSCALER_STORE_FRAMES=1 when recorded)
```

A recorded 29-step run is in the repository: `autoscaler/results/replays/trainticket_run.jsonl`.

---

## 7. Connect to a cluster, run, fine-tune, demonstrate

### 7.1 What the cluster must have

| Needed | Why |
|---|---|
| Kubernetes API access for a service account that can read Deployments and pods and patch Deployments and their scale | state and actions |
| **Prometheus** scraping **cAdvisor** and **kube-state-metrics** | CPU, memory, pod counts |
| **OpenTelemetry traces** in **Jaeger** | request rate, latency, the call graph, the critical path |

Services that emit no traces are observed (CPU, memory) but have no call edges and cannot be controlled. Istio, Linkerd, Zipkin and Tempo are not supported.
Trace sampling should be high: a low rate hides rarely-called branches (`OTEL_SAMPLE_RATE`, default 1.0).

### 7.2 Reach it

If the cluster is on the same network, use its Prometheus and Jaeger URLs directly. If it is on another machine (as with Train-Ticket on a Windows PC), tunnel:

```bash
ssh -fN -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
  -L 9090:127.0.0.1:9090 -L 16686:127.0.0.1:16686 -L <api-port>:127.0.0.1:<api-port> "<user>"@<tailscale-ip>
```

The Kubernetes API port of minikube **changes every time it restarts**: on the cluster machine run `kubectl config view --minify`, put the same port in the tunnel and in `server:` of the kubeconfig.
Stop a tunnel with `kill <pid>`; never `pkill -f` a pattern, it kills the other forwards. Sync the cluster machine's clock (`w32tm /resync` on Windows): it was 28 s ahead once.

### 7.3 A limited identity for the agent (not cluster-admin)

On the cluster, create a service account that may only do what the agent needs, and a 24-hour token for it:

```bash
kubectl create serviceaccount autoscaler-agent -n default
kubectl create role autoscaler-agent -n default --verb=get,list,watch,patch,update --resource=deployments,deployments/scale
kubectl create role autoscaler-pods -n default --verb=get,list --resource=pods
kubectl create rolebinding autoscaler-agent -n default --role=autoscaler-agent --serviceaccount=default:autoscaler-agent
kubectl create rolebinding autoscaler-pods  -n default --role=autoscaler-pods  --serviceaccount=default:autoscaler-agent
kubectl create token autoscaler-agent -n default --duration=24h
```

Put the token in a kubeconfig on the machine that runs the agent:

```bash
K=~/.kube/tt-agent-config
kubectl --kubeconfig $K config set-cluster tt --server=https://127.0.0.1:<api-port> --certificate-authority=<path-to-cluster-ca.crt> --embed-certs=true
kubectl --kubeconfig $K config set-credentials autoscaler-agent --token=<token>
kubectl --kubeconfig $K config set-context tt --cluster=tt --user=autoscaler-agent --namespace=default
kubectl --kubeconfig $K config use-context tt
kubectl --kubeconfig $K auth can-i patch deployments            # must print: yes
```

The role commands are a sketch of what `docs/TRAINTICKET.md` describes; they were not run in exactly this form here, so check `auth can-i` afterwards.
Tokens last 24 hours: when the preflight says `Unauthorized`, create a new one and update it with
`kubectl --kubeconfig $K config set-credentials autoscaler-agent --token=<new>`. (This role setup follows `docs/TRAINTICKET.md`; adjust names to your cluster.)

Optional: let the agent read the `kube-system` namespace UID so the stored history of two clusters can never be mixed even if they get the same name:
`kubectl apply -f k8s/03-autoscaler/rbac-cluster-identity.yaml` (edit the subject to your service account).

### 7.4 Inspect it, then run in shadow mode

```bash
bash scripts/20_autoscale_any_cluster.sh --namespace <ns> --prometheus http://localhost:9090 --jaeger http://localhost:16686 \
  --kubeconfig <your-kubeconfig> --discover-only                  # prints what it found and starts nothing
bash scripts/20_autoscale_any_cluster.sh --namespace <ns> --kubeconfig <your-kubeconfig> \
  --arm mo_td3_large --slo <your p95 target in ms>                # SHADOW: observe and decide, change nothing; http://localhost:8080
```

`--kubeconfig` is required if it is not your default one: the launcher does **not** read `~/.kube/tt-agent-config` on its own (the Train-Ticket script `19_...` does).
Generate load on the cluster first, or there are no traces and the agent holds.

| Option of `scripts/20_autoscale_any_cluster.sh` | Meaning |
|---|---|
| `--namespace`, `--prometheus`, `--jaeger`, `--kubeconfig`, `--port` | where to look, and the control-room port (8080) |
| `--entry SERVICE` | entry service, when the traces do not reveal it |
| `--controlled a,b` | the only services the agent may change (others are observed) |
| `--slo MS` | p95 target; without it the default is a depth-based formula, not a measured target |
| `--arm NAME` | checkpoint in `autoscaler/models` |
| `--reserve MIB` | memory the node must keep free after a scale-up |
| `--act` | really scale (replicas only by default); without it nothing is changed |
| `--discover-only` | print what it found, start nothing |
| `--database-url URL`, `--cluster-name NAME` | store the history (section 9) |

### 7.5 Act, carefully

```bash
bash scripts/20_autoscale_any_cluster.sh --namespace <ns> --kubeconfig <your-kubeconfig> \
  --controlled svc-a,svc-b --slo <ms> --arm mo_td3_large --act
```

Start with two or three services, watch `kubectl get pods -w` and the decision feed, then widen `--controlled`. Replica-only is the default (`ALLOW_RESIZE=0` in the generated profile);
a resize is a rolling restart. The guard also refuses a scale-up that would leave the node with too little memory, and holds (does nothing) when the data is stale or too many CPU series are missing.

For Train-Ticket specifically: `bash scripts/19_trainticket_live.sh` (shadow) and `bash scripts/19_trainticket_live.sh act`, using `profiles/trainticket.env`. Options are environment variables:

```bash
AUTOSCALER_DATABASE_URL=... AUTOSCALER_CLUSTER_NAME=train-ticket \      # store the history
ARM=mo_td3_trainticket_cpu RESIZE=cpu SLO_MS=3000 \                       # model, allow CPU changes (memory fixed), p95 target in ms
CONTROLLED_SERVICES=ts-order-service NODE_RESERVE_MB=600 \
bash scripts/19_trainticket_live.sh act
```

`RESIZE=cpu` lets the agent change CPU limits only (memory, request and limit, is left exactly as it is); `RESIZE=all` also memory; without it, replicas only.
`scripts/20_autoscale_any_cluster.sh` has the same as `--resize cpu|all`. **A resize restarts the pod** (the Deployment of `ts-order-service` uses `Recreate`, so the service is
unavailable until the new pod is Ready).

### 7.6 Which model for which cluster

| Cluster | `--arm` |
|---|---|
| a general cluster, up to about 64 services, pods that start in seconds | `mo_td3_large` |
| a small application (about 4-12 services) | `mo_td3_full` |
| Java services that need 1-4 minutes to start, replica-only | fine-tune for it (Recipe B or C), or `mo_td3_trainticket_cap6` if it resembles Train-Ticket |

Check first in shadow mode whether its decisions are sensible; if the pods are slow to start, or the response to load differs a lot from the demo service, fine-tune (section 5).

### 7.7 Then demonstrate it

Follow section 6D (the same flow works on any cluster): `--discover-only`, then the critical-path proof, then a staged or real scale event, then the History panel.
To record a fallback run, section 6E.

---

## 8. Settings you will actually change

Environment variables (`autoscaler/src/config.py`); the launcher scripts set most of them from `profiles/`.

| Variable | Default | Meaning |
|---|---|---|
| `METRICS_SOURCE` | `prometheus` | `prometheus` = the demo app's own metrics; `otel` / `trainticket` = cAdvisor + traces (any real application) |
| `AUTOSCALER_NAMESPACE` | `autoscaler-poc` | namespace to control |
| `PROMETHEUS_URL`, `JAEGER_QUERY_URL` | `localhost:9090`, `localhost:16686` | endpoints (over tunnels, raise `PROMETHEUS_TIMEOUT_SECONDS`) |
| `AUTOSCALER_ENTRY_SERVICE` | none | entry service; the graph is learned from traces |
| `CONTROLLED_SERVICES` | all | whitelist of services the agent may change |
| `STEP_SECONDS`, `POLL_INTERVAL_SECONDS` | 10 | control step |
| `POD_STARTUP_SECONDS`, `ROLLING_SECONDS_PER_REPLICA` | 4, 4.4 | how long a new pod takes (95 for Java) |
| `METRIC_WINDOW` | `10s` | must be at least 4x the Prometheus scrape interval (`120s` for a 30 s scrape) or CPU series drop |
| `MIN_REPLICAS`, `MAX_REPLICAS` | 1, 8 | replica bounds |
| `CPU_MIN/MAX`, `MEM_MIN_MB/MAX_MB` | 0.1-1.0 cores, 96-512 MiB | resource bounds (derived from your limits by `scripts/20`) |
| `SLO_LATENCY_MS` | formula | absolute p95 target |
| `BUDGET_PER_SERVICE`, `CLUSTER_BUDGET_CORES` | 1.0, none | cost budget; with many observed services leave `CLUSTER_BUDGET_CORES` unset |
| `ALLOW_RESIZE` | `1` (`0` in generated profiles) | allow CPU/memory changes (pod restarts) |
| `SCALE_UP_MIN_UTIL`, `SCALE_DOWN_MAX_UTIL` | `0`, `1.01` (off; `0.5`, `0.5` in the Train-Ticket profile) | demand gate: capacity is added only to a service whose CPU utilisation is at least the first, removed only from one at most the second |
| `ALLOW_CPU_DECREASE`, `MAX_RESIZES_PER_STEP` | `1`, `0` (no limit; `0` and `1` in the Train-Ticket profile) | never shrink a CPU limit; at most this many services resized per step (the busiest first) |
| `SCALE_UP_MIN_ENTRY_RPS` | `0` (off; `0.5` in the Train-Ticket profile) | no scale-up anywhere while the entry service gets fewer requests per second than this: a starting JVM burns its whole CPU limit and looks like demand |
| `IDLE_RECLAIM_STEPS`, `IDLE_RECLAIM_UTIL` | `0` (off; `6` in the Train-Ticket profile), `0.15` | a service idle (CPU below the second) for that many steps in a row gives one replica back (nothing restarts), once the cooldown and the 5-minute window after a scale-up have passed |
| `ALLOW_MEMORY_RESIZE` | `1` | `0` = CPU may change but memory never does (in training and live) |
| `NODE_MEMORY_RESERVE_MB` | 768 | memory the node must keep free after a scale-up |
| `SCALE_DOWN_AFTER_UP_STEPS` | 0 | stabilisation window after a scale-up (10 in the Train-Ticket profile) |
| `OBSERVE_UNTRACED` | 0 | also show running services that emit no traces |
| `CRITICAL_PATH_SOURCE` | `dfs` | `crisp` feeds the measured critical set to the agent; only for a model trained with it |
| `AUTOSCALER_SHADOW` | 0 | same as `--shadow` |
| `AUTOSCALER_DATABASE_URL`, `AUTOSCALER_CLUSTER_NAME` | empty | history database (section 9) |

Server options: `python -m src.dashboard.server --mode demo|live|replay --port 8080 --shadow --arm NAME --controller mo_td3|hpa|vpa|threshold|static --preference SLO COST BALANCE --scenario burst --interval SECONDS`.

---

## 9. History database

Opt-in PostgreSQL storage of the agent's actions, per-service metrics (24 h), critical-path analysis, SLO-violation episodes and events (90 days), kept apart per cluster.
The details are in `docs/PERSISTENCE.md`.

```bash
bash scripts/22_db_up.sh                                   # PostgreSQL 16 in Docker; prints the URL
export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler
export AUTOSCALER_CLUSTER_NAME=train-ticket-pc             # what you call this cluster; give every cluster its own
bash scripts/19_trainticket_live.sh                        # or scripts/20_... --database-url ... --cluster-name ...
```

* **See it:** the History panel in the Analysis tab, `curl 'localhost:8080/api/history/summary?hours=24'`, or `docker exec -it autoscaler-postgres psql -U autoscaler -d autoscaler`.
* **Export for research:** `cd autoscaler && python -m src.persistence.export --cluster train-ticket-pc --out results/export --hours 72` (CSV per table; `--table actions --format jsonl --out -` for one table to the screen).
* **Replay from it:** `AUTOSCALER_STORE_FRAMES=1` while running, then `--mode replay --replay-db train-ticket-pc`.
* **The training dataset:** `python -m src.evaluation.train_agent ... --record` (or `scripts/23_train_trainticket_cpu.sh`) stores, per training run, the settings, a summary of every simulated episode,
  step-by-step trajectories of a sample, the validation curve, the checkpoint's sha256, and the real inputs the simulator is built from (the 253 real traces, the calibration measurements, the simulator
  parameters, the call graph, the regime). From `autoscaler/`:
  ```bash
  python -m src.persistence.training register --cluster-name train-ticket     # the real traces, calibration and profile -> training_inputs (once)
  python -m src.persistence.training summary                                   # every stored input and training run, with counts
  python -m src.persistence.training attach --run <run-id> --file results/<evaluation>.json     # keep an evaluation with its run
  python -m src.persistence.training export --run <run-id> --out results/dataset                  # CSV/JSON files of one run for the panel
  ```
* **Stop it:** `docker compose -f docker-compose.db.yml down` (data stays); add `-v` to delete the data.

---

## 10. Tests

```bash
make test                                                  # or: bash scripts/11_run_tests.sh [-k pattern]
cd autoscaler && python -m pyflakes src tests              # lint (CI runs it)
bash scripts/22_db_up.sh
cd autoscaler && TEST_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler python -m pytest -q    # includes the 12 database tests
```

Without `TEST_DATABASE_URL` the database tests are skipped. `make e2e` is the cluster smoke test. CI (`.github/workflows/ci.yml`) runs lint, tests with a PostgreSQL service, the README-sync check, the image builds and the kind end-to-end.

---

## 11. When something goes wrong

| What you see | Cause and fix |
|---|---|
| Preflight: `Unauthorized` / "token expired" | the 24 h token ran out: `kubectl create token autoscaler-agent -n default --duration=24h` on the cluster, update the kubeconfig |
| Preflight: `EOF` / "context deadline exceeded" on the Kubernetes API | the minikube API port changed or the tunnel for it is down: `kubectl config view --minify` on the cluster machine, re-open the tunnel, fix `server:` in the kubeconfig |
| Dashboard: "Holding - not acting: cluster read failed ... timed out" | a tunnel is down or slow; the agent never acts on missing data. `bash scripts/19_trainticket_live.sh` shows which |
| `no entry service found` | no recent traces: start the load generator, or pass `--entry SERVICE` |
| The agent proposes `+1 pod` but nothing changes | it is in **shadow** mode (nothing is sent) - add `--act`; or a guard vetoed it (the decision feed and the History table give the reason: node memory, cooldown, replica cap, budget) |
| Nothing is proposed at all | no load, or utilisation is low; for a visible scale-up stage a bottleneck (`scripts/21_tt_demo_bottleneck.sh`) |
| The graph is empty or stuck | no traces: check the load and the "newest trace is N s old" line of the preflight |
| `scripts/21 ... create` says "already created" | a stale state file: `status` shows it; an empty `~/.tt_demo_state` is now ignored. `restore` puts the saved limit back |
| Pod stays `0/1` for minutes after a CPU-limit change | normal for Java under 0.3 core (3-5 minutes); wait for `1/1` before judging |
| Scale-up refused: node memory | the node has too little free memory (SkyWalking held 851 MiB in Train-Ticket); free memory or lower `--reserve` |
| Node stalls (pods restarting) under load | too much load for the node (more than 3 users in the Train-Ticket load); stop the generator first, the agent only holds |
| CPU series missing / "holding" with a short metric window | `METRIC_WINDOW` below 4x the scrape interval: use `120s` for a 30 s scrape |
| History panel: "history is off" | `AUTOSCALER_DATABASE_URL` is not set or the database is down: `bash scripts/22_db_up.sh` |
| History: "name already used by another cluster" | that cluster name was stored for a different `kube-system` UID: choose another `AUTOSCALER_CLUSTER_NAME` |
| Everything is down | `python -m src.dashboard.server --mode replay --replay autoscaler/results/replays/trainticket_run.jsonl` |

---

## 12. Cheat sheet

```bash
# --- setup and tests
python3 -m venv .venv && source .venv/bin/activate && pip install -r autoscaler/requirements-dev.txt
make test

# --- models
cd autoscaler && python -m src.evaluation.train_agent mo_td3_full --episodes 1000 --seed 0
SEEDS="0 1 2" JOBS=4 bash scripts/06_train_all_arms.sh
python -m src.evaluation.train_agent mo_td3_full --init-from models/mo_td3_large.pt --episodes 500 --demo-episodes 0 \
  --target <topology> --target-prob 0.5 --hp actor_lr=5e-5 critic_lr=2e-4 randomize_initial=1 --out models/mo_td3_mine.pt

# --- evaluation
bash scripts/09_run_ablation.sh && bash scripts/12_run_pareto.sh
cd autoscaler && python -m src.evaluation.scale_eval && python -m src.evaluation.critical_path_eval && python -m src.evaluation.report --check

# --- demos
bash scripts/10_run_dashboard.sh                                  # simulator, no cluster
make kind-up docker-build kind-load deploy && bash scripts/05_port_forward.sh && bash scripts/18_demo_traffic.sh burst
docker compose up --build

# --- a real cluster
bash scripts/20_autoscale_any_cluster.sh --namespace <ns> --kubeconfig <cfg> --discover-only
bash scripts/20_autoscale_any_cluster.sh --namespace <ns> --kubeconfig <cfg> --arm mo_td3_large --slo <ms>                       # shadow
bash scripts/20_autoscale_any_cluster.sh --namespace <ns> --kubeconfig <cfg> --arm mo_td3_large --slo <ms> --controlled a,b --act # act
bash scripts/19_trainticket_live.sh [act]                         # Train-Ticket
bash scripts/21_tt_demo_bottleneck.sh create 300m | restore | status

# --- history and fallback
bash scripts/22_db_up.sh
cd autoscaler && python -m src.persistence.export --cluster <name> --out results/export
python -m src.dashboard.record --out results/replays/run.jsonl --seconds 600
python -m src.dashboard.server --mode replay --replay results/replays/run.jsonl
```

---

## 13. Honest limits

* On Train-Ticket the agent is **on par with a tuned HPA**, not better. With room to scale it has fewer SLO violations (7 % against 11 %) at a 14 % higher cost; with the node's 2-replica cap every controller is capacity-limited. Its value is the critical-path analysis, the multi-objective control and the safety layer.
* The only live scaling shown is a **staged** bottleneck (a lowered CPU limit) on one service.
* Large-cluster numbers (25, 50, 100 services) are **simulation** results. Live, the biggest cluster run had 47 Deployments, 33 running, 11 with traces.
* The simulator's queueing terms come from the demo service; only memory, idle CPU, CPU per request and topology were measured for Train-Ticket.
* Feeding the measured critical set to the agent did not measurably improve its scaling (`docs/TRAINTICKET.md`), so `CRITICAL_PATH_SOURCE` stays `dfs`.
* Turning your cluster into a fine-tuning target (section 5, Recipe C steps 2-3) is manual; there is no one-command tool yet.
* The history database has been tested against a real PostgreSQL and recorded real traces, but not over a long live session.
