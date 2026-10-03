# Demo guide - running it and showing it to your supervisor

Everything below was run on the project machine (macOS arm64, Docker via OrbStack, kind cluster `autoscaler`).
Open **two or three terminal tabs** at the repo root; each command block says which.

## 0. One-time setup (skip if already done)

```bash
brew install kind kubectl k9s          # kind creates the cluster, k9s is a terminal UI for pods (optional but good for demos)
python3 -m venv .venv && source .venv/bin/activate
pip install -r autoscaler/requirements-dev.txt
make docker-build                       # builds both images
make kind-up kind-load                  # creates the kind cluster, loads the images
```

If the cluster already exists but Docker was restarted: `docker start autoscaler-control-plane` and
`kubectl config use-context kind-autoscaler`.

## 1. Fast demo that needs no cluster (2 minutes) - the simulator control room

```bash
source .venv/bin/activate
bash scripts/10_run_dashboard.sh        # open http://localhost:8080
```

What you see, left to right, top to bottom:

| Panel | What to say |
|---|---|
| Header (p95 / SLO / pods / cost) | "End-to-end p95 latency against the 250 ms SLO, pods and cost against the cluster budget." |
| Service dependency graph | "This is the SDG. Edge thickness is the live latency pressure (the EHG); the red chain is the critical path found by DFS. Node labels are pods × CPU cores." |
| Decision feed | "Every scale / resize the agent made, and every action the safety layer blocked, with the reason." |
| Services table | "Per-service pods, CPU and memory size - the hybrid action space." |
| Objectives & preference | **Drag the SLO slider to 100 and the others to 0** - the SAME trained network moves to more pods / lower latency. Drag cost up - it saves money. "One network, any trade-off." |
| Controller dropdown (top) | Switch **MO_TD3 → HPA → VPA → static** on the burst scenario and watch p95 explode for the baselines. |
| Scenario dropdown | burst / diurnal / rampup / normal. The burst starts after ~2 minutes of demo time. |
| Sim-to-real calibration | "The simulator is fitted to measurements from a real cluster; capacity within 3 %." |
| Controller comparison | Results table (24 matched-seed episodes). Switch to the 14-service topology. |
| Cost vs. latency trade-off | One MO-TD3 network vs the HPA tuning sweep. |
| Training convergence | Learning curves per training seed. |

## 2. Real Kubernetes demo (10-15 minutes)

### 2.1 Start the stack

```bash
# terminal 1
kubectl config use-context kind-autoscaler
bash scripts/03_deploy_all.sh           # namespace, Prometheus, Jaeger, 6 services, the in-cluster MO-TD3 autoscaler
bash scripts/04_verify_deployment.sh    # waits until everything is Ready
kubectl get pods -n autoscaler-poc      # show the supervisor the pods
```

### 2.2 Open the UIs (leave this running in its own terminal tab)

```bash
# terminal 2
bash scripts/05_port_forward.sh
```

| What | URL | What to show |
|---|---|---|
| **Control-room dashboard** (the in-cluster autoscaler) | http://localhost:8080 | live mode: real pods, real CPU, decisions |
| **Prometheus** | http://localhost:9090 | Graph tab, try: `sum by (service)(rate(http_requests_total[30s]))`, `rate(process_cpu_seconds_total[30s])`, `histogram_quantile(0.95, sum by (le)(rate(http_request_duration_seconds_bucket{service="gateway-service"}[30s])))` |
| **Jaeger** | http://localhost:16686 | Service `gateway-service` → Find Traces → open one: the call chain gateway → catalog → inventory, each with a `cpu_work` span |
| Gateway (raw) | http://localhost:8000/work | one real request through the whole chain |

### 2.3 Kubernetes "dashboard"

No web Kubernetes Dashboard is installed in this project (it is not needed). For a live view use either:

```bash
# terminal 3 - the best live view
kubectl get pods -n autoscaler-poc -w
# or the terminal UI (press :deploy then Enter; use 0-9 to switch namespaces; q to quit)
k9s -n autoscaler-poc
# resources actually applied to a service (CPU/memory that the agent resized):
kubectl get deploy -n autoscaler-poc -o custom-columns=NAME:.metadata.name,REPLICAS:.spec.replicas,CPU:.spec.template.spec.containers[0].resources.limits.cpu,MEM:.spec.template.spec.containers[0].resources.limits.memory
```

OrbStack users can also open the OrbStack app → Containers to see the kind node (`autoscaler-control-plane`).
If you really want the official web Kubernetes Dashboard, install it with its Helm chart and a token; that part is
**not tested** in this repo.

### 2.4 Generate load and watch the autoscaler react

```bash
# terminal 3 (or 4)
bash scripts/18_demo_traffic.sh burst      # 12 rps -> ~40 rps -> 12 rps, ~7 minutes, generated INSIDE the cluster
```

While it runs, point at: the dashboard p95 line, the decision feed (scale-ups, resizes, "blocked: exceed cluster budget"),
`kubectl get pods -w` (pods appearing / terminating), Prometheus request-rate graph rising, a Jaeger trace.

**Be ready to explain what you will probably see:** on the real cluster the 3× burst briefly pushes p95 far above the 250 ms SLO
(about 1 s on the project machine), and after 5 breached windows the **circuit breaker** opens and the HPA rules take over until the
SLO recovers. That is the safety design working, and it is also the honest sim-to-real gap documented in the README (the
simulator is optimistic about overload). Do not hide it - it is a good discussion point.

Useful variations:

```bash
bash scripts/18_demo_traffic.sh diurnal                 # smooth wave - the agent tracks it with less drama
bash scripts/18_demo_traffic.sh steady 12 300           # constant 12 rps for 5 minutes
bash scripts/08_run_autoscaler.sh --shadow --iterations 20   # observe-only headless loop from your laptop (scale the in-cluster one to 0 first)
```

Switch the controller **live** in the dashboard dropdown (MO_TD3 / HPA / VPA / threshold / static) between bursts to compare behaviour on the real cluster.

### 2.5 Stop / reset

```bash
# stop port-forwards: Ctrl+C in terminal 2
kubectl scale deploy --all -n autoscaler-poc --replicas=0     # frees the laptop's memory, keeps the cluster
bash scripts/03_deploy_all.sh                                  # later: brings everything back at the manifest sizes
make kind-down                                                 # delete the whole cluster
```

## 3. Show the evidence (no cluster needed)

```bash
make test                                   # 219 tests
open docs/pareto.svg docs/learning_curves.svg docs/live_vs_sim.svg
open README.md                               # or the GitHub page; results tables are generated from the JSON files
python -m src.evaluation.calibration         # (from autoscaler/) re-fit the simulator to the committed real measurements and print the error
```

## 4. Suggested 10-minute storyline for the supervisor

1. **Problem and gap** (1 min): HPA/VPA look at one metric per service; no topology; one objective. Show the TAF diagram.
2. **What was built** (2 min): the four sub-components → `docs/TAF_TRACEABILITY.md` (every TAF requirement mapped to code/tests).
   Hybrid actions (replicas −3…+3, CPU, memory), GCN (128 hidden, 64-d embedding), DFS critical path, MO-TD3 with one Q per objective.
3. **Why the simulator can be trusted** (1 min): calibration panel + `calibration/` measurements from the real kind cluster; capacity within 3 %.
4. **Live demo** (3 min): section 2.4 - burst, graph, feed, preference slider, controller switch.
5. **Results** (2 min): the comparison table - default HPA/VPA/static fail the SLO in 51-94 % of windows; MO-TD3 2.9 %;
   tuned HPA is a strong baseline and the agent is on par (0.219 ± 0.026 vs 0.230). Pareto: one network = a whole front.
6. **Honest findings** (1 min): hybrid CPU/memory actions clearly help; topology benefit suggestive but not established (seed noise);
   HPA demonstrations hurt; live cluster is worse than simulated under overload; resize churn.

## 5. Likely supervisor questions - and the honest answers

| Question | Answer |
|---|---|
| Does it beat HPA? | Beats default HPA/VPA/static by a lot; against a *tuned* HPA it is on par to slightly better (0.219 ± 0.026 vs 0.230), within seed noise. One network covers the whole cost/SLO front. |
| Does the GCN / topology help? | In the pure-RL family the full agent beats no-topology and SDG-only on all three evaluation sets, but seed spread is as large as the gaps and the demo family reverses. Suggestive, not established. |
| Why not real Train-Ticket? | ~40 Java services + databases do not fit in a laptop's 4 GiB Docker VM. A 14-service app with Train-Ticket's *call structure* is used for the unseen-topology test; it was not deployed on the cluster. |
| Why simulation? | A real step takes ≥ 10 s; TD3 needs ~10⁵ steps. The simulator is fitted to real measurements and cross-checked with live runs. |
| Why does it look worse live? | The simulator models overload as a queue draining at nominal capacity; real saturated pods serve less. Recalibrating the overloaded regime and retraining is the next step. |
| Is the safety layer real? | Yes: bounds, memory floor, cooldowns, resize deadband, cluster budget, circuit breaker → HPA, staleness hold. Same code in training and live. |
| What is VPA here? | A faithful re-implementation of the recommender/updater logic, not the VPA controller image. The real Kubernetes HPA was also run on the cluster. |

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `kubectl` cannot connect | `docker start autoscaler-control-plane; kubectl config use-context kind-autoscaler` |
| Pods `Pending` | The cluster has no free CPU: `kubectl scale deploy --all -n autoscaler-poc --replicas=0` then `bash scripts/03_deploy_all.sh`; lower `CLUSTER_BUDGET_CORES` |
| Dashboard shows "Holding - not acting" | Prometheus has no data yet (wait ~30 s after deploy) or the port-forward died; restart `scripts/05_port_forward.sh` |
| Jaeger UI empty / slow | Only 25 % of requests are traced; send traffic first. If Jaeger hit its memory limit: `kubectl rollout restart deploy/jaeger -n autoscaler-poc` |
| Port 8080/9090 already in use | `lsof -ti :8080 | xargs kill` |
| Laptop swapping / very slow | Close other apps; do not run training at the same time as the live demo |
