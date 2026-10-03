# Topology-Aware Multi-Objective DRL Autoscaler — v2

**Project J26-SE-315**

A Kubernetes autoscaler that reads the *service dependency graph* and the
*live latency-pressure graph*, then uses a topology-agnostic reinforcement
learning policy to scale services — balancing SLO compliance, infrastructure
cost, and load distribution.

This version supersedes the earlier POC and real-cluster build. What changed
and why is documented in [What's new in v2](#whats-new-in-v2).

---

## Results

Every arm evaluated on **identical topologies and identical traffic**
(matched seeds, 20 episodes across burst / diurnal / ramp-up / normal
patterns), so the *evaluation* is controlled. Note that each RL arm is a
**single training run** (see limitation 5 on seed sensitivity).
`tests/test_models_and_eval.py` re-runs this evaluation and asserts it still
reproduces `results/ablation_results.json` exactly.

| arm | p50 ms | p95 ms | p99 ms | SLO violations | cost (cores) | composite |
|---|---|---|---|---|---|---|
| static (no scaling) | 171.5 | 583.5 | 597.5 | 40.5% | 1.20 | 1.583 |
| **hpa** (industry baseline) | 58.6 | 143.2 | 209.3 | 2.8% | 1.58 | **0.181** |
| rl_no_topology | 101.8 | 310.8 | 411.4 | 22.6% | 1.63 | 0.723 |
| rl_sdg_only | 69.3 | 248.2 | 393.5 | 18.1% | 2.21 | 0.581 |
| **rl_full** (dual-graph) | **49.3** | **73.2** | **75.2** | **0.0%** | 4.54 | 0.520 |
| rl_single_obj | 49.3 | 73.2 | 75.2 | 0.0% | 4.57 | 0.524 |
| rl_hpa_init *(experimental, see limitation 5)* | 73.6 | 196.5 | 237.7 | 16.2% | 1.68 | 0.347 |

Lower is better everywhere. `composite` = the same weighted latency+cost
penalty used in training, so arms are comparable on one number.

### What this shows

**Each topology component earns its keep.** The ablation gradient is clean:

```
no topology awareness   →  310.8ms p95,  22.6% violations
+ SDG (structure only)  →  248.2ms p95,  18.1% violations
+ EHG (latency graph)   →   73.2ms p95,   0.0% violations
```

Removing the EHG branch alone costs 175ms of p95 latency and 18 percentage
points of SLO compliance. That is direct evidence for the core research
claim — topology awareness, and specifically *dynamic* topology awareness,
materially improves scaling decisions.

**Against HPA:** the full system cuts p95 latency by 49% (143.2 → 73.2ms)
and eliminates SLO violations entirely (2.8% → 0%).

### Honest limitations

These belong in your write-up as findings, not omissions:

1. **The full system costs 2.9× more than HPA** (4.54 vs 1.58 cores). On the
   cost-weighted composite metric, **HPA still wins** (0.181 vs 0.520). The
   RL agent buys its latency advantage with over-provisioning. Whether that
   trade is worth it depends entirely on how you price an SLO breach against
   a CPU core — which is exactly the trade-off a multi-objective formulation
   is supposed to make explicit, and it does.

2. **The reward weights drive the outcome, strongly.** At `W_COST = 0.3` the
   agent learned that maxing out replicas was optimal, and every RL arm
   collapsed to the same degenerate policy. Raising it to `0.55` produced the
   differentiated behaviour above. Both result sets are kept
   (`results/ablation_lowcost_w030.json` vs `results/ablation_results.json`)
   because the sensitivity is itself a result worth reporting.

3. **`rl_single_obj` matches `rl_full` here.** With a latency-only reward the
   agent has no reason not to over-provision, and in this workload that
   happens to produce the same policy as the cost-aware arm at these weights.
   This ablation would separate more clearly under tighter cost pressure or a
   capacity-constrained cluster.

4. **All numbers come from the simulator, not a real cluster.** The
   simulator was calibrated against measurements from the deployed services,
   but the gap is real: running the service app natively and load-testing a
   single replica (`traffic-generator/calibrate.py`) gave a saturation
   throughput of roughly **90–130 req/s per replica**, whereas the simulator
   assumes `MAX_CONCURRENCY_PER_REPLICA × 1000/WORK_MS` = **267 req/s**. The
   service is CPU-bound Python behind the GIL, so its four worker threads do
   not give four-fold throughput. (That measurement ran on a shared 2-core
   machine, so treat it as an order of magnitude, not a constant — run
   `bash scripts/13_calibrate.sh` on your own cluster.) The capacity is now an
   env-var (`MAX_CONCURRENCY_PER_REPLICA`) so the live engine's utilisation —
   and a re-calibrated simulator — can use the measured value. **The
   comparison has still not been run on a real cluster.**

5. **PPO from scratch is unreliable on this task; results depend on the
   seed.** Re-training `rl_full` with a different random seed (150 PPO
   iterations, same weights — an ad hoc check, not a committed result) gave
   p95 73.2 ms / 4.56 cores for seed 0 but p95 127.6 ms / 2.40 cores / 2.7%
   violations for seed 1: a very different point on the trade-off. A
   `W_COST` sweep (`scripts/12_run_pareto.sh`, `results/pareto_results.json`)
   shows the from-scratch agent ending at 3.5–4.6 cores for every cost weight
   from 0.05 to 1.5, versus HPA's 1.6:

   | point | p95 ms | cost (cores) | SLO viol | Pareto-optimal |
   |---|---|---|---|---|
   | static | 583.5 | 1.20 | 40.5% | yes |
   | hpa | 143.2 | 1.58 | 2.8% | yes |
   | rl_full @ W_COST=0.9 | 75.5 | 3.48 | 0.6% | yes |
   | rl_full @ W_COST=1.5 | 74.1 | 3.56 | 0.4% | yes |
   | rl_full @ W_COST=0.3 | 73.2 | 4.32 | 0.0% | yes |
   | rl_full @ W_COST=0.55 (shipped) | 73.2 | 4.54 | 0.0% | no |
   | rl_full @ W_COST=0.05 / 0.15 | 73.2 | 4.57 | 0.0% | no |

   The agent behaves as "scale everything to the maximum" (visible in the
   dashboard demo within seconds). `rl_hpa_init` is an attempt at a fix:
   behaviour-clone HPA first (`src/rl_agent/imitation.py`), then fine-tune with
   PPO. It lands at 1.68 cores — close to HPA's cost, and a better composite
   (0.347) than every from-scratch arm — but with 16% SLO violations, so it
   does **not** beat HPA either. It is shipped as an experimental arm, not as
   the headline. Credible next steps: per-service credit assignment, a
   multi-seed study with confidence intervals, a longer imitation/fine-tune
   schedule.

6. **The ablation gradient is one seed too.** The ordering
   no-topology < SDG-only < full is consistent with the claim, but given
   limitation 5 it should be confirmed across several seeds before it is
   presented as established.

Reproduce with `bash scripts/06_train_all_arms.sh && bash scripts/09_run_ablation.sh`.

---

## What's new in v2

| | v1 | v2 |
|---|---|---|
| **Agent architecture** | One output slot per *named* service (`MultiDiscrete([3]*6)`). Adding a 7th service broke the network shape. | **Shared per-node policy.** One small actor applied to every node's embedding. Verified: the same 16,676 parameters drive 6, 7, 20 or 100 services. |
| **Graphs** | One merged graph; latency lived only in node features. | **SDG + EHG restored** as separate GCN branches — structure and live latency pressure. Tested: SDG stays fixed under load, EHG reacts. |
| **Training topology** | Only the fixed 6-service graph — a shared policy would have overfit to it. | **Random topologies every episode** (4–9 services, varied fan-out), which is what actually teaches a general rule. |
| **RL library** | Stable-Baselines3 | **Custom PPO** (`src/rl_agent/ppo.py`). SB3 needs a flat fixed-size observation and action vector; forcing graphs through it would reintroduce the fixed-size limit the shared policy exists to remove. |
| **Safety** | Rate limit + bounds | Adds **asymmetric scale-down cooldown**, **circuit breaker** (5 consecutive breaches → HPA fallback → auto-recover), **staleness detection**. |
| **Evidence** | "trained agent beats random" | **Full ablation vs HPA and static**, with per-component attribution. |
| **Observability** | Terminal logs | **Live web dashboard** with the dependency graph, critical path, and decision feed. |

---

## What this revision adds

Found and fixed while testing the system end to end:

- **Prometheus outage no longer scales the cluster to minimum.** A failed
  query used to be read as "0 requests/s", which the agent/HPA would act on.
  Failures are now distinguished from "no traffic"; the live engine **holds**
  (no scaling, dashboard banner, `stale: true`) when Prometheus or a
  Deployment can't be read. The documented staleness protection was not
  previously wired in.
- **Live latency now matches what the policy was trained on.** Prometheus
  latency is *inclusive* of downstream calls (gateway ≈ 220 ms when its own
  work is 17 ms); the simulator/policy use a service's *own* latency. The live
  engine now uses Jaeger `cpu_work` spans (with an inclusive→exclusive
  estimate as fallback).
- **Dashboard**: the ablation panel the README described didn't exist (the
  endpoint was never called) — added, plus a Pareto panel, pause/resume/reset,
  demo scenario switching, shadow/stale banners, and de-duplication of the
  "blocked: would exceed max" feed spam. One background ticker now serves all
  browser tabs (previously each tab ran its own control loop).
- **Zero-touch topology** (listed under "future work"): `AUTODISCOVER_TOPOLOGY=1`
  rebuilds the call graph and traffic shares from Jaeger parent/child spans,
  merging conservatively (never drops a service it already manages). Verified
  against the real service app: recovered 0.58/0.42 and 0.55/0.45 for
  declared 0.6/0.4 and 0.5/0.5.
- **In-cluster deployment** (also listed as future work): `autoscaler/Dockerfile`,
  `k8s/03-autoscaler/` with a namespace-scoped Role (get/list deployments,
  get/patch/update `deployments/scale` — nothing else), in-cluster credentials
  with kubeconfig fallback, config via env vars.
- **Pareto sweep**, **HPA-warm-start arm**, **calibration tool**,
  `docker-compose.yml`, `Makefile`, CI (`.github/workflows/ci.yml`, including a
  kind end-to-end job), an e2e smoke script, 66 pytest tests (was 10).

---

## Verification status

What was actually executed, and what was not — please read before relying on it.

| | status |
|---|---|
| 66 tests (`make test`) | **passing** |
| Ablation re-evaluation reproduces `ablation_results.json` | **exact match** |
| Training | every arm trains (1–5 iteration smoke tests; `rl_hpa_init` fully trained, 60 iterations). The four original checkpoints are the ones you shipped — **not re-trained** |
| Dashboard in a real (headless Chromium) browser: graph, feed, table, ablation + Pareto panels, pause/reset/scenario | **run, no JS errors** |
| Service app + Prometheus 2.53 + Jaeger 1.60 as plain processes; Prometheus/Jaeger clients; topology discovery; `LiveEngine` ticking on real metrics | **run** |
| Live engine with no cluster / Prometheus down | **run** (holds, as designed) |
| Kubernetes API calls (`k8s_client`) | **unit-tested with a fake API only** |
| `docker build` of both images, `docker compose up` | **not run** — the sandbox this was developed in could not pull base images from any registry. `docker compose config` validates. |
| kind/minikube deployment, in-cluster autoscaler, RBAC, `14_e2e_smoke.sh` | **not run** (same reason). The CI `e2e-kind` job runs it on GitHub's runners. |

First thing to do on a machine with Docker: `make docker-build kind-up kind-load deploy e2e`
(or push and let CI do it). If something in the untested Docker/K8s layer is off,
that's where.

---

## Relationship to the original proposal

The project description (J26-SE-315) specifies an MO-TD3 agent with a hybrid
continuous/discrete action space (CPU/memory *and* replica counts −3…+3),
Train-Ticket as the benchmark application, 64-d embeddings and HPA **and VPA**
baselines. **This codebase is a different design**: a shared per-node PPO policy
with discrete replica deltas {−1, 0, +1}, a six-service synthetic application,
a 16+16-d dual-GCN embedding, an HPA baseline only, and a scalarised (not
vector-valued) reward. The README's "Design decisions" explain these choices,
but if the proposal is what you'll be assessed against, the gaps above need
either implementing or an explicit justification.

---

## The dashboard

```bash
bash scripts/10_run_dashboard.sh          # demo mode — no cluster needed
bash scripts/10_run_dashboard.sh live     # real Prometheus / Jaeger / Kubernetes
```

Then open **http://localhost:8080**.

- **Dependency graph** — nodes ringed by utilisation, edge thickness = live
  EHG latency pressure, and the current critical path illuminated with a
  travelling pulse. This is the research thesis made visible.
- **Decision feed** — every scaling action *and* every action the safety
  layer blocked, with the reason.
- **Service table** — live replicas, utilisation, latency, per-service action.
- **Latency & cost timeline** — critical-path latency against the SLO line.
- **Header** — RPS, critical-path latency, replica count, cores, and circuit
  breaker state.

`demo` mode drives the built-in simulator and needs nothing else running —
useful for presentations and for your defence. `live` mode reads real metrics
and applies real scaling (requires `scripts/05_port_forward.sh`).

---

## Quick start

### Just the research results (no Kubernetes required)

```bash
cd autoscaler
pip install -r requirements.txt
cd ..

bash scripts/11_run_tests.sh          # 66 tests (or: make test)
bash scripts/06_train_all_arms.sh     # train all 4 ablation arms
bash scripts/09_run_ablation.sh       # head-to-head comparison table
bash scripts/10_run_dashboard.sh      # live dashboard at :8080
```

### Docker only (no Kubernetes)

```bash
docker compose up --build            # 6 services + Prometheus + Jaeger + dashboard (demo mode)
python traffic-generator/generate_traffic.py --pattern burst   # real load -> Prometheus/Jaeger
```

### Full real-cluster deployment

Needs Docker, kind **or** minikube, kubectl, Python 3.11+.
Windows users: see `WINDOWS_SETUP.md` — WSL2 is recommended.

```bash
# kind
make kind-up                         # or: bash scripts/00_kind_up.sh
make docker-build kind-load
# minikube instead:
#   minikube start --cpus=4 --memory=6144 && eval $(minikube docker-env)
#   bash scripts/01_build_images.sh

make deploy                          # namespace, Prometheus, Jaeger, 6 services, in-cluster autoscaler
make e2e                             # traffic -> metrics/traces -> autoscaler scales (self-checking)

# dashboard of the in-cluster autoscaler (set AUTOSCALER_SHADOW=1 in
# k8s/03-autoscaler/autoscaler.yaml to observe without scaling):
bash scripts/05_port_forward.sh      # also forwards the dashboard to :8080
```

Or run the autoscaler from your laptop instead of in-cluster
(`WITH_AUTOSCALER=0 bash scripts/03_deploy_all.sh`):

```bash
bash scripts/04_verify_deployment.sh # wait for readiness

# separate terminal, leave running:
bash scripts/05_port_forward.sh      # gateway :8000, prometheus :9090, jaeger :16686

bash scripts/06_train_all_arms.sh    # train (no cluster needed for this)
bash scripts/07_run_traffic.sh burst 180 20     # real HTTP load
bash scripts/08_run_autoscaler.sh --shadow --iterations 20   # observe only
bash scripts/08_run_autoscaler.sh --iterations 60            # act for real
bash scripts/10_run_dashboard.sh live            # watch it happen
```

Watch pods react in another terminal:
```bash
kubectl get pods -n autoscaler-poc -w
```

---

## The deployed application

```
                    gateway-service  (entry)
                    /0.6            \0.4
          catalog-service          auth-service
          /0.5        \0.5              \1.0
inventory-service  pricing-service   user-service
```

All six are the **same** Docker image (`services/generic-service/`),
configured per-Deployment by environment variable. Each does real CPU-bound
work through a bounded thread pool, so more replicas genuinely means more
throughput and overload genuinely queues. Prometheus metrics on `/metrics`,
OpenTelemetry traces exported to Jaeger.

---

## Project structure

```
├── services/generic-service/     one real microservice, deployed 6×
├── k8s/                          namespace, services, Prometheus, Jaeger,
│                                 03-autoscaler/ (in-cluster autoscaler + RBAC)
├── deploy/compose/               Prometheus config for docker-compose.yml
├── docker-compose.yml  Makefile  .github/workflows/ci.yml
├── traffic-generator/            real async HTTP load generator
├── autoscaler/
│   ├── models/                   trained policies (one per ablation arm)
│   ├── results/                  ablation results as JSON
│   ├── Dockerfile                autoscaler + dashboard image
│   ├── tests/                    66 tests: components, clients, live engine,
│   │                             dashboard API, models/eval, service app
│   └── src/
│       ├── config.py             topology, weights, safety, all tunables
│       ├── topology_intel/
│       │   ├── graph_builder.py  SDG (static) + EHG (dynamic latency)
│       │   └── dual_gcn.py       two-branch GCN, ablation switches built in
│       ├── rl_agent/
│       │   ├── shared_policy.py  ★ topology-agnostic actor-critic
│       │   ├── env.py            graph-structured environment
│       │   ├── ppo.py            custom PPO for variable-size graphs
│       │   └── imitation.py      HPA behaviour-cloning warm start
│       ├── safety/guards.py      bounds, rate limit, cooldown, breaker
│       ├── baselines/rule_based.py   faithful HPA reimplementation
│       ├── evaluation/
│       │   ├── train_arm.py      train one ablation arm
│       │   ├── evaluate.py       head-to-head comparison
│       │   └── pareto.py         reward-weight sweep + Pareto front
│       ├── simulation/simulator.py   calibrated sim + random topologies
│       ├── data_collection/      real Prometheus + Jaeger clients
│       ├── k8s_action/           real Kubernetes scaling
│       ├── inference/live_loop.py    headless control loop
│       └── dashboard/            FastAPI + WebSocket + frontend
└── scripts/                      numbered, run in order
```

---

## Design decisions

**Scalarized multi-objective reward, not a Pareto front.** A weighted sum of
SLO violation, cost, and load variance. True Pareto-front MORL is a separate
research undertaking; the scalarized form still expresses the trade-off
explicitly, and the weight-sensitivity finding above shows the knob works.

**Replica-count actions only.** In-place pod CPU resize is still a beta
Kubernetes feature and unsafe to assume on a student cluster. This mirrors
what HPA itself does.

**Critical path via exact DAG dynamic programming.** Longest-path is NP-hard
on cyclic graphs, but our call graph is acyclic by construction, so the DP is
exact and cheap. On the real cluster the per-service latencies come from real
Jaeger spans (each service emits a `cpu_work` span for its own processing).

**Simulator-first training.** A real cluster step means waiting for pods to
schedule — seconds to minutes. Thousands of RL steps that way is impractical.
The simulator is calibrated against real measurements and trains in minutes.

**GCN and policy trained jointly here** (unlike v1's frozen pretrained GCN).
With the graph encoder inside the shared policy and PPO driving it end to
end, training stayed stable across all arms — but a pretraining stage remains
a reasonable option if you extend to larger topologies.

---

## Known gaps / future work

- **Sim-to-real gap is unmeasured.** Run the ablation on the live cluster.
- **Topology discovery adds, never removes.** `AUTODISCOVER_TOPOLOGY=1`
  picks up new services/edges from Jaeger but won't drop a service it already
  manages (a thin trace sample can't distinguish "removed" from "not sampled").
- **Cold start.** A brand-new service has no history; falling back to HPA
  rules for it until enough traffic accumulates is the standard answer.
- **Cluster capacity.** This scales pods, not nodes. Pair with the Kubernetes
  Cluster Autoscaler for node-level capacity.
- **In-cluster manifests are untested here** (see Verification status). The
  Role grants only what `k8s_client.py` uses.
- **Rolling deployments** of the services themselves aren't coordinated with
  scaling actions; pausing the autoscaler during a rollout would be sensible.
