# Traceability: Topic Assessment Form (J26-SE-315) → this repository

The TAF specifies a four-component, closed-loop framework: **Environment Modeling → Topology
Feature Extraction → Multi-Objective DRL Agent → Reward Engineering & Evaluation**. This table
maps every sub-component and every concrete requirement in the TAF to the code that implements
it, the test that guards it, and an honest status. "Deviation" rows are explained in the README
section *Deviations from the proposal*.

Legend: ✅ implemented as specified · ◐ implemented with a stated deviation · ⛔ not done

## Sub-component 1 — Environment modeling & state representation (Dilhara W. L.)

| TAF requirement | Where | Tests | Status |
|---|---|---|---|
| 1.1 Deploy the benchmark on Kubernetes with resource limits | `services/generic-service/`, `k8s/01-services/` (generated from `autoscaler/src/topologies.py` by `scripts/gen_manifests.py`), `scripts/00–05` | `test_manifests.py` | ◐ Train-Ticket *shape* on a generic service image (14-service preset `trainticket14`); the 6-service app is what runs on the cluster. The real ~40-service Java/Spring Train-Ticket does not fit a laptop-class cluster |
| 1.2 Normal, burst and diurnal workloads | `autoscaler/src/simulation/simulator.py::generate_traffic` (+ ramp-up), `traffic-generator/generate_traffic.py` (open-loop Poisson, same curves) | `test_simulator.py`, `test_service_and_traffic.py` | ✅ |
| 1.3 Collect CPU, memory, request rate, replica count, latency, service-call traces | `data_collection/prometheus_client.py` (12 queries/tick: rps, own + inclusive latency mean/p50/p95/p99, **real** CPU via `process_cpu_seconds_total`, **real** RSS, in-flight, errors), `k8s_client.get_state`, `jaeger_client.py` | `test_clients.py`, `test_engine_and_api.py` | ✅ |
| 1.4 Clean / normalise / combine into RL-ready state | `topology_intel/graph_builder.py` (18 node features, 6 cluster scalars), `inference/live_state.py` (real signals → the simulator's schema) | `test_topology_and_graphs.py`, `test_engine_and_api.py` | ✅ |
| 5-second sliding trace windows; SDG edges = call frequency; EHG edges = latency percentiles | `jaeger_client.aggregate_window` (per-edge count, p50, p95 from callee request spans; `TRACE_WINDOW_SECONDS = 5`), `graph_builder.build_sdg/build_ehg` | `test_clients.py::test_window_aggregation…` | ✅ |

## Sub-component 2 — Topology feature extraction (Upasena P. M. R. K.)

| TAF requirement | Where | Tests | Status |
|---|---|---|---|
| 2.1 Service Dependency Graph | `graph_builder.build_sdg`, `jaeger_client.discover_topology` (zero-touch topology from traces) | `test_topology_and_graphs.py`, `test_engine_and_api.py` | ✅ |
| 2.2 Execution History Graph | `graph_builder.build_ehg` (edge = share × callee inclusive p95 / SLO; trace-derived edge percentiles when available) | `test_topology_and_graphs.py` | ✅ |
| 2.3 2-layer GCN, 128 hidden units, 64-d embedding | `topology_intel/dual_gcn.py` (two branches × 32-d = 64-d per service; `config.GCN_HIDDEN_DIM = 128`); `MOTD3Agent.graph_embedding()` returns the 64-d whole-application vector + critical-path nodes | `test_agent.py::test_gcn_embedding_dimensions…` | ✅ |
| 2.4 DFS critical path | `topology_intel/critical_path.py::critical_path_dfs` (memoised DFS; verified against brute force) | `test_topology_and_graphs.py` | ✅ |
| "Critical Path nodes" passed to the agent | node features `on_critical_path`, `critical_path_share` (+ dashboard) | `test_topology_and_graphs.py` | ✅ |

## Sub-component 3 — Multi-objective DRL agent (Perera H. A. T. J.)

| TAF requirement | Where | Tests | Status |
|---|---|---|---|
| 3.1 RL environment (state, observation flow, transitions) | `rl_agent/env.py` (graph-structured state, guard in the loop, vector reward), `simulation/simulator.py` (calibrated on a real cluster) | `test_guard_and_env.py`, `test_simulator.py` | ✅ |
| 3.2 Hybrid action space: continuous CPU/memory, discrete replica delta −3…+3 | `config.REPLICA_DELTA_OPTIONS`, `env.decode_action` (log-scale CPU 0.1–1.0 core, memory 96–512 MiB targets; replica delta = `round(3a)` of a continuous head, or a 7-way categorical head as an ablation), `k8s_client.set_resources` (real resize via rolling update) | `test_guard_and_env.py`, `test_clients.py` | ✅ |
| 3.3 MO-TD3 | `rl_agent/mo_td3.py`: twin critics with **one Q per objective**, target-policy smoothing, delayed actor, Polyak targets; preference-conditioned actor; deep-set critic that gives each service its own ∂Q/∂aᵢ | `test_agent.py` | ✅ |
| 3.4 Train and optimise the policy | `rl_agent/trainer.py` (random topologies every episode, preference sampling + relabelling, n-step returns, TD3+BC demonstrations, held-out validation, best-checkpoint selection) | `test_training_and_eval.py` | ✅ |
| Concrete output "Scale Service-A to 5 pods, allocate 2.0 CPU cores" | `engines.py` payload / `live_loop.py` log lines (`gateway: +2pod, cpu 0.25->0.50`) | `test_engine_and_api.py` | ✅ |

## Sub-component 4 — Reward engineering & evaluation (Perera G. A. T. D.)

| TAF requirement | Where | Tests | Status |
|---|---|---|---|
| 4.1 Normalised multi-objective reward: SLO violation, cost, load-distribution variance | `rl_agent/rewards.py` (penalty vector in ~[0,1] per objective) | `test_guard_and_env.py` | ✅ |
| 4.2 Baselines: Kubernetes HPA / VPA / rule-based; metrics p95/p99, SLO, cost, utilisation, adaptation time, convergence | `baselines/rule_based.py` (HPA with tolerance, scale-up policy, stabilisation window; VPA recommender+updater; threshold; static), `evaluation/metrics.py`, learning curves in `results/training/` | `test_baselines.py`, `test_training_and_eval.py` | ✅ + a tuned HPA so the learned agent is not compared with a straw man |
| 4.3 Benchmarking under normal / burst / diurnal workloads | `evaluation/evaluate.py` (simulator, matched seeds), `evaluation/live_experiment.py` (**real cluster**, in-cluster load generator, paired with the simulator) | `test_training_and_eval.py` | ✅ |
| 4.4 Trade-off analysis, Pareto-optimal trade-offs | `evaluation/pareto.py` (one network swept over preferences vs. the HPA tuning sweep; hypervolume) | `test_training_and_eval.py` | ✅ |
| Compare "against Kubernetes HPA/VPA" on the live system | `live_experiment.py` supports our HPA, the **real Kubernetes HPA controller** (`k8s_hpa`, metrics-server via `scripts/16_install_metrics_server.sh`), VPA-style right-sizing | — (needs a cluster) | ✅ / VPA: our faithful re-implementation, not the VPA controller image |

## Cross-cutting requirements in the TAF

| TAF statement | Where |
|---|---|
| "closed-loop … decisions applied to the Kubernetes environment and the resulting performance is continuously monitored and returned" | `inference/engines.py::LiveEngine.tick`, in-cluster deployment `k8s/03-autoscaler/` |
| "Open-source libraries such as Gymnasium, Stable-Baselines3, MORL-Baselines, and PyTorch" | PyTorch only. Gymnasium/SB3 assume a fixed-size flat observation/action vector, which contradicts the topology-agnostic shared policy (README, *Design decisions*) |
| "data … generated from the open-source benchmark … stored after removing unnecessary identifiers" | the benchmark is synthetic; no personal data exists. Calibration measurements are committed in `calibration/` |
| "Cluster … SDG 12: dynamically scaling down redundant instances" | cost objective + scale-down guard; visible as cost reduction vs. static in the results |
