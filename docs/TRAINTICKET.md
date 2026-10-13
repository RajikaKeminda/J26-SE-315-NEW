# Running the agent against the real Train-Ticket

The agent runs on one machine (here: a MacBook) and orchestrates Train-Ticket (46 Java/Spring services on minikube)
running on another (a Windows PC). Nothing is run inside the cluster, and no demo services are involved.

```
MacBook (agent + dashboard)                      Windows PC (Train-Ticket, minikube)
scripts/19_trainticket_live.sh        ssh/Tailscale tunnels
  Prometheus client  ───────────────► 9090   Prometheus  (cAdvisor + kube-state-metrics)
  Jaeger client      ───────────────► 16686  Jaeger      (OpenTelemetry spans from the Java services)
  Kubernetes client  ───────────────► API    scale Deployments (limited service account)
```

## What the agent reads, and from where

| Signal | Source | Code |
|---|---|---|
| CPU, memory, pod count per service | cAdvisor pod-level series, kube-state-metrics | `data_collection/trainticket_metrics.py` |
| Request rate, own and inclusive latency, errors | OpenTelemetry **server spans** in Jaeger | `data_collection/otel_traces.py` |
| Call graph, traffic shares, critical path | Edges between spans of different services, learned **every control step** | `topology_intel/live_topology.py` |
| Replicas, resource limits | Kubernetes API | `k8s_action/k8s_client.py` |

Own latency of a service = its server-span duration minus the time it waited on other services (parallel calls
counted once). Database time stays in the service's own latency.

The dependency graph is **not declared anywhere**. Only the entry service (`ts-gateway-service`) is configured. Edges
decay (half-life 120 s) and are dropped after 300 s unseen, so the graph shrinks as well as grows; an idle window
freezes it. Only 11 of the 46 services emit traces, so those are the only ones the agent can see.

## Setup (once)

1. **Tunnels** (one command; the port of the Kubernetes API changes whenever minikube restarts):
   ```bash
   ssh -fN -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
     -L 9090:127.0.0.1:9090 -L 16686:127.0.0.1:16686 -L <api-port>:127.0.0.1:<api-port> <user>@<windows-tailscale-ip>
   ```
   Stop a tunnel with `kill <pid>`, never `pkill -f` on a pattern (it kills the other forwards).
2. **A limited service account** on the cluster (not cluster-admin): it may get/list/watch/patch/update `deployments`,
   `deployments/scale` and `pods` in `default` and nothing else. Put its token and the cluster CA in a kubeconfig
   (`~/.kube/tt-agent-config`, `server: https://127.0.0.1:<api-port>`). Tokens last 24 h.
3. **Sync the cluster PC's clock** (`w32tm /resync`). It was 28 s ahead of the Mac. The trace window follows the Jaeger
   server's clock (`jaeger_client.clock_skew_s`), but span timestamps are still shifted.

## Run

```bash
bash scripts/19_trainticket_live.sh                 # SHADOW: observe and decide, change nothing   -> http://localhost:8080
bash scripts/19_trainticket_live.sh act             # really scale
CONTROLLED_SERVICES=ts-order-service,ts-seat-service bash scripts/19_trainticket_live.sh act
```

The script checks Prometheus, Jaeger, the API, the service account, the clock skew and trace freshness first.
The header badge on the dashboard shows where the graph comes from ("live from traces") and how old it is.

## Safety for real actions

On top of the guard used in training (bounds, cooldowns, cluster budget, circuit breaker, staleness hold):

* `CONTROLLED_SERVICES`: only these may be scaled; the others are observed (dashed ring on the dashboard).
* `ALLOW_RESIZE=0` (profile default): replica scaling only. A resize is a rolling restart that runs old and new pod
  side by side, which this node cannot afford.
* **Node memory headroom**: a scale-up is refused unless the node keeps `NODE_MEMORY_RESERVE_MB` (768) free after the
  extra pods (their measured working set) are placed. If the node's memory cannot be read it refuses.
* Services without a Deployment (databases, tracing) are external: they never stop the loop.

## Measured on the real cluster (2026-10-11, node 6 CPUs / 11.7 GiB)

| Quantity | Value |
|---|---|
| New Java pod Ready after a scale-up | **94 s** (the demo app: 4 s) |
| Scale-down | 3 s |
| Node memory in use at 36 pods | 10.6 of 11.7 GiB (about 1 GiB free) |
| CPU per request (ms), median of steady windows | order **81.6**, travel 24.9, seat 18.4, basic 17.1, gateway 14.4, route 13.0, train 10.0, contacts 8.6, station 6.6 |
| End-to-end p95 at ~12 rps | 360-1000 ms (median 730 ms) |
| Where slow requests spend time | `ts-order-service` own time (about 198 ms per slow request); only 5 % of it is database queries |

`order` costs about five times the CPU of the other services, so it saturates its 1-core limit at roughly 12 rps and is
the first bottleneck. These are the priors in `TRAINTICKET_LIVE.work_ms`.
Raw recording of the first live run: `autoscaler/results/trainticket_live_run1.jsonl` (one row per 30 s step).

## Result of the first live run (36 steps, ~12 rps)

* The p95 exceeded the SLO in **33 of 33** loaded steps. The SLO used (370 ms, then 500 ms) is **not calibrated for
  this application**: the default formula is 70 ms + 60 ms per hop. Choose it from a measured baseline.
* The circuit breaker was open for 30 of 35 steps, so the fallback HPA rules were in control most of the time.
* On the controlled services the policy proposed **scale-down 15 times and scale-up 4 times**. Scale-downs were blocked at
  the 1-replica minimum; scale-ups (the HPA fallback rule on `order`) were blocked by the node-memory check.
  When the load stopped (p95 27 ms) the policy proposed scale-down, which is correct.
* The action path itself works: the agent's Kubernetes client scaled `ts-seat-service` 1 -> 2 -> 1 on the real
  cluster through the limited account, and every Deployment was verified identical to its pre-test state afterwards.

**Reading of the result:** the plumbing (observation, discovery, safety, actuation) is validated on a real, heavy
application. The learned policy is not: it was trained in a simulator calibrated on a small Python service where high
latency means high CPU, and its weights, the 4 s pod-start assumption and the SLO do not carry over to a JVM
application. It needs recalibration and fine-tuning before it can be judged.

## Calibration load runs (2026-10-11, steady load of 1, 2 and 3 users, 10 minutes each)

Data: `autoscaler/results/trainticket_calibration.json` (recorded with `src/evaluation/tt_recorder.py`; per-endpoint latency
from `ts-load.py`). The runs were coordinated with a second Claude session on the Windows PC that generated the load
and watched the node.

| Load | req/s (client / entry spans) | e2e p50 | e2e p95 | e2e p99 |
|---|---|---|---|---|
| 1 user | 2.5 / 2.3 | 98 ms | 819 ms | 944 ms |
| 2 users | 4.2 / 3.9 | 116 ms | 1120 ms | 1472 ms |
| 3 users | 6.1 / 5.6 | 104 ms | 1126 ms | 1619 ms |

* **The p95 is set by the request mix, not by load.** The p50 is flat at about 100 ms. Booking requests (`preserve`,
  `preserveOther`) take 0.9-1.3 s even with one user and 1.3-2.1 s with three. A single end-to-end SLO such as the
  default 370 ms cannot be met by this application at any load; an SLO has to be per request class or relative to a
  measured baseline (about 1.1-1.5 s for the whole mix).
* **No service is CPU-saturated** in this range: CPU stays at or below about a third of the 1-core limits, `order` being the
  highest (0.13 core at 1.6 rps, 0.33 at 2.9 rps). Replicas and CPU limits therefore cannot lower latency here.
* **The binding resource is node memory, not CPU.** Memory per service is flat from idle to 3 users (about 290-410 MiB working
  set against 800 MiB limits), so it is a fixed cost per replica. At 3 users the node already stalled once (entry throughput
  fell to 1.1 rps, p95 5.6 s, memory pressure 77 %), so 4 users were not run. Reaching CPU saturation of `order` (about
  9 rps by extrapolation) is not possible on this machine without memory thrash.
* **Fit quality:** `cpu = idle + work x rps` could be fitted reliably only for the gateway (0.043 + 19.8 ms x rps, R2 = 1.00);
  for the others idle JVM CPU dominates. Run 3's CPU series are missing for `order`, `basic`, `seat` and `station`
  (cAdvisor scrapes were missed under node stress) and are stored as null.

**Consequence for the research:** on this cluster the real-cluster experiment validates the pipeline (observation, trace-learned
topology, safety, actuation) but cannot discriminate between scaling policies, because nothing it can scale is the bottleneck
and the node cannot take more load. Policy training and comparison against HPA/VPA therefore stay in the simulator, which has to
be recalibrated for this application (Java start-up 94 s, fixed memory per replica, a node memory budget, measured CPU cost per
request) before its results are claimed for it.

## Critical path: checked against real requests (2026-10-11)

`src/evaluation/critical_path_eval.py` attributes each real request's end-to-end time to services by walking its trace backwards
from the end (`otel_traces.critical_time_by_service`; the values sum to the request's duration) and compares that with the path
the agent finds (the learned graph + the depth-first search). 253 recorded requests (`tests/fixtures/trainticket_traces.json`),
downstream time only (the entry service is on every path, and time spent in the 35 untraced services is blamed on it):

| Path | Downstream time explained | Slowest service is on the path |
|---|---|---|
| **Agent (DFS on measured latency)** | **80 %** | **90 %** |
| Random chain (expected value) | 48 % | 54 % |
| Deepest chain | 4 % | 0 % |

Limits: 91 of the 253 requests left the gateway through traced services; `ts-order-service` carries about three quarters of the
downstream time, so any path through it scores well. The live dashboard shows the same measurement for the current window.

### Critical path: graph-level chain vs. CRISP-style critical shares (held out)

Uber's CRISP (https://www.uber.com/us/en/blog/crisp-critical-path-analysis-for-microservice-architectures/) gives every sampled request its own
critical path - replayed backwards from the end, charging the last-finishing call and ignoring parallel calls that finish earlier - and aggregates the
time per service (and at P50/P95/P99). The agent's depth-first search instead finds ONE chain for the whole graph, weighted by mean own latency.
`src/topology_intel/crisp.py` adds the CRISP-style view: a service's critical share (overall, in the slow tail, and per request type).

Held-out comparison (`python -m src.evaluation.critical_path_eval`; 253 real requests, 5-fold x 20 repeats; the graph and every weight are learned on the training
traces and scored on requests not seen; every method names the same number of services; downstream time only):

| Method | Time explained | Slowest service of a request is on it | Same, slowest 20 % of requests |
|---|---|---|---|
| Current: DFS chain on mean own latency | 80.2 % | 90.1 % | 82.5 % |
| DFS chain weighted by critical share | 79.7 % | 88.0 % | 75.9 % |
| DFS chain weighted by tail critical share | 80.2 % | 90.0 % | 82.2 % |
| Top services by critical share | 88.7 % | 98.5 % | 77.8 % |
| Top services by tail critical share | 85.3 % | 90.1 % | 90.7 % |
| **Top services per request type** | **95.6 %** | **99.7 %** | **91.0 %** |
| Random chain (expected) | 47.3 % | 53.8 % | 41.4 % |

What this shows: changing only the WEIGHTS of the chain search does not help; the gain comes from the representation - a ranked set of services instead of one chain, and
one set per request type (a `contacts` request is 0 % covered by the chain). The overall ranking alone is worse than the chain in the slow tail, so the tail view matters.
Limits: one application, 91 requests with downstream time, hindsight-free but small; the new view explains where requests wait - whether feeding it to the agent improves
scaling decisions is NOT measured (it would need the simulator to produce the same shares and a retrain). It is available as an opt-in input
(`CRITICAL_PATH_SOURCE=crisp`, off by default because the shipped models were trained with the chain) and always shown on the dashboard.

## Live orchestration on the real cluster (2026-10-11)

Setup: `ts-order-service` limited to 0.3 core (an operator action, restored afterwards) so that it saturates at a load the node can take;
the agent (`mo_td3_trainticket_cap6`, replica-only) in act mode, restricted to `ts-order-service`, at most 2 replicas, SLO 1400 ms (1.25 x the
p95 measured at 3 users). Records: `results/trainticket_live_orchestration_run{1,2}.json`; replayable frames: `results/replays/trainticket_run.jsonl`.

| | One pod, CPU saturated | Two pods Ready |
|---|---|---|
| Run 1 (no bookings, see caveat): order own p95 / end-to-end p95 | 934 / 648 ms | 504 / 434 ms |
| Run 2 (with bookings, 5.6 req/s, 0 errors): order own p95 / end-to-end p95 | 1191 / 1920 ms | 824 / 1346 ms |

In both runs the agent scaled `order` 1 -> 2 within the first minutes of load, kept the second pod while the load lasted (the pod needed about 4 minutes
to become Ready under the CPU limit) and scaled back to 1 when the load ended; the cluster was verified identical to its saved baseline afterwards.
In run 2 the circuit breaker opened (p95 2283 ms for 5 steps, one saturated pod), the HPA rules took over, and it closed again once the second pod
was Ready. Run 2 was the faithful one: **run 1 had no booking requests** (the services run on China time and the load script took the PC's date, so
from 21:30 IST every trip search was empty), so its latencies are not comparable with the calibration runs.
Both runs had a one-minute node stall at about 7-10 minutes into a 3-user run (also in calibration run 3, cause unknown) that never reached an abort line.

### What the live runs found and fixed

* **CPU series dropped by a too-short window.** A 60 s rate window over a 30 s cAdvisor scrape interval silently dropped the CPU of 13 of 36 services
  (including the saturated one, which then read as idle). Now the window is 4x the scrape interval (derived by `auto_profile`), and the agent holds when more than 25 % of CPU series are missing.
* **Scale-up undone at once.** The guard's cooldown counted only the time since the last scale-DOWN. It now also blocks a scale-down within `SCALE_DOWN_AFTER_UP_STEPS`
  (10 steps = 5 min) of a scale-up, and refuses while the newest pod is younger than that (read from the cluster, so it survives an agent restart). Off by default so the
  simulated results and trained models stay as shipped.
* **Cluster budget vs. observed services.** With untraced services observed, the cost counted all 33 running services against a budget sized for 11 and every scale-up was refused.
* **Node-memory reserve** (768 MiB) refused a scale-up by 21 MiB where the real margin was ~420 MB above the abort line; lowered to 600 MiB.
* **Memory** was the binding resource: SkyWalking (851 MiB, not even working) had to be stopped for the node to have room for a second `order` pod.

## Training results in the Train-Ticket regime (simulator: real topology, Java timing, measured CPU cost and memory)

`results/trainticket_sim_cap2.json`, `trainticket_sim_cap6.json`, `trainticket_pareto_cap6.json`, `src/evaluation/regime_eval.py`, `pareto_regime.py`:

| Regime | Controller | SLO violations | p95 | Cost | Score (lower is better) |
|---|---|---|---|---|---|
| replicas <= 2 (what the node allows) | tuned HPA | 47.6 % | 1090 ms | 13.6 | **0.458** |
| | agent (fine-tuned) | 45.3 % | 1218 ms | 14.2 | 0.471 |
| replicas <= 6 (room to scale) | tuned HPA | 10.6 % | 312 ms | 17.3 | **0.211** |
| | agent trained in this regime | **7.1 %** | **281 ms** | 19.7 | 0.225 |

With room to scale, the agent has fewer SLO violations and a lower tail latency than the tuned HPA at a 14 % higher cost; at the default preference the HPA's weighted score
is marginally better. Swept over preferences, one network's trade-off front (hypervolume 0.510) matches the best front of the HPA tuning sweep (0.500): parity with more
flexibility, not a win. With the 2-replica cap that this node imposes every controller is capacity-limited, and the agent is on par with the tuned HPA.

## Many services (simulation; `results/scale_results.json`, `scale_results_size_trained.json`, `src/evaluation/scale_eval.py`)

Random topologies of N services, matched seeds, the HPA re-tuned for each size. The original agent was trained on 4-12 services; "size-trained" was fine-tuned on 4-64 services.

| Services | Decision time | Violations: agent | size-trained | tuned HPA | Cost: agent | tuned HPA |
|---|---|---|---|---|---|---|
| 8 | 0.4 ms | 16.4 % | 15.0 % | 17.5-33.9 % | 4.6 | 4.9 |
| 14 | 0.4 ms | 5.6 % | 4.4 % | **2.4 %** | 8.3 | 9.0 |
| 25 | 0.4 ms | 14.9 % | 12.8 % | 28.3 % | 15.0 | 16.3 |
| 50 | 0.6-0.8 ms | 22.1 % | 17.5 % | 49.7 % | 29.2 | 30.9 |
| 100 | 0.9 ms | 49.3 % | 47.2 % | 96.8 % | 52.8 | 57.9 |

Live: the real cluster has 47 Deployments (33 running); one API call reads all of them per step, traces give the call graph for 11, and the rest are shown as observed-only nodes.
A synthetic 50-service cluster does not fit the 4 GiB Docker VM of the controller machine.

## Not done yet

1. The simulator is only partly re-calibrated for Java (memory, idle CPU, CPU per request, topology); the queueing terms are the demo service's.
2. The SLO of 1400 ms is a baseline-derived choice, not a requirement of the application.
3. The cause of the one-minute node stalls (about 7-10 minutes into 3-user runs) is unknown.
4. Only 11 of 47 services emit traces: the others are observed, not part of the call graph.
3. Fine-tune the policy in the recalibrated simulator, then repeat the live test, against the real Kubernetes HPA.
4. Verify the trace sampling rate (`OTEL_SAMPLE_RATE` is assumed 1.0; all rates scale with it).
5. More memory for the node: the agent could not carry out a single scale-up on this cluster.
6. Services that do not emit traces (35 of 46) are invisible.
