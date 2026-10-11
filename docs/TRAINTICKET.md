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

## Not done yet

1. Recalibrate the simulator for Java (per-service CPU cost above, 94 s start-up, capacity per core, latency at low CPU).
2. Choose an SLO from a measured baseline; run a step-load test (1 vs 2 replicas of `order`) to see how much replicas help.
3. Fine-tune the policy in the recalibrated simulator, then repeat the live test, against the real Kubernetes HPA.
4. Verify the trace sampling rate (`OTEL_SAMPLE_RATE` is assumed 1.0; all rates scale with it).
5. More memory for the node: the agent could not carry out a single scale-up on this cluster.
6. Services that do not emit traces (35 of 46) are invisible.
