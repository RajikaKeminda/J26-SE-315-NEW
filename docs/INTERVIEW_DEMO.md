# Interview demo - the autoscaler on the real Train-Ticket

The cluster (Train-Ticket, 47 Deployments, 33 running) is on a separate PC; the agent and the control room run on the MacBook.
Four things to show, in this order. Every number below was measured; the limits are stated where they matter.

## The 60-second pitch

"The agent is plugged into a cluster it has never seen. It discovers the services and their call graph from live traces,
finds the critical path, and decides how to scale each service. One command inspects any cluster and derives the settings;
it starts in shadow mode, so nothing changes until I say so. The critical path it finds is checked against what real
requests actually waited for."

## Before you go in (20 minutes ahead)

1. **Windows PC:** Docker Desktop + minikube running; `ts-start.ps1` done; all Train-Ticket pods Ready; clock synced
   (`w32tm /resync`, admin PowerShell); do not run `ts-stop.ps1` while presenting.
2. **Tunnels from the Mac** (one command; the API port changes when minikube restarts - `kubectl config view --minify` on the PC):
   ```bash
   ssh -fN -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
     -L 9090:127.0.0.1:9090 -L 16686:127.0.0.1:16686 -L <api-port>:127.0.0.1:<api-port> "<user>"@<windows-tailscale-ip>
   ```
   Put the same `<api-port>` in `server:` of `~/.kube/tt-agent-config`; the service-account token lasts 24 h.
3. **Load (on the PC):** `python "D:\train ticket\ts-load.py" <url> --users 3 --think 1.0 --seconds 1800` (about 6 req/s).
   More than 3 users makes this node stall; do not exceed it.
4. **Preflight:** `bash scripts/19_trainticket_live.sh` prints what is reachable (Ctrl+C after the checks).
5. **Fallback:** record a good run now so there is something to replay if the network fails:
   `python -m src.dashboard.record --out results/replays/trainticket_run.jsonl --seconds 600` while the control room runs.

## Demo A - plug into any cluster (3 min)

```bash
bash scripts/20_autoscale_any_cluster.sh --namespace default --discover-only
```

Nothing is told to it except the namespace and the two URLs. Read the output aloud:

* entry service detected from the traces (the service that is the root of most of them);
* which Deployments are traced (11 of 47) and therefore controllable;
* action bounds derived from the current limits and the **measured memory working sets** (never below 1.25x);
* replica cap derived from the node's free memory (the node has about 0.9 GiB free, so 2);
* pod start-up **measured from real pods** (creation to Ready) and the control step derived from it;
* a cost budget from the current allocation; shadow mode and replica-only scaling as defaults.

Then start it: `bash scripts/20_autoscale_any_cluster.sh --namespace default` -> http://localhost:8080.
Work with another application by changing only `--namespace`, the URLs and (if traces do not reveal it) `--entry`.

## Demo B - how the critical path is detected (6 min)

Open the control room next to Jaeger (http://localhost:16686).

1. The dependency graph starts with **one node** (the entry service) and fills in as traces arrive. The header says
   "topology - live from traces - N services - M edges - updated Ns ago". Nothing is declared.
2. The graph is learned from spans: an edge is a server span whose nearest ancestor belongs to another service; call
   counts decay (half-life 2 min) and an edge unseen for 5 min is dropped, so it can shrink as well as grow.
3. The lit red chain is the critical path (depth-first search of the weighted graph, as in the proposal).
4. **Proof panel "Critical path - verified against real requests":** for each real request in the window the end-to-end
   time is attributed to services by walking its trace backwards from the end. The panel shows how much of the downstream
   time the agent's path explains, against a random chain through the same graph, and links to the slowest request in Jaeger.
   Open that trace and show that the services on the waterfall are the ones on the red chain.
5. **Critical shares (CRISP-style, below the proof panel):** per service, the share of the downstream time it is the critical path for, over all requests and over
   the slowest 20 %, and the services that carry each request type. Say where it comes from: Uber's CRISP (per-request critical path, replayed backwards,
   aggregated per service). It shows what a single chain cannot: `contacts` requests wait on `contacts`, which no chain through `order` contains.
6. Offline, on 253 real recorded requests: `cd autoscaler && python -m src.evaluation.critical_path_eval` (prints the table below and a held-out comparison)

| Path | Share of a request's downstream time explained | Slowest service of the request is on the path |
|---|---|---|
| **Agent (DFS on measured latency)** | **80 %** | **90 %** |
| Random chain (expected) | 48 % | 54 % |
| Deepest chain | 4 % | 0 % |

Held-out (learned on training requests, scored on others): the chain explains 80 % of the downstream time, the top services per request type 96 %; weighting the chain
differently changes nothing - the gain is from using a set per request type instead of one chain.

Be ready for: "downstream time only" (the gateway is on every path and waits on 35 services that emit no traces, which
would inflate any number); 91 of 253 requests left the gateway through traced services; `ts-order-service` carries about
three quarters of the downstream time, so a path through it scores well - the point is that the agent finds it by itself.

## Demo C - how it orchestrates the cluster (8 min)

The node's scarce resource is memory, not CPU, so at the load it can take nothing is CPU-bound. To have something to scale, stage a
bottleneck (an operator action, reversible, with the same limited service account), then let the agent act on that one service:

```bash
bash scripts/21_tt_demo_bottleneck.sh create 300m      # ts-order-service: 1 core -> 0.3 core (rolling restart, 1-2 min)
bash scripts/20_autoscale_any_cluster.sh --namespace default --controlled ts-order-service \
     --slo 1400 --arm mo_td3_trainticket_cap6 --reserve 600 --act
kubectl get pods -w                                     # second terminal
# ... after the demo:
bash scripts/21_tt_demo_bottleneck.sh restore
```

Preconditions that mattered in the rehearsals: SkyWalking stopped (it held 851 MiB of a node with about 1 GiB free; without that the agent
rightly refuses to add a pod), and the load generator running with the date fix (so the booking requests run).

**What was measured live (two runs, `docs/TRAINTICKET.md`):** within the first minutes of 3-user load `order` hits 100 % CPU on its single
0.3-core pod; the agent scales it 1 -> 2 (the second pod needs about 4 minutes to become Ready - Java under a CPU limit); `order`'s own p95
falls from 1191 to 824 ms and the end-to-end p95 from 1920 to 1346 ms (booking mix, run 2); when the load stops it scales back to 1. In run 2 the
circuit breaker opened while one pod was saturated and closed again once the second pod was Ready.
What to point at: the utilisation ring of `order` going red; the decision feed (proposed vs applied, and the guard's reasons: node memory,
cooldown, replica cap); the new pod in `kubectl get pods -w`; the p95 line; the breaker pill.

Say it plainly: with a 4-minute pod start-up a reactive controller is slow, which is the case for anticipating load; the node can hold only
one extra `order` pod; and on this node a tuned HPA would do about as well (see the table in `docs/TRAINTICKET.md`).

The safety layer: bounds, memory floor, scale-down cooldown, **a stabilisation window after a scale-up** (found necessary in the rehearsal: the
agent had undone its own scale-up), cluster budget, circuit breaker (5 breaches -> the HPA rules take over), staleness hold (also when many CPU
series are missing), a node-memory check before any scale-up, and a whitelist of services the agent may touch.

## Demo D - scale (3 min)

* **Decision time vs. number of services** (simulator, random topologies, the agent was trained on 4-12 services):

| Services | Decision | SLO violations: agent | tuned HPA | no scaling |
|---|---|---|---|---|
| 8 | 0.4 ms | 16 % | 18 % | 96 % |
| 14 | 0.4 ms | 6 % | 2 % | 98 % |
| 25 | 0.5 ms | 15 % | 28 % | 99 % |
| 50 | 0.6 ms | 22 % | 50 % | 100 % |
| 100 | 0.9 ms | 49 % | 97 % | 100 % |

  (`python -m src.evaluation.scale_eval`; the tuned HPA was re-tuned for each size; at 14 services it is better.)
* **Live:** the same cluster has 47 Deployments (33 running). The agent reads all of them in **one** API call per step,
  traces give the call graph for 11, and `OBSERVE_UNTRACED=1` shows the other running services as observed-only nodes.
* Prometheus and Jaeger are queried in bulk: the number of queries per step does not depend on the number of services.

## If something fails

| Symptom | Do |
|---|---|
| Tunnel dropped / "Holding - not acting" | `scripts/19_trainticket_live.sh` shows which one; restart the tunnel command above |
| Node stalls (pods restarting) | stop the load generator first; the agent only holds |
| Everything is down | `python -m src.dashboard.server --mode replay --replay results/replays/trainticket_run.jsonl` - the page says it is a replay |
| API port changed | new port in the tunnel and in `server:` of `~/.kube/tt-agent-config` |

## Honest answers to likely questions

| Question | Answer |
|---|---|
| Is a single critical path enough? | It is what the proposal specifies and it explains 80 % of the downstream time here, but a single chain misses request types with their own bottleneck (`contacts`: 0 %). The CRISP-style critical shares (per service, tail, per request type) explain 96 % held out and are shown live. Whether they improve the scaling decisions is not measured yet. |
| Does it work with any cluster? | It needs Kubernetes access, Prometheus with cAdvisor + kube-state-metrics, and OpenTelemetry traces in Jaeger for the call graph. Verified on two applications (the demo app and Train-Ticket). Istio/Linkerd metrics, Zipkin and Tempo are not implemented. Without traces it can observe but has no call graph. |
| Why only 11 services controlled? | Only those emit traces. The others are observed (CPU, memory) but have no call edges. |
| Did it scale Train-Ticket under load? | Yes, in two live runs it scaled the staged bottleneck `ts-order-service` 1 -> 2 and back, and the second pod cut its own p95 by about 30-50 %. The bottleneck was staged (CPU limit lowered), the node allows one extra pod, and run 1 lacked the booking requests (a load-script date bug, documented). |
| Is 50+ services proven? | In simulation up to 100 services (table above; the agent was trained on 4-12 and fine-tuned on up to 64). Live: 47 Deployments in the cluster, 33 running and all observed, 11 with a call graph; a 50-service synthetic cluster does not fit this Mac's 4 GiB Docker VM. |
| Is the agent better than a tuned HPA? | With room to scale (simulation, Train-Ticket topology) it has fewer SLO violations (7 % against 11 %) and a lower tail latency at a 14 % higher cost; swept over preferences its front matches the best HPA front. With the node's 2-replica cap every controller is capacity-limited and they are on par. Not a clear win; the added value is the critical path, the multi-objective control and the safety layer. |
| Does the simulator match reality? | Calibrated on the demo app (capacity within 3 %). For Train-Ticket only memory, idle CPU, CPU per request and the topology are measured; the queueing terms are not re-fitted for Java. |
| Why is the p95 so high? | Booking requests take 1-2 s on their own: the p95 of this application is set by the request mix, not by load. |
