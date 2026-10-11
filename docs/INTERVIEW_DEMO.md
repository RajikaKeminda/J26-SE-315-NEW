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
5. Offline, on 253 real recorded requests: `cd autoscaler && python -m src.evaluation.critical_path_eval`

| Path | Share of a request's downstream time explained | Slowest service of the request is on the path |
|---|---|---|
| **Agent (DFS on measured latency)** | **80 %** | **90 %** |
| Random chain (expected) | 48 % | 54 % |
| Deepest chain | 4 % | 0 % |

Be ready for: "downstream time only" (the gateway is on every path and waits on 35 services that emit no traces, which
would inflate any number); 91 of 253 requests left the gateway through traced services; `ts-order-service` carries about
three quarters of the downstream time, so a path through it scores well - the point is that the agent finds it by itself.

## Demo C - how it orchestrates the cluster (8 min)

The node's scarce resource is memory, not CPU, so at the load it can take nothing is CPU-bound. To have something to
scale, stage a bottleneck (an operator action, reversible, using the same limited service account):

```bash
bash scripts/21_tt_demo_bottleneck.sh create 300m      # ts-order-service: 1 core -> 0.3 core (rolling restart, 1-2 min)
CONTROLLED_SERVICES=ts-order-service bash scripts/20_autoscale_any_cluster.sh --namespace default --act
kubectl get pods -w                                     # in a second terminal
```

With 3 users, `order` (about 80 ms of CPU per request) saturates its 0.3 core. What to point at: its utilisation ring
goes red; the p95 rises; the decision feed shows the proposed and applied scale-up and the safety layer's reasons for anything
blocked (cooldown, cluster budget, node memory); `kubectl get pods -w` shows the new pod (about 95 s to become Ready - Java);
the p95 recovers; stop the load and watch it scale back after the cooldown. Then `bash scripts/21_tt_demo_bottleneck.sh restore`.

The safety layer (say it, it is part of the design): bounds, memory floor, scale-down cooldown, cluster budget, circuit breaker
(5 breaches -> the HPA rules take over), staleness hold, a node-memory headroom check before any scale-up, and a whitelist
of services the agent may touch.

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
| Does it work with any cluster? | It needs Kubernetes access, Prometheus with cAdvisor + kube-state-metrics, and OpenTelemetry traces in Jaeger for the call graph. Verified on two applications (the demo app and Train-Ticket). Istio/Linkerd metrics, Zipkin and Tempo are not implemented. Without traces it can observe but has no call graph. |
| Why only 11 services controlled? | Only those emit traces. The others are observed (CPU, memory) but have no call edges. |
| Did it scale Train-Ticket under load? | See the fine-tuning section: results of the live run are in `docs/TRAINTICKET.md`. On this node scaling is limited to about two extra pods by memory. |
| Is 50+ services proven? | In simulation up to 100 (table above). Live: 47 Deployments observed, 11 with a call graph; a 50-service synthetic cluster does not fit this Mac's 4 GiB Docker VM. |
| Does the simulator match reality? | Calibrated on the demo app (capacity within 3 %). For Train-Ticket only memory, idle CPU, CPU per request and the topology are measured; the queueing terms are not re-fitted for Java. |
| Why is the p95 so high? | Booking requests take 1-2 s on their own: the p95 of this application is set by the request mix, not by load. |
