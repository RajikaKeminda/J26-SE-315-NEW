# Does the agent find the bottleneck wherever it is? (2026-10-24 / 25)

The point of the system is to read live data, find the service that limits the application, add capacity there, and leave the rest alone. A single-service demo
cannot show that (the agent might simply love `ts-order-service`), so this document tests it directly: **in the simulator for every service of an application, on a
real Kubernetes cluster for three different services of the demo app, and on the real Train-Ticket for two different services.** Raw data: `autoscaler/results/dataset/`,
`autoscaler/results/bottleneck_live_runs.json`, `autoscaler/results/bottleneck_eval_*.json`.

## 1. What was wrong, measured
The first replicas+CPU model (`mo_td3_trainticket_cpu`) proposed the same thing for `ts-order-service` at every load, idle or busy, and never scaled down. In the simulator, when any
one service was turned into the bottleneck, only **11 %** of the capacity it added went to that service (random choice: 10 %). Reason: in training, order was always the heavy service,
so "scale order" was a good habit and reading the load was not needed.

## 2. What was changed
| Change | Where |
|---|---|
| **Bottleneck scenarios:** the request mix is shifted towards one random service, the traffic scaled until that service needs more CPU than its small starting limit, and no other service saturates | `topologies.py::bottleneck_scenario`, `shift_mix` |
| **Training on them** (70 % of episodes, any application, a different service each time), validation on a scenario for every service | `trainer.py` (`hot_prob`), `scripts/24_train_bottleneck_aware.sh` -> `models/mo_td3_bn.pt` (demo regime), `models/mo_td3_bn_tt.pt` (Train-Ticket regime) |
| **A test that measures it:** did the bottleneck get capacity in 2 steps, how much capacity went to idle services, steps until the SLO is back, violations | `evaluation/bottleneck_eval.py` |
| **Demand gate:** capacity may be added only to a service that is busy (CPU >= 50 %) and removed only from one that is not | `safety/guards.py` (`SCALE_UP_MIN_UTIL`, `SCALE_DOWN_MAX_UTIL`) |
| **Resize safety**, found necessary the hard way (section 5): never shrink a CPU limit, at most one resize per step, memory locked | `ALLOW_CPU_DECREASE=0`, `MAX_RESIZES_PER_STEP=1`, `ALLOW_MEMORY_RESIZE=0` (in `profiles/trainticket.env`) |

## 3. Simulator: every service of the 6-service app as the bottleneck (`bottleneck_eval --topology demo6`, 6 episodes each)
| Controller (gate on, memory locked) | bottleneck got capacity in 2 steps | capacity wasted on idle services | steps until SLO is back | SLO violations | score (lower is better) |
|---|---|---|---|---|---|
| **`mo_td3_bn` (new)** | **100 %** | 0 % | **4.4** | **27.8 %** | **0.370** |
| `mo_td3_full` (previous agent) | 100 % | 0 % | 8.7 | 36.4 % | 0.420 |
| tuned HPA | 100 % | 0 % | never | 100 % | 0.768 |
| static | 0 % | 0 % | never | 100 % | 0.730 |

Without the gate `mo_td3_bn` also reacts in the first 2 steps (SLO back in 2.4 steps) but 58 % of the capacity it adds goes to services that were not busy (small CPU nudges).
The tuned HPA cannot recover here because the scenario starts the bottleneck at 0.1 core and the HPA only adds replicas (a handicap of the scenario, not a general statement about the HPA).

## 4. Live on real Kubernetes (kind, the 6-service demo app, `scripts/25_bottleneck_demo_kind.sh`, 12 req/s, agent `mo_td3_bn` acting, history in the database as cluster `kind-demo`)
The same agent, three different services made the bottleneck one after another (limit 25 millicores, 1 pod):

| Bottleneck | saturated at (UTC) | what the agent did | utilisation afterwards |
|---|---|---|---|
| `pricing-service` | 05:42:35 | +1 pod at 05:42:46, 3 pods then CPU 0.55 core at 05:43:08, back to 1 pod | 0.04-0.13 |
| `inventory-service` | 05:47:16 | CPU raised to 0.35 core by 05:47:52 (about 35 s) | 0.33 |
| `user-service` | 05:49:12 | pods 1 -> 2 -> 4 -> 6 with 0.1 core each by 05:50:18 | 0.24 |

Each time the saturated service was found within one control step (10 s) and fixed in under a minute; the end-to-end p95 stayed mostly between 110 and 400 ms, with short spikes up to 1.6 s. Besides, the agent re-sized other services a little
(CPU nudges on idle services: the known "resize churn", which causes short p95 spikes of 1-2 s).

## 5. Live on the real Train-Ticket (47 Deployments, `mo_td3_trainticket_cpu`, 12 controlled services, the agent acting, history as cluster `train-ticket`, run `d0dbc6fe-480b-466a-bc31-7023f201cddb`)
Normal request mix, 3 users, 26 minutes; the PC session ran the load and a watchdog.

| Time (PC clock) | Event |
|---|---|
| before the load | agent acting on an idle cluster: **0 changes** (every proposal blocked: no demand / CPU decrease disabled) |
| 12:56:56 | load starts; `ts-order-service` staged at 300m, 1 pod (saturated) |
| 12:57:48 | **the agent raises `ts-order-service` to 1 core and 2 pods** (about 50 s after the load started); Ready 12:59:02 |
| 13:06:10 | the operator cuts `ts-order-other-service` (untraced: only its CPU is visible) from 1 core to 200m; Ready 13:07:57 |
| 13:08:29 | **the agent raises `ts-order-other-service` to 850m** (it saw 94 % utilisation one step after the pod was Ready); Ready 13:09:33 |
| whole run | **10 of the 12 controlled services were never changed**, `ts-order-service` was not touched again when the second bottleneck appeared |

Cost, from the load generator: 8330 calls, **243 failed (2.9 %)**, all inside the three restarts (order resize 104; the two `ts-order-other-service` restarts 139), none in the steady states between them.
The two outages hit different call paths (order/refresh and travel failed in the first; orderOther/refresh, travel2 search and booking in the second). Node: MemAvailable never below 863 MiB, no pod restarted, no OOM.

## 6. Train-Ticket in the simulator
`bottleneck_eval --topology trainticket-sim` (gate on): the bottleneck gets capacity in 2 steps for every model (100 %), nothing is wasted on idle services, but SLO recovery is slow for all of them
(16-25 steps, 68-77 % violations) because a Java pod takes 95 s to start and the node allows 2 replicas. The Train-Ticket-specific bottleneck-aware model (`mo_td3_bn_tt`, score 0.499) is **not** better than the replica+CPU model (`mo_td3_trainticket_cpu`,
0.497); that one stays the deployed model there. Only `ts-order-service` and `ts-basic-service` can be made the only bottleneck at the traffic this node can take: the request mix of Train-Ticket sends
almost every path through order and order-other.

## 7. What went wrong on the way (kept, because the fixes are part of the result)
* **A restart storm:** with 12 controlled services the agent's first step shrank the CPU limit of 11 of them at once (idle right-sizing), recreating 10 Java pods together. It came up in about 3 minutes with no harm (lowest MemAvailable 1581 MiB) and I restored all limits in three batches. Hence `ALLOW_CPU_DECREASE=0` and `MAX_RESIZES_PER_STEP=1`.
* **Memory growth on the kind node** (requests = limits) left pods Pending; memory is now locked (`ALLOW_MEMORY_RESIZE=0`) and the demo script sets it.
* **Overlapping load generators** on the kind cluster made my first kind runs meaningless; they were discarded, only the clean run (73 steps) is used.
* **The `focus` metric** I first used (share of added capacity that went to the bottleneck) was misleading: the agent legitimately also adds to other busy services; it was replaced by the metrics above.

## 8. What this does and does not show
* It shows an agent that reads where the load is - on the demo app for any service, on Train-Ticket for the two services that can really be the bottleneck - and acts only there, with every decision, outcome and failure stored.
* It is **one seed per model**; the simulator scenarios are mine and generous to the agent (the fix always exists, every other service stays below 70 % of its starting capacity).
* Bottlenecks that are **not CPU** (a slow database, a lock) are invisible to it; the gate makes this explicit: no CPU demand, no capacity.
* A CPU resize restarts the pod (about 65-120 s for this application): in this cluster, finding the bottleneck costs a short outage of that service.
* The agent does not scale back down after the load ends in the Train-Ticket run (CPU shrinking is disabled by design there); the operator restores.
