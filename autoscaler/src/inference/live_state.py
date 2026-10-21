"""
live_state.py

Turns REAL measurements (Prometheus + the Kubernetes API) into the exact `metrics`
schema the simulator produces, so the agent - trained in simulation - reads the same
features on a real cluster.

The mapping (real signal -> feature):

    utilisation       rate(process_cpu_seconds_total) / (ready pods x CPU limit)   [<= 1: pods are
                      throttled at their limit, exactly like the simulator's saturation]
    own latency       service_own_duration_seconds mean / p95 (queueing + CPU, no downstream)
    inclusive p95     http_request_duration_seconds p95 (the callee's cost to its caller; at
                      the entry service it is the end-to-end latency a client experiences)
    memory util.      process_resident_memory_bytes / memory limit
    pending replicas  desired - ready (pods starting, or a rolling update in flight)
    work_ms           CPU ms per request, estimated online from cpu / requests (EMA)
    offered load      derived from the above by graph_builder (needs no extra signal)
"""

from src import config
from src.simulation import service_model as sm
from src.safety.guards import pod_cost
from src.topology_intel.graph_builder import offered_load


class LiveStateBuilder:
    """`priors` = the declared CPU cost per request of each service (topology work_ms). The online
    estimate may only move within WORK_BAND of the prior and only from a *steady* window - a window
    with a saturated pod, pods still starting, or a rollout in flight burns CPU that is not
    attributable to the requests completed in it (start-up imports, queue drain) and would otherwise
    poison the estimate for the rest of the run."""

    WORK_BAND = (0.6, 1.8)
    UNKNOWN_BAND = (0.1, 12.0)

    def __init__(self, services, entry, call_graph, ema=0.3, priors=None):
        self.services, self.entry, self.call_graph = list(services), entry, call_graph
        self.ema = ema
        self.priors = dict(priors or {})
        self.work_est = {s: self.priors.get(s, config.WORK_MS) for s in services}
        self.prev_rps = {s: None for s in services}

    def set_topology(self, services, entry, call_graph):
        for s in services:
            self.work_est.setdefault(s, self.priors.get(s, config.WORK_MS))
            self.prev_rps.setdefault(s, None)
        self.services, self.entry, self.call_graph = list(services), entry, call_graph

    def build(self, prom: dict, k8s: dict, edges: dict | None = None, min_edge_samples: int = 3):
        """-> (metrics {svc: {...}}, e2e {...}, cost). `prom`: prometheus_client.collect();
        `k8s`: k8s_client.get_state(); `edges`: jaeger_client.aggregate_window() - per-edge
        latency percentiles from traces (the EHG), used where enough samples exist."""
        p = sm.params()
        metrics = {}
        for s in self.services:
            m, k = prom.get(s, {}), k8s[s]
            ready = max(1, k["ready"])
            rps = m.get("requests", 0.0)

            # online estimate of the CPU cost of one request: only from a steady, unsaturated window
            known = s in self.priors
            prior = self.priors.get(s, config.WORK_MS)
            raw_util = m.get("cpu_cores", 0.0) / max(ready * k["cpu"], 1e-6)
            steady = (rps >= 3.0 and raw_util < 0.85 and k["ready"] == k["replicas"] and k["ready"] > 0)
            if steady:
                w = (m["cpu_cores"] - p.idle_cores * ready) / rps * 1000.0 - p.cpu_overhead_ms
                band = self.WORK_BAND if known else self.UNKNOWN_BAND      # no measurement of this app yet: trust the data
                lo, hi = band[0] * prior, band[1] * prior
                if lo <= w <= hi:
                    self.work_est[s] += self.ema * (w - self.work_est[s])
            else:                                                       # drift back toward the declared cost
                self.work_est[s] += 0.05 * (prior - self.work_est[s])
            work = self.work_est[s]

            if rps >= 0.5 and m.get("own_latency_ms", 0.0) > 0:
                own = m["own_latency_ms"]
                own_p95 = m.get("own_p95_ms") or own * sm.tail_factor(0.5, p)
            else:        # no traffic: an idle replica's latency, from the fitted model
                own = sm.base_latency_ms(k["cpu"], work, p)
                own_p95 = own * p.tail_idle
            incl = m.get("incl_latency_ms", 0.0) or own
            incl_p95 = m.get("incl_p95_ms", 0.0) or incl * 1.5

            util = min(1.0, m.get("cpu_cores", 0.0) / max(ready * k["cpu"], 1e-6))
            metrics[s] = {
                "requests": rps,
                "rps_prev": self.prev_rps[s] if self.prev_rps[s] is not None else rps,
                "latency_ms": own,
                "latency_p95_ms": max(own_p95, own),
                "utilization": util,
                "replicas": k["replicas"],
                "ready_replicas": ready,
                "pending_replicas": max(0, k["replicas"] - k["ready"]),
                "cpu": k["cpu"],
                "mem": k["mem"],
                "mem_utilization": m.get("mem_mb", 0.0) / max(k["mem"], 1.0),
                "inflight": m.get("inflight", 0.0),
                "work_ms": work,
                "error_rate": m.get("error_rate", 0.0),
                "backlog": 0.0,
                "youngest_pod_s": k.get("youngest_pod_s"),
                "latency_incl_ms": incl,
                "latency_incl_p95_ms": max(incl_p95, incl),
            }
            self.prev_rps[s] = rps
            metrics[s]["edge_p95"] = {}
            metrics[s]["offered_load"] = offered_load(metrics[s])

        for (caller, callee), st in (edges or {}).items():
            if caller in metrics and callee in metrics and st["count"] >= min_edge_samples:
                metrics[caller]["edge_p95"][callee] = st["p95_ms"]

        entry = prom.get(self.entry, {})
        em = metrics[self.entry]
        e2e = {
            "mean_ms": em["latency_incl_ms"],
            "p50_ms": entry.get("incl_p50_ms") or em["latency_incl_ms"],
            "p95_ms": em["latency_incl_p95_ms"],
            "p99_ms": entry.get("incl_p99_ms") or em["latency_incl_p95_ms"] * p.p99_over_p95,
            "error_rate": em["error_rate"],
        }
        cost = sum(m["replicas"] * pod_cost(m["cpu"], m["mem"]) for m in metrics.values())
        return metrics, e2e, cost
