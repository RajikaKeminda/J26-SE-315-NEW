"""
auto_profile.py

Point the autoscaler at ANY Kubernetes cluster: give it a kubeconfig/namespace, a Prometheus URL and (for the call graph)
a Jaeger URL, and it works out the settings that were hand-written for Train-Ticket in profiles/trainticket.env.

  entry service     the service whose server spans are the root of most traces            (Jaeger)
  services          Deployments of the namespace that appear in the traces                (Jaeger + Kubernetes)
  metrics source    OpenTelemetry spans if traces exist, else the demo app's Prometheus metrics
  action bounds     from the CURRENT limits (CPU/memory of the Deployments) and the measured memory working sets:
                    never below 1.25x the largest working set, never above what the operator already allows
  replica cap       from the node's free memory (so scale-outs that cannot fit are not even proposed)
  timing            pod start-up measured from the cluster itself (ready time - creation time of real pods),
                    control step = a third of it
  safety defaults   shadow mode, replica scaling only (a resize is a rolling restart), a cost budget of 1.3x the
                    allocation found

It reads only. It writes one env file; scripts/20_autoscale_any_cluster.sh sources it and starts the dashboard.

    python -m src.inference.auto_profile --namespace default --prometheus http://localhost:9090 \
        --jaeger http://localhost:16686 --out profiles/auto-default.env
"""

import argparse
import collections
import math
import statistics as st

import requests

from src.data_collection import otel_traces


# ------------------------------------------------------------------------------------------------
# pure derivations (unit-tested without a cluster)
# ------------------------------------------------------------------------------------------------
def choose_entry(root_counts: dict):
    """The service that is the root of most traces (None when there are no traces)."""
    return max(root_counts, key=root_counts.get) if root_counts else None


def estimate_startup_seconds(durations):
    """Median time from pod creation to Ready over real pods; 4 s (the demo app) when nothing usable was seen."""
    ok = [d for d in durations if 1.0 <= d <= 900.0]
    return float(st.median(ok)) if ok else 4.0


def _ceil_to(x, q):
    return math.ceil(x / q - 1e-9) * q


def derive_bounds(deployments, working_set_mb, node_headroom_mb=None, default_cpu=1.0, default_mem=512.0):
    """Action bounds for the controlled deployments.

    deployments: {name: {"replicas", "cpu" (limit in cores or None), "mem" (limit in MiB or None)}}
    working_set_mb: {name: measured working set per pod}"""
    cpus = [d["cpu"] for d in deployments.values() if d.get("cpu")]
    mems = [d["mem"] for d in deployments.values() if d.get("mem")]
    ws = [working_set_mb[n] for n in deployments if working_set_mb.get(n)]
    cpu_max = _ceil_to(max(cpus), 0.25) if cpus else default_cpu
    cpu_min = max(0.1, _ceil_to(cpu_max / 4.0, 0.05))
    mem_floor = _ceil_to(max(96.0, 1.25 * max(ws)), 64) if ws else 96.0
    mem_max = _ceil_to(max(mems), 64) if mems else max(default_mem, mem_floor * 1.5)
    if mem_max < mem_floor * 1.2:                       # the limits in place leave no room to move: allow a bit more
        mem_max = _ceil_to(mem_floor * 1.5, 64)
    cpu_def = st.median(cpus) if cpus else cpu_max
    mem_def = st.median(mems) if mems else min(mem_max, max(mem_floor, default_mem))
    cap = 4
    if node_headroom_mb is not None and ws:
        cap = int(max(2, min(6, 1 + node_headroom_mb // (2.0 * max(ws)))))
    return {"CPU_MIN": round(cpu_min, 2), "CPU_MAX": round(cpu_max, 2), "CPU_DEFAULT": round(cpu_def, 2),
            "MEM_MIN_MB": int(mem_floor), "MEM_MAX_MB": int(mem_max), "MEM_DEFAULT_MB": int(mem_def),
            "MAX_REPLICAS": cap, "MIN_REPLICAS": 1, "INITIAL_REPLICAS": 1}


def allocation_cost(deployments, mem_cost_per_gib=0.13):
    """Core-equivalents currently allocated (the unit of the agent's cost objective); None if limits are unknown."""
    total = 0.0
    for d in deployments.values():
        if not d.get("cpu") or not d.get("mem"):
            return None
        total += d["replicas"] * (d["cpu"] + mem_cost_per_gib * d["mem"] / 1024.0)
    return total


def derive_timing(startup_s):
    step = min(60, max(10, int(round(startup_s / 3.0 / 5.0)) * 5))
    return {"STEP_SECONDS": step, "POLL_INTERVAL_SECONDS": step, "POD_STARTUP_SECONDS": round(startup_s),
            "ROLLING_SECONDS_PER_REPLICA": round(startup_s), "METRIC_WINDOW": f"{max(30, 2 * step)}s",
            "JAEGER_LOOKBACK_SECONDS": max(30, 2 * step), "JAEGER_TRACE_LIMIT": 200}


# ------------------------------------------------------------------------------------------------
# probing (I/O is injected so the whole discovery can be tested with recorded data)
# ------------------------------------------------------------------------------------------------
def http_json(url, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def prom_query(prometheus, q, get=http_json):
    return get(f"{prometheus}/api/v1/query", {"query": q})["data"]["result"]


def probe_prometheus(prometheus, namespace, get=http_json):
    def scalar(q):
        r = prom_query(prometheus, q, get)
        return float(r[0]["value"][1]) if r else None
    found = {"cadvisor": (scalar(f'count(container_cpu_usage_seconds_total{{namespace="{namespace}"}})') or 0) > 0,
             "kube_state_metrics": (scalar(f'count(kube_deployment_spec_replicas{{namespace="{namespace}"}})') or 0) > 0,
             "demo_request_metrics": (scalar('count(http_requests_total{service!=""})') or 0) > 0,
             "istio_request_metrics": (scalar("count(istio_requests_total)") or 0) > 0}
    alloc = scalar('sum(kube_node_status_allocatable{resource="memory"})')
    used = scalar('sum(container_memory_working_set_bytes{id="/"})')
    found["node_headroom_mb"] = None if alloc is None or used is None else (alloc - used) / 2 ** 20
    ws = {}
    for r in prom_query(prometheus, f'avg by (pod) (container_memory_working_set_bytes{{namespace="{namespace}",pod!="",container=""}})', get):
        pod = r["metric"].get("pod", "")
        parts = pod.rsplit("-", 2)
        if len(parts) == 3:
            ws.setdefault(parts[0], []).append(float(r["value"][1]) / 2 ** 20)
    found["working_set_mb"] = {k: max(v) for k, v in ws.items()}
    return found


def probe_traces(jaeger, deployment_names, get=http_json, per_service=8):
    """-> (root_counts, traced services, one list of recent traces of the most likely entry service)."""
    services = [s for s in get(f"{jaeger}/api/services").get("data", []) if s in deployment_names]
    roots = collections.Counter()
    for svc in services:
        for t in get(f"{jaeger}/api/traces", {"service": svc, "limit": per_service}).get("data", []):
            rows = otel_traces.analyse_trace(t)
            for r in rows:
                if r["parent_service"] is None:
                    roots[r["service"]] += 1
    return dict(roots), services


def probe_cluster(namespace):
    """Deployments (replicas, limits) and measured pod start-up times, through the Kubernetes API (read-only)."""
    from kubernetes import client
    from src.k8s_action import k8s_client
    apps = k8s_client._load_client()
    core = client.CoreV1Api()
    deployments = {}
    for d in apps.list_namespaced_deployment(namespace=namespace).items:
        c = d.spec.template.spec.containers[0]
        lim = (c.resources.limits or {}) if c.resources else {}
        deployments[d.metadata.name] = {"replicas": d.spec.replicas or 0, "ready": d.status.ready_replicas or 0,
                                        "cpu": k8s_client.parse_cpu(lim.get("cpu")) or None,
                                        "mem": k8s_client.parse_mem_mb(lim.get("memory")) or None}
    startups = []
    for p in core.list_namespaced_pod(namespace=namespace).items:
        ready = next((c for c in (p.status.conditions or []) if c.type == "Ready" and c.status == "True"), None)
        if ready and ready.last_transition_time and p.metadata.creation_timestamp:
            startups.append((ready.last_transition_time - p.metadata.creation_timestamp).total_seconds())
    return deployments, startups


def discover(namespace, prometheus, jaeger, entry=None, probes=None):
    """-> (settings: {ENV: value}, report: [lines]). `probes` injects fakes for tests."""
    pr = probes or {}
    k8s = pr.get("cluster", probe_cluster)(namespace)
    deployments, startups = k8s
    prom = pr.get("prometheus", probe_prometheus)(prometheus, namespace)
    report, settings = [], {}
    report.append(f"namespace {namespace!r}: {len(deployments)} Deployments, "
                  f"{sum(1 for d in deployments.values() if d['ready'] > 0)} running")
    report.append("Prometheus: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in prom.items()
                                             if k in ("cadvisor", "kube_state_metrics", "demo_request_metrics", "istio_request_metrics")))
    if not prom["cadvisor"]:
        report.append("WARNING: no cAdvisor metrics in Prometheus - CPU and memory cannot be observed; shadow mode will hold")

    roots, traced = ({}, [])
    if jaeger:
        try:
            roots, traced = pr.get("traces", probe_traces)(jaeger, set(deployments))
        except Exception as e:
            report.append(f"Jaeger unreachable ({e}); no call graph")
    entry = entry or choose_entry(roots)
    if traced and entry:
        settings.update(METRICS_SOURCE="otel", AUTOSCALER_ENTRY_SERVICE=entry, AUTODISCOVER_TOPOLOGY=1)
        report.append(f"traces: {len(traced)} traced services of {len(deployments)}; entry service = {entry} "
                      f"(root of {roots.get(entry, 0)} of {sum(roots.values())} sampled traces)")
    elif prom["demo_request_metrics"]:
        settings.update(METRICS_SOURCE="prometheus")
        report.append("no traces; request metrics found in the demo-app format (http_requests_total{service=...}); "
                      "pass --entry or set AUTOSCALER_TOPOLOGY to describe the call graph")
    else:
        report.append("WARNING: neither traces nor request metrics found - the agent can only observe CPU/memory; "
                      "enable OpenTelemetry tracing (see docs/TRAINTICKET.md) and rerun")
    settings["AUTOSCALER_NAMESPACE"] = namespace
    settings["PROMETHEUS_URL"], settings["JAEGER_QUERY_URL"] = prometheus, jaeger or ""

    controlled = sorted(s for s in traced if deployments.get(s, {}).get("ready", 0) > 0)
    settings["CONTROLLED_SERVICES"] = ",".join(controlled)
    ctl_deps = {s: deployments[s] for s in controlled}
    bounds = derive_bounds(ctl_deps or deployments, prom["working_set_mb"], prom["node_headroom_mb"])
    settings.update(bounds)
    report.append(f"controlled services: {len(controlled)} (traced and running)" if controlled else
                  "controlled services: none (nothing traced) - observe only")
    report.append(f"bounds from current limits/working sets: cpu {bounds['CPU_MIN']}-{bounds['CPU_MAX']} cores, "
                  f"memory {bounds['MEM_MIN_MB']}-{bounds['MEM_MAX_MB']} MiB, replicas {bounds['MIN_REPLICAS']}-{bounds['MAX_REPLICAS']}"
                  + (f" (node has {prom['node_headroom_mb']:.0f} MiB free)" if prom["node_headroom_mb"] is not None else ""))
    startup = estimate_startup_seconds(startups)
    timing = derive_timing(startup)
    settings.update(timing)
    report.append(f"pod start-up measured on {len(startups)} pods: {startup:.0f} s -> control step {timing['STEP_SECONDS']} s")
    cost = allocation_cost(ctl_deps or deployments)
    if cost:
        n = max(1, len(controlled) or len(deployments))
        settings["CLUSTER_BUDGET_CORES"] = round(cost * 1.3, 1)
        settings["BUDGET_PER_SERVICE"] = round(cost * 1.3 / n * 1.2, 2)
        report.append(f"cost budget: {cost:.1f} core-equivalents allocated now -> budget {settings['CLUSTER_BUDGET_CORES']}")
    else:
        report.append("WARNING: some Deployments have no CPU/memory limits - cost cannot be derived; default budget used")
    settings["ALLOW_RESIZE"] = 0
    settings["OBSERVE_UNTRACED"] = 1
    settings["NODE_MEMORY_RESERVE_MB"] = 768 if (prom["node_headroom_mb"] or 0) < 4096 else 1024
    report.append("safety defaults: shadow mode, replica scaling only (ALLOW_RESIZE=0), node-memory reserve "
                  f"{settings['NODE_MEMORY_RESERVE_MB']} MiB, circuit breaker + guard as in training")
    return settings, report


def render_env(settings, header=""):
    lines = [f"# {l}" for l in header.splitlines()] if header else []
    lines += [f"{k}={v}" for k, v in settings.items() if v not in ("", None)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--prometheus", default="http://localhost:9090")
    ap.add_argument("--jaeger", default="http://localhost:16686")
    ap.add_argument("--entry", default=None, help="entry service (default: detected from traces)")
    ap.add_argument("--out", default=None, help="write an env file here")
    a = ap.parse_args()
    settings, report = discover(a.namespace, a.prometheus, a.jaeger, a.entry)
    print("\n".join(f"  {l}" for l in report))
    if a.out:
        open(a.out, "w").write(render_env(settings, "Generated by src/inference/auto_profile.py - review before use.\n"
                                                    "Source with:  set -a; source <file>; set +a"))
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
