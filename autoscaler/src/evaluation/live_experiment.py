"""
live_experiment.py

The head-to-head comparison on a REAL Kubernetes cluster, with ground-truth latency from
an in-cluster, open-loop load generator - and, for every real run, the same run in the
simulator, so the sim-to-real gap of a whole closed-loop controller (not just of one
service) is measured rather than assumed.

For each (controller, traffic pattern, seed):
  1. reset the cluster to the manifest configuration (2 pods x 0.25 core x 128 MiB per service)
     and wait until every rollout has settled;
  2. start the load generator inside the cluster on the SAME per-step traffic curve the
     simulator uses for that seed;
  3. run the control loop once per step (10 s), driving the real Kubernetes API;
  4. read the client-observed latency percentiles per step from the load generator and the
     provisioned cost from the cluster;
  5. run the identical episode in the simulator.

Controllers: static, threshold, hpa (our faithful HPA), hpa_tuned (CPU size + target found
by the held-out grid search), vpa, mo_td3 (the learned agent), and k8s_hpa - the REAL
Kubernetes HorizontalPodAutoscaler controller (needs metrics-server; see
scripts/16_install_metrics_server.sh).

    python -m src.evaluation.live_experiment --controllers static hpa_tuned mo_td3 \
        --patterns burst diurnal --seed 1000

Requires: a cluster with the stack deployed (scripts/03_deploy_all.sh), the in-cluster
autoscaler scaled to 0 (this script IS the control loop), and `kubectl` on PATH.
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import requests

from src import config
from src.evaluation import evaluate
from src.evaluation.metrics import adaptation_time, run_episode
from src.inference.engines import LiveEngine, load_agent
from src.k8s_action import k8s_client
from src.rl_agent.controllers import RLController, RuleController
from src.rl_agent.env import topology_base_rps
from src.simulation.simulator import generate_traffic, slo_for
from src.topologies import DEMO6

NS = config.NAMESPACE
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOADGEN_SCRIPT = os.path.join(REPO, "traffic-generator", "generate_traffic.py")
LOADGEN_MANIFEST = os.path.join(REPO, "k8s", "04-loadgen", "loadgen.yaml")
GATEWAY_URL = f"http://{config.ENTRY_SERVICE}.{NS}.svc.cluster.local:8000"
ALL_CONTROLLERS = ["static", "threshold", "hpa", "hpa_tuned", "vpa", "mo_td3", "k8s_hpa"]


def kubectl(*args, check=True, input=None):
    return subprocess.run(["kubectl", *args], check=check, capture_output=True, text=True, input=input)


# ---------------------------------------------------------------------------
class Forwards:
    """kubectl port-forwards to Prometheus and Jaeger for the duration of the experiment."""

    def __enter__(self):
        self.procs = [
            subprocess.Popen(["kubectl", "port-forward", "-n", NS, f"svc/{svc}", f"{port}:{port}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for svc, port in (("prometheus", 9090), ("jaeger", 16686))]
        for _ in range(60):
            try:
                if requests.get(f"{config.PROMETHEUS_URL}/-/ready", timeout=2).ok:
                    return self
            except requests.RequestException:
                pass
            time.sleep(1)
        self.__exit__(None, None, None)
        raise SystemExit("could not reach Prometheus through a port-forward (is the stack deployed?)")

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()


def ensure_loadgen():
    kubectl("apply", "-f", LOADGEN_MANIFEST)
    kubectl("wait", "--for=condition=ready", "pod/loadgen", "-n", NS, "--timeout=120s")
    kubectl("cp", LOADGEN_SCRIPT, f"{NS}/loadgen:/tmp/generate_traffic.py")


def scale_to(service, replicas):
    apps = k8s_client._load_client()
    apps.patch_namespaced_deployment_scale(name=service, namespace=NS, body={"spec": {"replicas": replicas}})


def reset_cluster(cpu, mem=config.MEM_DEFAULT_MB, replicas=config.INITIAL_REPLICAS, timeout=300, settle=20):
    """Back to the manifest configuration, rolled out and quiet."""
    for svc in config.SERVICES:
        scale_to(svc, replicas)
        k8s_client.set_resources(svc, cpu, mem)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if k8s_client.rollout_complete(config.SERVICES):
            time.sleep(settle)                       # let queues drain and the metric window clear
            return
        time.sleep(3)
    raise SystemExit("cluster did not settle after reset (pods pending? out of node resources?)")


def delete_hpas():
    kubectl("delete", "hpa", "--all", "-n", NS, "--ignore-not-found")


def create_hpas(target_pct=60):
    """The real Kubernetes HPA, configured like the simulated one (scale-down stabilisation 60 s)."""
    docs = []
    for svc in config.SERVICES:
        docs.append(f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {{name: {svc}, namespace: {NS}}}
spec:
  scaleTargetRef: {{apiVersion: apps/v1, kind: Deployment, name: {svc}}}
  minReplicas: {config.MIN_REPLICAS}
  maxReplicas: {config.MAX_REPLICAS}
  metrics:
    - type: Resource
      resource: {{name: cpu, target: {{type: Utilization, averageUtilization: {target_pct}}}}}
  behavior:
    scaleDown: {{stabilizationWindowSeconds: {int(config.HPA_SCALE_DOWN_STABILIZATION_STEPS * config.STEP_SECONDS)}}}
""")
    kubectl("apply", "-f", "-", input="---\n".join(docs))


# ---------------------------------------------------------------------------
def make_engine(controller, agent, name, hpa_best, w):
    """A LiveEngine whose controller is configured for this arm."""
    if controller in ("k8s_hpa", "static"):
        eng = LiveEngine(agent, name, controller="static", w=w)          # observes, never acts
        return eng
    kind = "hpa" if controller == "hpa_tuned" else controller
    eng = LiveEngine(agent, name, controller=kind if kind != "mo_td3" else "mo_td3", w=w)
    if controller == "hpa_tuned":
        eng.controller = RuleController("hpa", eng.services, label="hpa_tuned",
                                        target_utilization=hpa_best["target"])
        eng.controller_name = "hpa"
    return eng


def initial_cpu(controller, hpa_best):
    return hpa_best["cpu"] if controller == "hpa_tuned" else config.CPU_DEFAULT


def run_live(controller, pattern, seed, agent, name, hpa_best, w, steps, log=print):
    cpu0 = initial_cpu(controller, hpa_best)
    delete_hpas()
    reset_cluster(cpu0)
    if controller == "k8s_hpa":
        create_hpas()

    base = topology_base_rps(DEMO6)
    curve = [float(v) for v in generate_traffic(pattern, config.EPISODE_LENGTH_STEPS + 1, base,
                                                np.random.default_rng(seed))[:steps]]
    engine = make_engine(controller, agent, name, hpa_best, w)
    out_path = f"/tmp/live_{controller}_{pattern}_{seed}.json"
    gen = subprocess.Popen(
        ["kubectl", "exec", "-n", NS, "loadgen", "--", "python", "/tmp/generate_traffic.py",
         "--url", GATEWAY_URL, "--curve", ",".join(f"{v:.3f}" for v in curve),
         "--step-seconds", str(config.STEP_SECONDS), "--out", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    slo = slo_for(DEMO6)
    t0, trace = time.time(), []
    for s in range(steps):
        deadline = t0 + (s + 1) * config.STEP_SECONDS
        time.sleep(max(0.0, deadline - time.time()))
        if controller == "k8s_hpa":              # observe only; the HPA controller does the scaling
            state = k8s_client.get_state(config.SERVICES)
            replicas, cost = sum(v["replicas"] for v in state.values()), sum(
                v["replicas"] * (v["cpu"] + config.MEM_COST_PER_GIB * v["mem"] / 1024) for v in state.values())
            p = {"e2e": {"p95_ms": float("nan")}, "total_replicas": replicas, "cost": cost, "stale": False}
        else:
            p = engine.tick()
        trace.append({"step": s, "target_rps": curve[s], "prom_p95_ms": p["e2e"]["p95_ms"],
                      "replicas": p["total_replicas"], "cost": p["cost"], "stale": p.get("stale", False),
                      "controller_actions": sum(1 for n in p.get("nodes", [])
                                                if n["applied_delta"] or n["applied_cpu"] != n["cpu"]
                                                or n["applied_mem"] != n["mem"])})
        log(f"    step {s:2d}  rps {curve[s]:5.1f}  prom p95 {p['e2e']['p95_ms']:7.1f}ms  "
            f"pods {p['total_replicas']:3d}  cost {p['cost']:5.2f}")
    gen.wait(timeout=120)
    if gen.returncode != 0:
        raise SystemExit(f"load generator failed: {gen.stderr.read()}")
    local = f"/tmp/live_{controller}_{pattern}_{seed}_local.json"
    kubectl("cp", f"{NS}/loadgen:{out_path}", local)
    client = json.load(open(local))["steps"]
    delete_hpas()

    p95 = [c["p95_ms"] if c["p95_ms"] is not None else 10_000.0 for c in client]
    ok, err = sum(c["ok"] for c in client), sum(c["errors"] for c in client)
    live = {
        "p50_latency": float(np.mean([c["p50_ms"] or 10_000.0 for c in client])),
        "p95_latency": float(np.mean(p95)),
        "p99_latency": float(np.mean([c["p99_ms"] or 10_000.0 for c in client])),
        "slo_violation_rate": float(np.mean([v > slo for v in p95])),
        "mean_cost": float(np.mean([t["cost"] for t in trace])),
        "mean_replicas": float(np.mean([t["replicas"] for t in trace])),
        "error_rate": err / max(ok + err, 1),
        "adaptation_time_s": adaptation_time(curve, base, p95, slo),
        "slo_ms": slo,
    }
    return live, [{**t, "client_p95_ms": c["p95_ms"], "client_p50_ms": c["p50_ms"],
                   "client_ok": c["ok"], "client_errors": c["errors"]} for t, c in zip(trace, client)]


def run_sim(controller, pattern, seed, agent, hpa_best, w):
    S = DEMO6.services
    if controller == "mo_td3":
        ctrl = RLController(agent, S, w=w)
    elif controller == "hpa_tuned":
        ctrl = RuleController("hpa", S, label="hpa_tuned", initial={"cpu": hpa_best["cpu"]},
                              target_utilization=hpa_best["target"])
    elif controller == "k8s_hpa":
        ctrl = RuleController("hpa", S)                 # the closest simulated analogue
    else:
        ctrl = RuleController(controller, S)
    return run_episode(ctrl, DEMO6, seed, pattern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="*", default=["static", "hpa_tuned", "mo_td3"], choices=ALL_CONTROLLERS)
    ap.add_argument("--patterns", nargs="*", default=["burst", "diurnal"])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=config.EPISODE_LENGTH_STEPS)
    ap.add_argument("--arm", default=None, help="MO-TD3 checkpoint arm (default mo_td3_full)")
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "live_results.json"))
    args = ap.parse_args()

    agent, name = load_agent(args.arm)
    if "mo_td3" in args.controllers and agent is None:
        raise SystemExit("no trained MO-TD3 checkpoint to run")
    hpa_best = None
    if "hpa_tuned" in args.controllers or "mo_td3" in args.controllers:
        res = json.load(open(config.RESULTS_PATH)) if os.path.exists(config.RESULTS_PATH) else None
        hpa_best = res["meta"]["hpa_tuned"] if res else evaluate.tune_hpa(DEMO6)[0]
    w = config.DEFAULT_PREFERENCE

    runs = []
    with Forwards():
        ensure_loadgen()
        for pattern in args.patterns:
            for controller in args.controllers:
                print(f"\n=== {controller} / {pattern} / seed {args.seed} ===", flush=True)
                live, trace = run_live(controller, pattern, args.seed, agent, name, hpa_best, w, args.steps)
                sim = run_sim(controller, pattern, args.seed, agent, hpa_best, w)
                runs.append({"controller": controller, "pattern": pattern, "seed": args.seed,
                             "live": live, "sim": {k: sim[k] for k in live if k in sim}, "trace": trace})
                print(f"  LIVE p95 {live['p95_latency']:7.1f}ms viol {live['slo_violation_rate'] * 100:5.1f}% "
                      f"cost {live['mean_cost']:.2f} | SIM p95 {sim['p95_latency']:7.1f}ms "
                      f"viol {sim['slo_violation_rate'] * 100:5.1f}% cost {sim['mean_cost']:.2f}", flush=True)
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                json.dump({"meta": {"steps": args.steps, "step_seconds": config.STEP_SECONDS,
                                    "slo_ms": slo_for(DEMO6), "hpa_tuned": hpa_best,
                                    "policy": name, "preference": list(w)}, "runs": runs},
                          open(args.out, "w"), indent=2)
    reset_cluster(config.CPU_DEFAULT)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
