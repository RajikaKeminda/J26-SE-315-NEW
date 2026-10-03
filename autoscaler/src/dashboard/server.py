"""
server.py

Live dashboard backend + the autoscaler control engines.

Two modes:
  --mode demo   runs against the built-in simulator. No cluster needed -
                useful for development, demos and presentations.
  --mode live   reads real Prometheus/Jaeger metrics and applies real
                Kubernetes scaling actions (or only logs them with --shadow).

Both modes emit the SAME payload shape, so the frontend is identical.

    python -m src.dashboard.server --mode demo
    open http://localhost:8080

One background task ticks the engine and broadcasts each payload to every
connected WebSocket client, so opening N browser tabs does not run the
control loop N times.

HTTP API
    GET  /healthz                 liveness (always 200 once the app is up)
    GET  /api/state               mode, paused, scenario, policy, last error
    GET  /api/results             ablation results (comparison panel)
    GET  /api/pareto              reward-weight Pareto sweep results
    POST /api/pause | /api/resume pause / resume the control loop
    POST /api/reset               restart the episode (demo) / counters (live)
    POST /api/scenario/{name}     demo only: burst | diurnal | rampup | normal | random
    WS   /ws                      payload stream
"""

import argparse
import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.rl_agent.shared_policy import SharedPolicy
from src.safety.guards import SafetyGuard
from src.baselines.rule_based import HPABaseline
from src.simulation.simulator import (
    Simulator, fixed_topology, generate_traffic, critical_path)
from src.topology_intel.graph_builder import build_all, build_ehg

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
SCENARIOS = ["burst", "diurnal", "rampup", "normal", "random"]

STATE = {
    "mode": "demo", "paused": False, "policy": None, "policy_name": None,
    "engine": None, "latest": None, "error": None, "scenario": "burst",
    "interval": 1.0, "clients": set(), "lock": threading.Lock(),
}


# ---------------------------------------------------------------------------
# Shared: turn raw metrics into the dashboard payload
# ---------------------------------------------------------------------------
def build_payload(services, entry, call_graph, metrics, crit_lat, crit_path,
                   proposed, approved, blocked_reasons, guard, step,
                   rps, history, **extra):
    ehg = build_ehg(services, call_graph, metrics)
    idx = {s: i for i, s in enumerate(services)}

    edges = []
    for caller, targets in call_graph.items():
        for callee, share in targets:
            if caller not in idx or callee not in idx:
                continue
            on_path = (caller in crit_path and callee in crit_path
                       and abs(crit_path.index(caller) - crit_path.index(callee)) == 1)
            edges.append({
                "source": caller,
                "target": callee,
                "traffic_share": round(share, 3),
                "latency_pressure": round(float(ehg[idx[caller], idx[callee]]), 3),
                "on_critical_path": on_path,
            })

    nodes = []
    for s in services:
        m = metrics[s]
        util = m.get("utilization", 0.0)
        nodes.append({
            "id": s,
            "replicas": m["replicas"],
            "latency_ms": round(m["latency_ms"], 1),
            "requests": round(m["requests"], 1),
            "utilization": round(util, 3),
            "on_critical_path": s in crit_path,
            "is_entry": s == entry,
            "proposed_delta": int(proposed.get(s, 0)),
            "applied_delta": int(approved.get(s, 0)),
            "blocked_reason": blocked_reasons.get(s),
            "status": ("critical" if util > 1.0 else "warn" if util > 0.75 else "ok"),
        })

    payload = {
        "step": step,
        "timestamp": time.time(),
        "mode": STATE["mode"],
        "external_rps": round(float(rps), 1),
        "slo_ms": config.SLO_LATENCY_MS,
        "critical_path": crit_path,
        "critical_path_latency_ms": round(crit_lat, 1),
        "slo_breached": crit_lat > config.SLO_LATENCY_MS,
        "breaker_open": guard.breaker_open,
        "total_replicas": sum(m["replicas"] for m in metrics.values()),
        "total_cores": round(sum(m["replicas"] for m in metrics.values()) * config.CPU_CORES_PER_POD, 2),
        "nodes": nodes,
        "edges": edges,
        "history": history,
        "stale": False,
        "paused": STATE["paused"],
        "shadow": False,
        "scenario": STATE["scenario"] if STATE["mode"] == "demo" else None,
        "policy": STATE["policy_name"] or "hpa-fallback",
    }
    payload.update(extra)
    return payload


def _append_history(history, crit_lat, replicas_total, rps):
    history["latency"].append(round(crit_lat, 1))
    history["cost"].append(round(replicas_total * config.CPU_CORES_PER_POD, 2))
    history["rps"].append(round(rps, 1))
    for k in history:
        history[k] = history[k][-60:]


# ---------------------------------------------------------------------------
# Demo engine — simulator driven
# ---------------------------------------------------------------------------
class DemoEngine:
    def __init__(self, policy, scenario="burst", seed=None):
        self.services, self.entry, self.call_graph = fixed_topology()
        self.policy = policy
        self.guard = SafetyGuard(services=self.services)
        self.hpa = HPABaseline(self.services)
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.reset()

    def reset(self, scenario=None):
        if scenario:
            self.scenario = scenario
        self.sim = Simulator(self.services, self.entry, self.call_graph)
        self.traffic = generate_traffic(self.scenario, 200, config.BASE_RPS, self.rng)
        self.step_i = 0
        self.history = {"latency": [], "cost": [], "rps": []}
        self.guard.reset()
        self.hpa.reset()

    def tick(self):
        rps = float(self.traffic[self.step_i % len(self.traffic)])
        metrics = self.sim.step(rps)
        crit_lat, crit_path = critical_path(
            self.call_graph, self.entry, {s: m["latency_ms"] for s, m in metrics.items()})

        self.guard.observe_slo(crit_lat, self.step_i)
        proposed = self._decide(metrics, crit_lat)

        current = {s: metrics[s]["replicas"] for s in self.services}
        approved, reasons = self.guard.filter(proposed, current, self.step_i)
        for svc, d in approved.items():
            if d:
                self.sim.apply(svc, d)

        _append_history(self.history, crit_lat,
                        sum(m["replicas"] for m in metrics.values()), rps)
        payload = build_payload(self.services, self.entry, self.call_graph, metrics,
                                 crit_lat, crit_path, proposed, approved, reasons,
                                 self.guard, self.step_i, rps, self.history)
        self.step_i += 1
        return payload

    # decide: agent normally, HPA rules if the circuit breaker is open / no policy
    def _decide(self, metrics, crit_lat):
        if self.guard.breaker_open or self.policy is None:
            return self.hpa.decide(metrics)
        return self._agent_decide(metrics, crit_lat)

    def _agent_decide(self, metrics, crit_lat):
        X, A_sdg, A_ehg = build_all(self.services, self.call_graph, metrics,
                                     self.entry, config.MAX_REPLICAS)
        total_rep = sum(m["replicas"] for m in metrics.values())
        utils = np.array([m["utilization"] for m in metrics.values()])
        g = torch.tensor([crit_lat / config.SLO_LATENCY_MS,
                          total_rep / (len(self.services) * config.MAX_REPLICAS),
                          min(float(np.var(utils)), 1.0)], dtype=torch.float32)
        with torch.no_grad():
            actions, _, _ = self.policy.act(X, A_sdg, A_ehg, g, deterministic=True)
        return {s: config.REPLICA_DELTA_OPTIONS[int(actions[i])]
                for i, s in enumerate(self.services)}


# ---------------------------------------------------------------------------
# Live engine — real cluster
# ---------------------------------------------------------------------------
REDISCOVER_EVERY_TICKS = 6


def exclusive_latency_estimate(services, call_graph, inclusive):
    """
    Prometheus latency is INCLUSIVE (a caller's histogram contains its
    downstream wait). The simulator and the policy use each service's OWN
    latency, so convert: own = inclusive - sum(share * callee_inclusive).
    Used only when Jaeger has no own-time spans for a service.
    """
    own = {}
    for s in services:
        downstream = sum(w * inclusive.get(c, 0.0) for c, w in call_graph.get(s, []))
        own[s] = max(0.0, inclusive.get(s, 0.0) - downstream)
    return own


def _is_dag(services, graph):
    indeg = {s: 0 for s in services}
    for targets in graph.values():
        for c, _ in targets:
            if c in indeg:
                indeg[c] += 1
    queue, seen = [s for s, d in indeg.items() if d == 0], 0
    while queue:
        n = queue.pop()
        seen += 1
        for c, _ in graph.get(n, []):
            if c in indeg:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
    return seen == len(services)


class LiveEngine(DemoEngine):
    """
    Same decision logic, but state comes from real Prometheus/Jaeger and
    actions go to the real Kubernetes API. Imports are done lazily so demo
    mode works without kubernetes/requests configured.

    Safety behaviour beyond the guard:
      * If Prometheus queries fail, or a Deployment can't be read, the tick
        HOLDS (no scaling, payload flagged stale) rather than reading
        missing data as zero load.
      * `shadow=True` runs everything but never calls the Kubernetes API.
    """

    def __init__(self, policy, shadow=False, autodiscover=None):
        from src.data_collection import prometheus_client, jaeger_client
        from src.k8s_action import k8s_client
        self._prom, self._jaeger, self._k8s = prometheus_client, jaeger_client, k8s_client
        self.shadow = shadow
        self.autodiscover = config.AUTODISCOVER_TOPOLOGY if autodiscover is None else autodiscover
        self.topology_source = "config"
        self._last_ok_ts = None
        self._last_payload = None
        super().__init__(policy)
        if self.autodiscover:
            self._rediscover()

    def reset(self, scenario=None):
        self.step_i = 0
        self.history = {"latency": [], "cost": [], "rps": []}
        self.guard.reset()
        self.hpa.reset()

    # -- topology ------------------------------------------------------
    def _rediscover(self):
        """
        Merge the topology seen in live traces into the known one.

        Conservative on purpose: trace sampling can miss rarely-hit branches,
        so a service/edge absent from the sample is NOT treated as removed
        (dropping a service would silently stop managing it). Newly seen
        services are added; callers with observed outbound calls get their
        edge weights replaced by the observed traffic shares.
        """
        found = self._jaeger.discover_topology(
            self.entry, config.JAEGER_LOOKBACK_SECONDS * 4, config.JAEGER_TRACE_LIMIT * 4)
        if not found:
            return
        new_services, _, new_graph = found
        services = list(self.services) + [s for s in new_services if s not in self.services]
        graph = {s: list(self.call_graph.get(s, [])) for s in services}
        for caller, targets in new_graph.items():
            if targets:
                graph[caller] = [(c, float(w)) for c, w in targets]
        if not _is_dag(services, graph):
            return
        if services != self.services or graph != self.call_graph:
            added = [s for s in services if s not in self.services]
            self.services, self.call_graph = services, graph
            self.guard.services = services
            self.hpa = HPABaseline(services)
            if added:
                print(f"[live] topology discovery added services: {added}")
        self.topology_source = "jaeger"

    # -- stale hold ------------------------------------------------------
    def _hold(self, reason):
        """Payload for a tick where we refuse to act on missing data."""
        step = self.step_i
        self.step_i += 1
        age = None if self._last_ok_ts is None else round(time.time() - self._last_ok_ts, 1)
        if self._last_payload is not None:
            p = json.loads(json.dumps(self._last_payload))
            p.update(step=step, timestamp=time.time())
            for n in p["nodes"]:
                n["proposed_delta"] = n["applied_delta"] = 0
                n["blocked_reason"] = None
            p["nodes"][0]["blocked_reason"] = f"HOLD: {reason}"
        else:
            zero = {s: {"requests": 0.0, "latency_ms": 0.0, "replicas": 1, "utilization": 0.0}
                    for s in self.services}
            p = build_payload(self.services, self.entry, self.call_graph, zero, 0.0,
                              [self.entry], {}, {}, {self.entry: f"HOLD: {reason}"},
                              self.guard, step, 0.0, self.history)
        p.update(stale=True, stale_reason=reason, metrics_age_s=age,
                 shadow=self.shadow, topology_source=self.topology_source)
        return p

    # -- main tick -------------------------------------------------------
    def tick(self):
        if self.autodiscover and self.step_i % REDISCOVER_EVERY_TICKS == 0:
            self._rediscover()

        try:
            prom = self._prom.get_all_service_metrics(self.services)
            replicas = self._k8s.get_replica_counts(self.services)
        except Exception as e:           # e.g. kubeconfig missing, API unreachable
            return self._hold(f"cluster read failed: {e}")

        bad = [s for s in self.services if not prom.get(s, {}).get("ok")]
        if bad:
            return self._hold(f"prometheus unavailable for {', '.join(bad)}")
        missing = [s for s in self.services if s not in replicas]
        if missing:
            return self._hold(f"cannot read deployment(s): {', '.join(missing)}")
        self._last_ok_ts = time.time()

        inclusive = {s: prom[s]["latency_ms"] for s in self.services}
        fallback = exclusive_latency_estimate(self.services, self.call_graph, inclusive)
        own_lat = self._jaeger.get_real_per_service_latency(
            self.services, self.entry, config.JAEGER_LOOKBACK_SECONDS,
            config.JAEGER_TRACE_LIMIT, fallback=fallback)

        metrics = {}
        for s in self.services:
            r = max(1, replicas[s])
            capacity = r * config.MAX_CONCURRENCY_PER_REPLICA * (1000.0 / config.WORK_MS)
            metrics[s] = {
                "requests": prom[s]["requests"],
                "latency_ms": own_lat[s],
                "replicas": r,
                "utilization": prom[s]["requests"] / max(capacity, 1e-6),
            }
        crit_lat, crit_path = critical_path(
            self.call_graph, self.entry, {s: m["latency_ms"] for s, m in metrics.items()})

        self.guard.observe_slo(crit_lat, self.step_i)
        proposed = self._decide(metrics, crit_lat)

        current = {s: metrics[s]["replicas"] for s in self.services}
        approved, reasons = self.guard.filter(proposed, current, self.step_i)
        for svc, d in approved.items():
            if d and not self.shadow:
                try:
                    self._k8s.scale_replicas(svc, d)
                except Exception as e:      # never let a scaling failure kill the loop
                    reasons[svc] = f"k8s error: {e}"
                    approved[svc] = 0

        rps = metrics[self.entry]["requests"]      # external load = what the entry service sees
        _append_history(self.history, crit_lat,
                        sum(m["replicas"] for m in metrics.values()), rps)
        payload = build_payload(self.services, self.entry, self.call_graph, metrics,
                                 crit_lat, crit_path, proposed, approved, reasons,
                                 self.guard, self.step_i, rps, self.history,
                                 shadow=self.shadow, topology_source=self.topology_source)
        self.step_i += 1
        self._last_payload = payload
        return payload


# ---------------------------------------------------------------------------
# Background ticker + routes
# ---------------------------------------------------------------------------
async def _broadcast(payload):
    dead = []
    for ws in list(STATE["clients"]):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE["clients"].discard(ws)


def _locked_tick():
    with STATE["lock"]:
        return STATE["engine"].tick()


async def _ticker():
    while True:
        if not STATE["paused"] and STATE["engine"] is not None:
            try:
                payload = await asyncio.to_thread(_locked_tick)
                payload["paused"] = False
                STATE["latest"], STATE["error"] = payload, None
                await _broadcast(payload)
            except Exception as e:                        # keep the loop alive
                STATE["error"] = f"{type(e).__name__}: {e}"
                print("[ticker] error:", STATE["error"])
        await asyncio.sleep(STATE["interval"])


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(_ticker())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Topology-Aware Autoscaler — Control Room", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/state")
async def state():
    return {
        "mode": STATE["mode"], "paused": STATE["paused"],
        "scenario": STATE["scenario"] if STATE["mode"] == "demo" else None,
        "scenarios": SCENARIOS if STATE["mode"] == "demo" else [],
        "policy": STATE["policy_name"] or "hpa-fallback",
        "shadow": bool(getattr(STATE["engine"], "shadow", False)),
        "step": getattr(STATE["engine"], "step_i", 0),
        "error": STATE["error"],
    }


def _read_json(path):
    if not os.path.exists(path):
        return {"available": False}
    with open(path) as f:
        return {"available": True, "results": json.load(f)}


@app.get("/api/results")
async def results():
    """Serves the ablation study results for the dashboard's comparison panel."""
    return _read_json(config.RESULTS_PATH)


@app.get("/api/pareto")
async def pareto():
    """Serves the reward-weight Pareto sweep (see src/evaluation/pareto.py)."""
    return _read_json(config.PARETO_PATH)


@app.post("/api/pause")
async def pause():
    STATE["paused"] = True
    return {"paused": True}


@app.post("/api/resume")
async def resume():
    STATE["paused"] = False
    return {"paused": False}


@app.post("/api/reset")
async def reset():
    with STATE["lock"]:
        STATE["engine"].reset()
    STATE["latest"] = None
    return {"reset": True}


@app.post("/api/scenario/{name}")
async def scenario(name: str):
    if STATE["mode"] != "demo":
        raise HTTPException(400, "scenarios are only available in demo mode")
    if name not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {name!r}; choose from {SCENARIOS}")
    with STATE["lock"]:
        STATE["engine"].reset(name)
    STATE["scenario"] = name
    STATE["latest"] = None
    return {"scenario": name}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    STATE["clients"].add(websocket)
    try:
        if STATE["latest"] is not None:
            await websocket.send_json(STATE["latest"])
        while True:                      # ticker pushes; we just keep the socket open
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[ws] error:", e)
    finally:
        STATE["clients"].discard(websocket)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def load_policy(arm: str | None = None):
    """Returns the trained policy, or None (-> HPA fallback) if none exists."""
    path = os.path.join(config.MODELS_DIR, f"{arm}.pt") if arm else (
        config.POLICY_PATH if os.path.exists(config.POLICY_PATH)
        else os.path.join(config.MODELS_DIR, f"{config.DEFAULT_ARM}.pt"))
    if not os.path.exists(path):
        print("[server] no trained policy found - falling back to HPA rules.")
        STATE["policy_name"] = None
        return None
    ckpt = torch.load(path, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    policy = SharedPolicy(config.NODE_FEATURE_DIM, config.GCN_HIDDEN_DIM,
                          config.GCN_EMBED_DIM, config.POLICY_HIDDEN,
                          use_ehg=ckpt.get("use_ehg", True),
                          use_topology=ckpt.get("use_topology", True))
    policy.load_state_dict(sd)
    policy.eval()
    STATE["policy_name"] = os.path.splitext(os.path.basename(path))[0]
    print(f"[server] loaded policy from {path}")
    return policy


def configure(mode="demo", shadow=False, arm=None, scenario="burst",
              interval=None, seed=None):
    """Builds the engine and stores it in STATE (used by main() and tests)."""
    STATE.update(mode=mode, paused=False, latest=None, error=None, scenario=scenario)
    policy = load_policy(arm)
    STATE["policy"] = policy
    if mode == "demo":
        STATE["engine"] = DemoEngine(policy, scenario=scenario, seed=seed)
        STATE["interval"] = 1.0 if interval is None else interval
    else:
        STATE["engine"] = LiveEngine(policy, shadow=shadow)
        STATE["interval"] = float(config.POLL_INTERVAL_SECONDS if interval is None else interval)
    return STATE["engine"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--shadow", action="store_true",
                    default=os.environ.get("AUTOSCALER_SHADOW", "0") == "1",
                    help="live mode: compute and show decisions but never scale "
                         "(or set AUTOSCALER_SHADOW=1)")
    ap.add_argument("--arm", default=None, help="serve models/<arm>.pt (default rl_full)")
    ap.add_argument("--scenario", choices=SCENARIOS, default="burst", help="demo traffic")
    ap.add_argument("--interval", type=float, default=None, help="seconds between ticks")
    args = ap.parse_args()

    configure(args.mode, args.shadow, args.arm, args.scenario, args.interval)

    import uvicorn
    print(f"[server] {args.mode} mode{' (shadow)' if args.shadow else ''} "
          f"-> http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
