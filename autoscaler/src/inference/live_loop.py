"""
live_loop.py

Headless control loop against the REAL cluster - the same engine the dashboard uses,
without the web UI. Use this when you want the autoscaler running as a background
process and don't need visualisation.

    python -m src.inference.live_loop --shadow --iterations 20            # observe only
    python -m src.inference.live_loop --iterations 60                     # act for real
    python -m src.inference.live_loop --controller hpa --iterations 60    # run a baseline instead

Safety: every proposed action passes through SafetyGuard (bounds, memory floor,
scale-down cooldown, resize deadband + cooldown, cluster budget, circuit breaker) before
reaching Kubernetes. --shadow logs decisions without applying any of them.
"""

import argparse
import time

from src import config
from src.inference.engines import CONTROLLERS, LiveEngine, load_agent


def _short(s):
    return s.replace("-service", "")


def run(iterations: int, shadow: bool, controller: str = "mo_td3", preference=config.DEFAULT_PREFERENCE,
        arm: str | None = None, out=print):
    agent, name = load_agent(arm)
    engine = LiveEngine(agent, name, shadow=shadow, controller=controller, w=preference)

    mode = "SHADOW (no changes applied)" if shadow else "LIVE (applying real changes)"
    out(f"Autoscaler control loop — {mode} — controller={engine.controller_name}")
    out(f"Polling every {config.POLL_INTERVAL_SECONDS}s\n")
    out(f"{'step':>4} {'p95 ms':>8} {'slo':>6} {'pods':>5} {'cost':>6}  critical path / actions")

    for i in range(iterations):
        p = engine.tick()
        if p.get("stale"):
            out(f'{p["step"]:4d} {"—":>8} {"STALE":>6}  holding: {p.get("stale_reason")}')
        else:
            cp = "->".join(_short(s) for s in p["critical_path"])
            flag = "BREACH" if p["slo_breached"] else "ok"
            out(f'{p["step"]:4d} {p["e2e"]["p95_ms"]:8.1f} {flag:>6} {p["total_replicas"]:5d} '
                f'{p["cost"]:6.2f}  {cp}')
            acts = []
            for n in p["nodes"]:
                bits = []
                if n["applied_delta"]:
                    bits.append(f'{n["applied_delta"]:+d}pod')
                if n["applied_cpu"] != n["cpu"]:
                    bits.append(f'cpu {n["cpu"]:.2f}->{n["applied_cpu"]:.2f}')
                if n["applied_mem"] != n["mem"]:
                    bits.append(f'mem {n["mem"]}->{n["applied_mem"]}')
                if bits:
                    acts.append(f'{_short(n["id"])}: {", ".join(bits)}')
            if acts:
                out(f'       actions: {"; ".join(acts)}')
            blocked = [f'{_short(n["id"])}: {n["blocked_reason"]}' for n in p["nodes"] if n.get("blocked_reason")]
            if blocked:
                out(f'       blocked: {"; ".join(blocked)}')
            if p["breaker_open"]:
                out("       CIRCUIT BREAKER OPEN — using HPA fallback rules")
        if i < iterations - 1:
            time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--controller", choices=CONTROLLERS, default="mo_td3")
    ap.add_argument("--preference", type=float, nargs=config.N_OBJECTIVES, default=list(config.DEFAULT_PREFERENCE))
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    run(args.iterations, args.shadow, args.controller, args.preference, args.arm)
