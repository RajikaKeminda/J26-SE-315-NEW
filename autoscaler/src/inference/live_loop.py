"""
live_loop.py

Headless production control loop against the REAL cluster - the same engine
the dashboard uses, without the web UI. Use this when you want the
autoscaler running as a background process and don't need visualisation.

    python -m src.inference.live_loop --shadow --iterations 20   # observe only
    python -m src.inference.live_loop --iterations 60            # act for real

Safety: every proposed action passes through SafetyGuard (bounds, rate
limit, scale-down cooldown, circuit breaker) before reaching Kubernetes.
--shadow logs decisions without applying any of them.
"""

import argparse
import time

from src import config
from src.dashboard.server import LiveEngine, load_policy


def run(iterations: int, shadow: bool):
    policy = load_policy()
    engine = LiveEngine(policy, shadow=shadow)

    mode = "SHADOW (no changes applied)" if shadow else "LIVE (applying real scaling)"
    print(f"Autoscaler control loop — {mode}")
    print(f"Polling every {config.POLL_INTERVAL_SECONDS}s\n")
    print(f"{'step':>4} {'lat_ms':>8} {'slo':>5} {'reps':>5}  critical path / actions")

    for i in range(iterations):
        payload = engine.tick()
        if payload.get("stale"):
            print(f'{payload["step"]:4d} {"—":>8} {"STALE":>6}  holding: {payload.get("stale_reason")}')
            if i < iterations - 1:
                time.sleep(config.POLL_INTERVAL_SECONDS)
            continue
        cp = "->".join(s.replace("-service", "") for s in payload["critical_path"])
        acts = [f'{n["id"].replace("-service","")}{n["applied_delta"]:+d}'
                for n in payload["nodes"] if n["applied_delta"]]
        blocked = [f'{n["id"].replace("-service","")}: {n["blocked_reason"]}'
                   for n in payload["nodes"] if n.get("blocked_reason")]

        flag = "BREACH" if payload["slo_breached"] else "ok"
        print(f'{payload["step"]:4d} {payload["critical_path_latency_ms"]:8.1f} {flag:>6} '
              f'{payload["total_replicas"]:5d}  {cp}')
        if acts:
            print(f'       actions: {", ".join(acts)}')
        if blocked:
            print(f'       blocked: {"; ".join(blocked)}')
        if payload["breaker_open"]:
            print(f'       CIRCUIT BREAKER OPEN — using HPA fallback rules')

        if i < iterations - 1:
            time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--shadow", action="store_true")
    args = ap.parse_args()
    run(args.iterations, args.shadow)
