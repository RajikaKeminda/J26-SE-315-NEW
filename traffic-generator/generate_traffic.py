"""
generate_traffic.py

Generates REAL HTTP traffic against the actual deployed gateway-service -
this is what makes the whole system have something to autoscale against.

By default targets http://localhost:8000 (matches
`kubectl port-forward svc/gateway-service 8000:8000 -n autoscaler-poc`
- see the project README). Point --url elsewhere for a NodePort/Ingress/
cloud LoadBalancer setup instead.
"""

import argparse
import asyncio
import time

import httpx
import numpy as np


def traffic_curve(pattern: str, duration_seconds: int, base_rps: float) -> list[float]:
    """Returns a target requests/sec value for each 1-second tick of the run."""
    t = np.arange(duration_seconds)
    if pattern == "normal":
        return list(np.clip(base_rps + np.random.normal(0, base_rps * 0.1, duration_seconds), 1, None))
    if pattern == "burst":
        curve = np.full(duration_seconds, base_rps, dtype=float)
        start, end = duration_seconds // 3, duration_seconds // 3 + duration_seconds // 4
        curve[start:end] = base_rps * 6
        return list(np.clip(curve, 1, None))
    if pattern == "diurnal":
        cycle = 2 * np.pi * t / duration_seconds
        return list(np.clip(base_rps * (1.5 + 1.2 * np.sin(cycle - np.pi / 2)), 1, None))
    raise ValueError(f"unknown pattern: {pattern}")


async def fire_requests_for_one_second(client: httpx.AsyncClient, url: str, rps: float, stats: dict):
    """Fires approximately `rps` requests spread across one second."""
    count = max(1, round(rps))
    interval = 1.0 / count

    async def one_request():
        try:
            resp = await client.get(f"{url}/work", timeout=10.0)
            stats["success"] += 1
            stats["latencies_ms"].append(resp.elapsed.total_seconds() * 1000)
        except httpx.HTTPError:
            stats["errors"] += 1

    tasks = []
    for i in range(count):
        tasks.append(asyncio.create_task(one_request()))
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)


async def run(url: str, pattern: str, duration_seconds: int, base_rps: float):
    curve = traffic_curve(pattern, duration_seconds, base_rps)
    print(f"Generating '{pattern}' traffic against {url} for {duration_seconds}s "
          f"(base_rps={base_rps})...")

    async with httpx.AsyncClient() as client:
        for second, target_rps in enumerate(curve):
            stats = {"success": 0, "errors": 0, "latencies_ms": []}
            start = time.monotonic()
            await fire_requests_for_one_second(client, url, target_rps, stats)
            elapsed = time.monotonic() - start

            avg_latency = sum(stats["latencies_ms"]) / len(stats["latencies_ms"]) if stats["latencies_ms"] else 0.0
            print(f"t={second:4d}s target_rps={target_rps:6.1f} "
                  f"ok={stats['success']:4d} err={stats['errors']:3d} "
                  f"avg_latency={avg_latency:7.1f}ms (took {elapsed:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="gateway-service base URL")
    parser.add_argument("--pattern", choices=["normal", "burst", "diurnal"], default="burst")
    parser.add_argument("--duration", type=int, default=120, help="seconds to run")
    parser.add_argument("--base-rps", type=float, default=20.0)
    args = parser.parse_args()

    asyncio.run(run(args.url, args.pattern, args.duration, args.base_rps))
