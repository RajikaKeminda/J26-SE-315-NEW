"""
calibrate.py

Measures how ONE replica of the generic service behaves under open-loop
(Poisson) load, so the simulator's service model can be fitted to - and
validated against - the real thing.

For each offered rate it records achieved throughput, latency percentiles and
the CPU time the process actually burnt per request (from the service's own
`process_cpu_seconds_total` metric). Run it against a LEAF service so that you
measure that replica alone:

    # from inside the cluster (recommended - no port-forward in the path):
    kubectl cp traffic-generator/calibrate.py calib/<loadgen-pod>:/tmp/calibrate.py
    kubectl exec -n calib <loadgen-pod> -- python /tmp/calibrate.py \
        --url http://leaf.calib.svc:8000 --rates 5 10 15 20 25 --duration 20 --out /tmp/c.json

    # or from your laptop through a port-forward:
    kubectl port-forward deploy/inventory-service 8004:8000 -n autoscaler-poc &
    python calibrate.py --url http://localhost:8004

The output JSON is consumed by autoscaler/src/simulation/calibration.py.
"""

import argparse
import asyncio
import json
import random
import re
import time

import httpx


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


async def _scrape(client, url):
    """(process_cpu_seconds_total, process_resident_memory_bytes) or (None, None)."""
    try:
        text = (await client.get(f"{url}/metrics", timeout=5)).text
    except httpx.HTTPError:
        return None, None
    cpu = re.search(r"^process_cpu_seconds_total\s+([0-9.eE+-]+)", text, re.M)
    rss = re.search(r"^process_resident_memory_bytes\s+([0-9.eE+-]+)", text, re.M)
    return (float(cpu.group(1)) if cpu else None, float(rss.group(1)) if rss else None)


async def run_rate(url, rate, duration, max_inflight=2000):
    """Open-loop: arrivals are Poisson(rate) and do NOT wait for earlier responses."""
    lat, errors = [], 0
    sem = asyncio.Semaphore(max_inflight)
    limits = httpx.Limits(max_connections=max_inflight, max_keepalive_connections=200)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        cpu0, _ = await _scrape(client, url)

        async def one():
            nonlocal errors
            async with sem:
                t = time.monotonic()
                try:
                    r = await client.get(f"{url}/work")
                    r.raise_for_status()
                    lat.append((time.monotonic() - t) * 1000.0)
                except httpx.HTTPError:
                    errors += 1

        tasks, t0 = [], time.monotonic()
        next_arrival = t0
        while next_arrival - t0 < duration:
            delay = next_arrival - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one()))
            next_arrival += random.expovariate(rate)
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - t0
        cpu1, rss1 = await _scrape(client, url)

    lat.sort()
    n = len(lat)
    return {
        "offered_rps": rate,
        "achieved_rps": n / elapsed,
        "n": n,
        "errors": errors,
        "mean_ms": sum(lat) / n if n else None,
        "p50_ms": _percentile(lat, 0.50),
        "p95_ms": _percentile(lat, 0.95),
        "p99_ms": _percentile(lat, 0.99),
        "cpu_ms_per_request": ((cpu1 - cpu0) / n * 1000.0) if (n and cpu0 is not None and cpu1 is not None) else None,
        "cpu_cores_used": ((cpu1 - cpu0) / elapsed) if (cpu0 is not None and cpu1 is not None) else None,
        "rss_mb": (rss1 / 2**20) if rss1 is not None else None,
    }


async def main(args):
    rows = []
    print(f"calibrating {args.url}  ({args.duration}s per rate, open-loop Poisson)\n")
    print(f"{'offered':>8} {'achieved':>9} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} "
          f"{'err':>5} {'cpu ms/req':>11} {'cores':>6}")
    # warm-up so JIT-ish effects (imports, connection pools) don't skew the first row
    await run_rate(args.url, 2.0, 3)
    for rate in args.rates:
        r = await run_rate(args.url, rate, args.duration)
        rows.append(r)
        cpu = f"{r['cpu_ms_per_request']:.1f}" if r["cpu_ms_per_request"] is not None else "-"
        cores = f"{r['cpu_cores_used']:.2f}" if r["cpu_cores_used"] is not None else "-"
        print(f"{rate:8.1f} {r['achieved_rps']:9.1f} {r['mean_ms']:8.1f} {r['p50_ms']:8.1f} "
              f"{r['p95_ms']:8.1f} {r['p99_ms']:8.1f} {r['errors']:5d} {cpu:>11} {cores:>6}")
        await asyncio.sleep(args.cooldown)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"url": args.url, "label": args.label, "duration_s": args.duration,
                       "rows": rows}, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8004")
    ap.add_argument("--rates", type=float, nargs="*", default=[5, 10, 15, 20, 30, 40])
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    asyncio.run(main(ap.parse_args()))
