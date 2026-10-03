"""
calibrate.py

Measures the REAL capacity of one service and prints the value to use for
MAX_CONCURRENCY_PER_REPLICA, closing the sim-to-real gap documented in the
README. The simulator assumes each replica sustains
MAX_CONCURRENCY_PER_REPLICA * (1000 / WORK_MS) requests/sec; the real service is
CPU-bound Python behind the GIL on a possibly-throttled pod, so the true number is
usually lower.

Point it at ONE replica of a LEAF service (no downstream calls) so you measure that
replica alone, e.g.:

    kubectl port-forward deploy/inventory-service 8004:8000 -n autoscaler-poc &
    python calibrate.py --url http://localhost:8004 --work-ms 15

Then set MAX_CONCURRENCY_PER_REPLICA (env var) for the autoscaler / simulator.
"""

import argparse
import asyncio
import statistics
import time

import httpx


async def measure(url, concurrency, requests):
    lat = []
    sem = asyncio.Semaphore(concurrency)
    errors = 0
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=concurrency + 8)) as c:
        async def one():
            nonlocal errors
            async with sem:
                t = time.monotonic()
                try:
                    r = await c.get(f"{url}/work")
                    r.raise_for_status()
                    lat.append((time.monotonic() - t) * 1000)
                except httpx.HTTPError:
                    errors += 1
        t0 = time.monotonic()
        await asyncio.gather(*[one() for _ in range(requests)])
        elapsed = time.monotonic() - t0
    return len(lat) / elapsed, errors, lat


async def main(url, work_ms, levels, requests):
    print(f"calibrating {url}  (WORK_MS={work_ms})\n")
    print(f"{'concurrency':>11} {'rps':>8} {'mean ms':>9} {'p95 ms':>9} {'errors':>7}")
    best = 0.0
    for conc in levels:
        rps, errors, lat = await measure(url, conc, requests)
        if not lat:
            print(f"{conc:11d}  all requests failed")
            continue
        p95 = sorted(lat)[int(0.95 * (len(lat) - 1))]
        print(f"{conc:11d} {rps:8.1f} {statistics.mean(lat):9.1f} {p95:9.1f} {errors:7d}")
        best = max(best, rps)
    if best == 0:
        raise SystemExit("no successful measurements")
    effective = best * work_ms / 1000.0
    print(f"\nsaturation throughput : {best:.1f} req/s per replica")
    print(f"effective concurrency : {effective:.2f}   (= rps x WORK_MS)")
    print(f"\nexport MAX_CONCURRENCY_PER_REPLICA={effective:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--work-ms", type=float, default=15.0)
    ap.add_argument("--levels", type=int, nargs="*", default=[1, 4, 12, 32])
    ap.add_argument("--requests", type=int, default=300)
    args = ap.parse_args()
    asyncio.run(main(args.url, args.work_ms, args.levels, args.requests))
