"""
generate_traffic.py

Generates REAL, open-loop HTTP traffic against the deployed gateway-service - this is
what gives the autoscaler something to react to.

Arrivals are Poisson at the current target rate and do NOT wait for earlier responses
(closed-loop generators politely slow down when the system struggles, which hides
exactly the overload this project is about). The rate follows one value per control
step (default 10 s), the same curve the simulator trains and is evaluated on:

    # laptop, through a port-forward (needs numpy for the named patterns):
    python generate_traffic.py --pattern burst --duration 400 --base-rps 12 --seed 1000

    # in-cluster (no numpy needed): the caller supplies the per-step curve
    python generate_traffic.py --url http://gateway-service.autoscaler-poc.svc:8000 \
        --curve 12.1,12.0,36.5,... --step-seconds 10 --out /tmp/traffic.json

Per control step it records achieved throughput, errors and the client-observed latency
percentiles - the ground-truth end-to-end latency the live evaluation compares against
the SLO.
"""

import argparse
import asyncio
import json
import random
import time

import httpx


def pattern_curve(pattern: str, steps: int, base_rps: float, seed: int) -> list[float]:
    """The simulator's traffic patterns (same code path => same curve for the same seed)."""
    import numpy as np
    try:
        from src.simulation.simulator import generate_traffic      # repo checkout / autoscaler image
        return [float(v) for v in generate_traffic(pattern, steps, base_rps, np.random.default_rng(seed))]
    except ImportError:
        raise SystemExit("named patterns need the autoscaler package on PYTHONPATH "
                         "(run from autoscaler/ or pass --curve)")


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


async def run(url: str, curve: list[float], step_seconds: float, out: str | None,
              max_inflight: int = 4000, verbose: bool = True):
    steps = [{"lat": [], "ok": 0, "err": 0} for _ in curve]
    sem = asyncio.Semaphore(max_inflight)
    limits = httpx.Limits(max_connections=max_inflight, max_keepalive_connections=256)
    t_start = time.monotonic()
    tasks = []

    async with httpx.AsyncClient(timeout=10.0, limits=limits) as client:
        async def one(step_idx):
            async with sem:
                t0 = time.monotonic()
                try:
                    r = await client.get(f"{url}/work")
                    r.raise_for_status()
                    steps[step_idx]["lat"].append((time.monotonic() - t0) * 1000.0)
                    steps[step_idx]["ok"] += 1
                except httpx.HTTPError:
                    steps[step_idx]["err"] += 1

        next_arrival, reported = t_start, -1
        total = len(curve) * step_seconds
        while True:
            now = time.monotonic()
            elapsed = now - t_start
            if elapsed >= total:
                break
            step_idx = min(int(elapsed // step_seconds), len(curve) - 1)
            if verbose and step_idx != reported and reported >= 0:
                s = steps[reported]
                lat = sorted(s["lat"])
                print(f"step {reported:3d}  target {curve[reported]:6.1f} rps  ok {s['ok']:5d}  err {s['err']:4d}  "
                      f"p50 {_percentile(lat, .5) or 0:7.1f}  p95 {_percentile(lat, .95) or 0:7.1f} ms", flush=True)
            reported = step_idx
            rate = max(curve[step_idx], 0.1)
            delay = next_arrival - now
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(step_idx)))
            next_arrival += random.expovariate(rate)
        await asyncio.gather(*tasks, return_exceptions=True)

    summary = []
    for i, s in enumerate(steps):
        lat = sorted(s["lat"])
        summary.append({"step": i, "target_rps": curve[i], "ok": s["ok"], "errors": s["err"],
                        "achieved_rps": s["ok"] / step_seconds,
                        "p50_ms": _percentile(lat, .5), "p95_ms": _percentile(lat, .95),
                        "p99_ms": _percentile(lat, .99),
                        "mean_ms": (sum(lat) / len(lat)) if lat else None})
    if out:
        with open(out, "w") as f:
            json.dump({"url": url, "step_seconds": step_seconds, "steps": summary}, f, indent=2)
        if verbose:
            print(f"saved -> {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000", help="gateway-service base URL")
    ap.add_argument("--pattern", choices=["normal", "burst", "diurnal", "rampup"], default="burst")
    ap.add_argument("--duration", type=int, default=400, help="seconds to run (named patterns)")
    ap.add_argument("--base-rps", type=float, default=12.0)
    ap.add_argument("--step-seconds", type=float, default=10.0, help="control-step length")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--curve", default=None, help="comma-separated target rps per step (overrides --pattern)")
    ap.add_argument("--out", default=None, help="write per-step stats as JSON")
    args = ap.parse_args()

    if args.curve:
        curve = [float(x) for x in args.curve.split(",")]
    else:
        curve = pattern_curve(args.pattern, max(1, round(args.duration / args.step_seconds)),
                              args.base_rps, args.seed)
    print(f"Generating open-loop '{args.pattern if not args.curve else 'custom'}' traffic against {args.url} "
          f"for {len(curve) * args.step_seconds:.0f}s ({len(curve)} steps, peak {max(curve):.0f} rps)...")
    asyncio.run(run(args.url, curve, args.step_seconds, args.out))
