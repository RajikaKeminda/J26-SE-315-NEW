"""
calibration.py

Fits the service model (service_model.py) to real measurements and reports how
well it predicts them - the sim-to-real evidence for this project.

    python -m src.simulation.calibration                # report only
    python -m src.simulation.calibration --write        # also write sim_params.json

Input: calibration/leaf_cpu*.json, one file per CPU limit, produced by
traffic-generator/calibrate.py running open-loop load against a single replica of a
leaf service on a real cluster.

Fit procedure (deliberately small - 5 free parameters, ~36 points):
  * cpu_overhead_ms  from the CPU-per-request measured near saturation (where fixed
                     per-process overhead is amortised),
  * idle_cores       from CPU burnt at the lowest offered rates,
  * base_extra_ms, kappa, tail_idle, tail_busy by least squares on log-latency
                     of mean and p95, restricted to the steady-state region
                     (rho <= 0.92: a 15 s run at saturation never reaches steady state).
The report gives MAPE and the fraction of predictions within +-30 % / a factor of
two, on the fitted region AND on the held-out near-saturation points.
"""

import argparse
import glob
import json
import math
import os

import numpy as np

from src import config
from src.simulation import service_model as sm

DEFAULT_DIR = os.path.join(os.path.dirname(config.BASE_DIR), "calibration")
STEADY_RHO_MAX = 0.92


def load_points(directory: str = DEFAULT_DIR, work_ms: float = config.WORK_MS):
    """Rows -> list of dicts with the CPU limit and measured utilisation attached."""
    pts = []
    for path in sorted(glob.glob(os.path.join(directory, "leaf_cpu*.json"))):
        blob = json.load(open(path))
        cpu = float(os.path.basename(path).split("cpu")[1].split(".")[0].rstrip("m")) / 1000.0
        for r in blob["rows"]:
            if not r.get("mean_ms"):
                continue
            pts.append({"cpu": cpu, "rps": r["achieved_rps"], "mean": r["mean_ms"],
                        "p50": r["p50_ms"], "p95": r["p95_ms"], "p99": r["p99_ms"],
                        "cpu_ms_per_req": r.get("cpu_ms_per_request"),
                        "cores": r.get("cpu_cores_used"), "rss": r.get("rss_mb"),
                        "errors": r.get("errors", 0)})
    return pts


def _rho(pt, p, work_ms):
    return pt["rps"] / sm.capacity_rps(pt["cpu"], work_ms, p)


def predict(pt, p, work_ms=config.WORK_MS):
    rho = _rho(pt, p, work_ms)
    mean = sm.mean_latency_ms(rho, pt["cpu"], work_ms, p)
    return rho, mean, mean * sm.tail_factor(rho, p)


def _loss(x, pts, base, work_ms):
    p = sm.ModelParams(**{**base.__dict__, "base_extra_ms": x[0], "kappa": x[1],
                          "tail_idle": x[2], "tail_busy": x[3]})
    err = []
    for pt in pts:
        rho, mean, p95 = predict(pt, p, work_ms)
        if rho > STEADY_RHO_MAX:
            continue
        err.append(math.log(mean / pt["mean"]))
        err.append(math.log(p95 / max(pt["p95"], 1e-3)))
    return float(np.mean(np.square(err))) if err else 1e9


def fit(pts, work_ms=config.WORK_MS):
    base = sm.ModelParams()
    # --- directly measurable constants -------------------------------------------------
    # CPU burnt = idle + marginal_cost * rps, so regress on the unsaturated points
    # (cores is capped by the CPU limit once a replica is throttled).
    lin = [(pt["rps"], pt["cores"]) for pt in pts
           if pt["cores"] is not None and pt["cores"] < 0.9 * pt["cpu"]]
    if len(lin) >= 4:
        xs, ys = np.array([a for a, _ in lin]), np.array([b for _, b in lin])
        slope, intercept = np.polyfit(xs, ys, 1)
        base.cpu_overhead_ms = max(0.0, slope * 1000.0 - work_ms)
        base.idle_cores = float(np.clip(intercept, 0.0, 0.05))
    rss = [pt["rss"] for pt in pts if pt["rss"]]
    if rss:
        base.mem_base_mb = float(np.median(rss))

    # --- least squares on the latency shape (no scipy: coordinate search, deterministic) --
    x = np.array([base.base_extra_ms, base.kappa, base.tail_idle, base.tail_busy])
    lo = np.array([0.0, 0.2, 1.0, 1.5])
    hi = np.array([40.0, 1.5, 2.0, 4.0])
    step = np.array([4.0, 0.2, 0.2, 0.5])
    best = _loss(x, pts, base, work_ms)
    for _ in range(60):
        improved = False
        for i in range(len(x)):
            for sgn in (+1, -1):
                y = x.copy()
                y[i] = np.clip(y[i] + sgn * step[i], lo[i], hi[i])
                v = _loss(y, pts, base, work_ms)
                if v < best - 1e-9:
                    x, best, improved = y, v, True
        if not improved:
            step *= 0.5
            if step.max() < 1e-3:
                break
    base.base_extra_ms, base.kappa, base.tail_idle, base.tail_busy = [float(v) for v in x]
    return base


def evaluate(pts, p, work_ms=config.WORK_MS):
    out = {}
    for name, sel in (("steady (fitted region)", lambda r: r <= STEADY_RHO_MAX),
                      ("near-saturation (held out)", lambda r: r > STEADY_RHO_MAX)):
        rows = [(pt, *predict(pt, p, work_ms)) for pt in pts]
        rows = [(pt, r, m, q) for pt, r, m, q in rows if sel(r)]
        if not rows:
            continue
        mean_ratio = np.array([m / pt["mean"] for pt, _, m, _ in rows])
        p95_ratio = np.array([q / max(pt["p95"], 1e-3) for pt, _, _, q in rows])
        out[name] = {
            "n": len(rows),
            "mean_MAPE_%": float(np.mean(np.abs(mean_ratio - 1)) * 100),
            "p95_MAPE_%": float(np.mean(np.abs(p95_ratio - 1)) * 100),
            "mean_within_30%": float(np.mean(np.abs(mean_ratio - 1) <= 0.30)),
            "p95_within_30%": float(np.mean(np.abs(p95_ratio - 1) <= 0.30)),
            "p95_within_2x": float(np.mean((p95_ratio >= 0.5) & (p95_ratio <= 2.0))),
        }
    return out


def capacity_check(pts, p, work_ms=config.WORK_MS):
    """Predicted vs observed saturation throughput per CPU limit."""
    res = {}
    for cpu in sorted({pt["cpu"] for pt in pts}):
        observed = max(pt["rps"] for pt in pts if pt["cpu"] == cpu)
        res[cpu] = {"observed_max_rps": observed,
                    "model_capacity_rps": sm.capacity_rps(cpu, work_ms, p)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--write", action="store_true", help="write src/simulation/sim_params.json")
    args = ap.parse_args()

    pts = load_points(args.dir)
    if not pts:
        raise SystemExit(f"no calibration data in {args.dir}")
    p = fit(pts)
    report = evaluate(pts, p)
    caps = capacity_check(pts, p)

    print(f"calibration points: {len(pts)} from {len({pt['cpu'] for pt in pts})} CPU limits\n")
    print("fitted parameters:")
    for k, v in p.__dict__.items():
        print(f"  {k:22s} {v:.3f}")
    print("\nprediction error vs. measurements:")
    for name, r in report.items():
        print(f"  {name:28s} n={r['n']:2d}  mean MAPE {r['mean_MAPE_%']:5.1f}%  p95 MAPE {r['p95_MAPE_%']:5.1f}%  "
              f"p95 within 30%: {r['p95_within_30%']*100:3.0f}%  within 2x: {r['p95_within_2x']*100:3.0f}%")
    print("\nsaturation throughput per replica (req/s):")
    for cpu, c in caps.items():
        print(f"  cpu limit {cpu:4.2f}: observed >= {c['observed_max_rps']:5.1f}   model {c['model_capacity_rps']:5.1f}")

    if args.write:
        os.makedirs(os.path.dirname(config.SIM_PARAMS_PATH), exist_ok=True)
        with open(config.SIM_PARAMS_PATH, "w") as f:
            json.dump({"params": p.__dict__, "fit_report": report, "capacity_check":
                       {str(k): v for k, v in caps.items()},
                       "source": "calibration/leaf_cpu*.json", "n_points": len(pts)}, f, indent=2)
        print(f"\nwrote {config.SIM_PARAMS_PATH}")


if __name__ == "__main__":
    main()
