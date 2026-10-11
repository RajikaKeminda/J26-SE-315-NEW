"""
tt_recorder.py

Read-only recorder for the real Train-Ticket cluster: every `interval` seconds it writes one JSON line with the
per-service measurements the autoscaler sees (request rate, own / inclusive latency percentiles, CPU, memory, pods)
plus the end-to-end latency at the entry service and the node's free memory. It never calls the Kubernetes API with a
write verb and does not run the agent; use it to capture steady-load calibration runs and to slice them afterwards
by timestamp.

    python -m src.evaluation.tt_recorder --out results/trainticket_calibration.jsonl --interval 30
    python -m src.evaluation.tt_recorder --summarize results/trainticket_calibration.jsonl \
        --window 2026-10-11T19:00:00+05:30 2026-10-11T19:10:00+05:30

Timestamps in the file are this machine's clock ("t") and the cluster's clock ("t_cluster", from the Jaeger server's
Date header); slice with either, they differ by the clock skew between the two machines.
"""

import argparse
import datetime as dt
import json
import statistics as st
import time

from src import config
from src.data_collection import jaeger_client, trainticket_metrics


def snapshot() -> dict:
    m = trainticket_metrics.collect()
    try:
        headroom = trainticket_metrics.node_memory_headroom_mb()
    except Exception:
        headroom = None
    now = time.time()
    entry = m.get(config.ENTRY_SERVICE, {})
    return {
        "t": now,
        "t_cluster": now + jaeger_client.clock_skew_s(),
        "node_free_mb": None if headroom is None else round(headroom),
        "e2e": {k: round(entry.get(k, 0.0), 2) for k in ("requests", "incl_latency_ms", "incl_p50_ms", "incl_p95_ms",
                                                         "incl_p99_ms", "error_rate", "trace_samples")},
        "services": {s: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items() if k != "ok"}
                     for s, d in m.items() if d.get("pods_scraped") or d.get("trace_samples")},
    }


def record(out: str, interval: float, duration: float | None):
    start = time.time()
    with open(out, "a") as f:
        while duration is None or time.time() - start < duration:
            t0 = time.time()
            try:
                row = snapshot()
            except Exception as e:                       # tunnel hiccup: keep recording
                row = {"t": t0, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row) + "\n")
            f.flush()
            e2e = row.get("e2e", {})
            print(f"{dt.datetime.fromtimestamp(t0).strftime('%H:%M:%S')}  "
                  f"rps={e2e.get('requests', 0):5.1f}  p95={e2e.get('incl_p95_ms', 0):7.1f} ms  "
                  f"node_free={row.get('node_free_mb')} MB" + (f"  ERROR {row['error']}" if "error" in row else ""),
                  flush=True)
            time.sleep(max(0.0, interval - (time.time() - t0)))


def summarize(path: str, t_from: float = 0.0, t_to: float = float("inf"), clock: str = "t") -> dict:
    """Median per service over the rows whose `clock` timestamp lies in [t_from, t_to]."""
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if "services" in r and t_from <= r[clock] <= t_to]
    out = {"rows": len(rows), "e2e": {}, "services": {}}
    if not rows:
        return out
    for k in rows[0]["e2e"]:
        out["e2e"][k] = round(st.median(r["e2e"].get(k, 0.0) for r in rows), 2)
    names = {s for r in rows for s in r["services"]}
    for s in sorted(names):
        recs = [r["services"][s] for r in rows if s in r["services"]]
        agg = {}
        for k in ("requests", "cpu_cores", "mem_mb", "own_latency_ms", "own_p95_ms", "incl_p95_ms", "pods_scraped"):
            vals = [x.get(k, 0.0) for x in recs]
            agg[k] = round(st.median(vals), 3)
        rps, cpu = agg["requests"], agg["cpu_cores"]
        agg["cpu_ms_per_request"] = round(cpu / rps * 1000, 1) if rps >= 0.5 else None
        out["services"][s] = agg
    return out


def _parse_time(s: str) -> float:
    return dt.datetime.fromisoformat(s).timestamp()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=None, help="seconds; default: until interrupted")
    ap.add_argument("--summarize", metavar="FILE")
    ap.add_argument("--window", nargs=2, metavar=("FROM", "TO"), help="ISO times (with UTC offset)")
    ap.add_argument("--clock", choices=("t", "t_cluster"), default="t")
    a = ap.parse_args()
    if a.summarize:
        lo, hi = (_parse_time(a.window[0]), _parse_time(a.window[1])) if a.window else (0.0, float("inf"))
        print(json.dumps(summarize(a.summarize, lo, hi, a.clock), indent=1))
    else:
        record(a.out or "tt_record.jsonl", a.interval, a.duration)
