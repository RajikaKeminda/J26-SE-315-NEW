"""
plots.py

Dependency-free SVG figures for the README, rendered from the committed result files:

    docs/pareto.svg            cost vs. SLO-violation rate: the MO-TD3 preference sweep against the
                               HPA tuning sweep (+ static / VPA / threshold); Pareto-optimal points ringed
    docs/learning_curves.svg   validation score of every training run (convergence behaviour)
    docs/live_vs_sim.svg       real-cluster vs. simulator, per controller and pattern

    python -m src.evaluation.plots
"""

import glob
import json
import os

from src import config

DOCS = os.path.join(os.path.dirname(config.BASE_DIR), "docs")
PALETTE = {"mo_td3": "#7C5CFF", "hpa": "#E08A1E", "static": "#8A94A6", "vpa": "#2BA39B", "threshold": "#C64B6B"}
FONT = "font-family:ui-sans-serif,system-ui,Helvetica,Arial,sans-serif"


def _svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'style="{FONT};background:#fff">{body}</svg>')


def _axes(w, h, pad, xlabel, ylabel, xt, yt):
    l, r, t, b = pad
    out = [f'<rect x="{l}" y="{t}" width="{w - l - r}" height="{h - t - b}" fill="#fafbfc" stroke="#d6dbe3"/>']
    for x, lab in xt:
        out.append(f'<line x1="{x}" y1="{t}" x2="{x}" y2="{h - b}" stroke="#eceff4"/>'
                   f'<text x="{x}" y="{h - b + 16}" font-size="11" fill="#566" text-anchor="middle">{lab}</text>')
    for y, lab in yt:
        out.append(f'<line x1="{l}" y1="{y}" x2="{w - r}" y2="{y}" stroke="#eceff4"/>'
                   f'<text x="{l - 8}" y="{y + 4}" font-size="11" fill="#566" text-anchor="end">{lab}</text>')
    out.append(f'<text x="{(l + w - r) / 2}" y="{h - 8}" font-size="12" fill="#334" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="14" y="{(t + h - b) / 2}" font-size="12" fill="#334" text-anchor="middle" '
               f'transform="rotate(-90 14 {(t + h - b) / 2})">{ylabel}</text>')
    return "".join(out)


def pareto_svg(blob):
    pts = blob["points"]
    w, h, pad = 720, 420, (64, 24, 28, 52)
    xs = [p["slo_violation_rate"] * 100 for p in pts]
    ys = [p["mean_cost"] for p in pts]
    x1, y0, y1 = max(max(xs) * 1.05, 5), 0.0, max(ys) * 1.08
    X = lambda v: pad[0] + v / x1 * (w - pad[0] - pad[1])
    Y = lambda v: h - pad[3] - (v - y0) / (y1 - y0) * (h - pad[2] - pad[3])
    xt = [(X(v), f"{v:.0f}%") for v in [i * x1 / 5 for i in range(6)]]
    yt = [(Y(v), f"{v:.1f}") for v in [y0 + i * (y1 - y0) / 5 for i in range(6)]]
    body = [_axes(w, h, pad, "SLO violation rate (% of control windows with p95 > SLO)",
                  "cost (core-equivalents)", xt, yt)]
    for fam in ("static", "threshold", "vpa", "hpa", "mo_td3"):
        sel = sorted([p for p in pts if p["family"] == fam], key=lambda p: p["slo_violation_rate"])
        for p in sel:
            cx, cy, r = X(p["slo_violation_rate"] * 100), Y(p["mean_cost"]), 5 if fam == "mo_td3" else 4
            ring = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 4}" fill="none" stroke="#111" stroke-width="1.4"/>' \
                if p["pareto_optimal"] else ""
            body.append(f'{ring}<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{PALETTE[fam]}" fill-opacity=".85">'
                        f'<title>{p["name"]}: p95 {p["p95_latency"]:.0f} ms, cost {p["mean_cost"]:.2f}, '
                        f'{p["slo_violation_rate"] * 100:.1f}% violations</title></circle>')
    lx = w - 215
    for i, (fam, lab) in enumerate([("mo_td3", "MO-TD3 (one network, 15 preferences)"), ("hpa", "HPA tuning sweep"),
                                    ("vpa", "VPA"), ("threshold", "threshold rule"), ("static", "static")]):
        body.append(f'<circle cx="{lx}" cy="{44 + i * 18}" r="5" fill="{PALETTE[fam]}"/>'
                    f'<text x="{lx + 12}" y="{48 + i * 18}" font-size="11" fill="#334">{lab}</text>')
    body.append(f'<circle cx="{lx}" cy="{44 + 5 * 18}" r="7" fill="none" stroke="#111" stroke-width="1.4"/>'
                f'<text x="{lx + 12}" y="{48 + 5 * 18}" font-size="11" fill="#334">Pareto-optimal over all</text>')
    body.append('<text x="64" y="18" font-size="13" font-weight="600" fill="#223">Cost vs. SLO violations - lower-left is better</text>')
    return _svg(w, h, "".join(body))


def curves_svg(curves):
    w, h, pad = 720, 360, (64, 24, 28, 52)
    runs = {k: v for k, v in curves.items() if len(v) > 1}
    allv = [r["val_scalarized"] for v in runs.values() for r in v]
    xmax = max(r["episode"] for v in runs.values() for r in v)
    y0, y1 = min(allv) * 0.95, min(max(allv), 0.6)
    X = lambda v: pad[0] + v / xmax * (w - pad[0] - pad[1])
    Y = lambda v: h - pad[3] - (min(v, y1) - y0) / (y1 - y0) * (h - pad[2] - pad[3])
    xt = [(X(v), f"{v:.0f}") for v in [i * xmax / 5 for i in range(6)]]
    yt = [(Y(v), f"{v:.2f}") for v in [y0 + i * (y1 - y0) / 4 for i in range(5)]]
    body = [_axes(w, h, pad, "training episodes", "validation score (lower is better)", xt, yt)]
    colors = ["#7C5CFF", "#E08A1E", "#2BA39B", "#C64B6B", "#3C7BD9", "#8A94A6", "#A3B13A", "#B0568F"]
    for i, (name, v) in enumerate(sorted(runs.items())):
        c = colors[i % len(colors)]
        points = " ".join("%.1f,%.1f" % (X(r["episode"]), Y(r["val_scalarized"])) for r in v)
        body.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" opacity=".9" points="{points}"/>')
        body.append(f'<rect x="{w - 215}" y="{36 + i * 15}" width="12" height="3" fill="{c}"/>'
                    f'<text x="{w - 198}" y="{42 + i * 15}" font-size="10.5" fill="#334">{name.replace("mo_td3_", "")}</text>')
    body.append('<text x="64" y="18" font-size="13" font-weight="600" fill="#223">Training convergence (held-out validation scenarios)</text>')
    return _svg(w, h, "".join(body))


def live_svg(live):
    runs = live["runs"]
    w, h, pad = 720, 360, (64, 24, 40, 70)
    vmax = max(max(r["live"]["p95_latency"], r["sim"].get("p95_latency", 0)) for r in runs) * 1.1
    n = len(runs)
    slot = (w - pad[0] - pad[1]) / max(n, 1)
    Y = lambda v: h - pad[3] - v / vmax * (h - pad[2] - pad[3])
    yt = [(Y(v), f"{v:.0f}") for v in [i * vmax / 5 for i in range(6)]]
    body = [_axes(w, h, pad, "", "mean p95 latency (ms)", [], yt)]
    slo = live["meta"]["slo_ms"]
    body.append(f'<line x1="{pad[0]}" y1="{Y(slo)}" x2="{w - pad[1]}" y2="{Y(slo)}" stroke="#C64B6B" stroke-dasharray="5 4"/>'
                f'<text x="{w - pad[1] - 4}" y="{Y(slo) - 5}" font-size="11" fill="#C64B6B" text-anchor="end">SLO {slo:.0f} ms</text>')
    for i, r in enumerate(runs):
        x = pad[0] + i * slot + slot * 0.18
        bw = slot * 0.28
        for j, (val, col, lab) in enumerate([(r["live"]["p95_latency"], "#7C5CFF", "live"),
                                             (r["sim"].get("p95_latency", 0), "#B9C0CC", "sim")]):
            body.append(f'<rect x="{x + j * bw:.1f}" y="{Y(val):.1f}" width="{bw - 2:.1f}" height="{h - pad[3] - Y(val):.1f}" '
                        f'fill="{col}"><title>{r["controller"]}/{r["pattern"]} {lab}: {val:.0f} ms</title></rect>')
        body.append(f'<text x="{x + bw:.1f}" y="{h - pad[3] + 14}" font-size="10" fill="#334" text-anchor="middle">{r["controller"]}</text>'
                    f'<text x="{x + bw:.1f}" y="{h - pad[3] + 27}" font-size="10" fill="#789" text-anchor="middle">{r["pattern"]}</text>')
    body.append('<rect x="64" y="8" width="10" height="10" fill="#7C5CFF"/><text x="80" y="17" font-size="11" fill="#334">real cluster</text>'
                '<rect x="170" y="8" width="10" height="10" fill="#B9C0CC"/><text x="186" y="17" font-size="11" fill="#334">simulator (same traffic, same controller)</text>')
    return _svg(w, h, "".join(body))


def main():
    os.makedirs(DOCS, exist_ok=True)
    made = []
    if os.path.exists(config.PARETO_PATH):
        open(os.path.join(DOCS, "pareto.svg"), "w").write(pareto_svg(json.load(open(config.PARETO_PATH))))
        made.append("pareto.svg")
    curves = {}
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "training", "*.json"))):
        curves[os.path.splitext(os.path.basename(path))[0]] = json.load(open(path)).get("history", [])
    if curves:
        open(os.path.join(DOCS, "learning_curves.svg"), "w").write(curves_svg(curves))
        made.append("learning_curves.svg")
    live_path = os.path.join(config.RESULTS_DIR, "live_results.json")
    if os.path.exists(live_path):
        open(os.path.join(DOCS, "live_vs_sim.svg"), "w").write(live_svg(json.load(open(live_path))))
        made.append("live_vs_sim.svg")
    print("wrote", ", ".join(made) or "nothing (no result files yet)", "->", DOCS)


if __name__ == "__main__":
    main()
