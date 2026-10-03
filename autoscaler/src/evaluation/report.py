"""
report.py

Renders the committed result files into the tables of the README, so the documentation
cannot drift away from the data (the previous version quoted numbers by hand).

    python -m src.evaluation.report           # rewrite the generated blocks in README.md
    python -m src.evaluation.report --check   # exit 1 if README.md is out of date (used by a test)

Generated blocks are delimited by  <!-- BEGIN:name -->  /  <!-- END:name -->.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

from src import config

README = os.path.join(os.path.dirname(config.BASE_DIR), "README.md")
LABELS = {
    "static": "static (no scaling)", "threshold": "threshold rule (80 % / 30 %)",
    "hpa": "HPA (default 60 % target, 0.25-core pods)", "hpa_tuned": "**HPA, tuned** (CPU size × target, held-out grid)",
    "vpa": "VPA (right-sizing only)",
    "mo_td3_full": "**MO-TD3 full** (dual-graph GCN, hybrid actions)",
    "mo_td3_sdg_only": "MO-TD3, SDG branch only", "mo_td3_no_topology": "MO-TD3, no topology (no GCN)",
    "mo_td3_single_obj": "MO-TD3, single objective (latency only)", "mo_td3_replica_only": "MO-TD3, replicas only (no CPU/mem actions)",
    "mo_td3_no_demo": "MO-TD3, no demonstrations",
    "mo_td3_categorical": "MO-TD3, categorical replica head (straight-through)",
}


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _row(name, r, best):
    cells = []
    for key, fmt in (("p95_latency", "{:.0f}"), ("slo_violation_rate", "{:.1%}"), ("mean_cost", "{:.2f}"),
                     ("adaptation_time_s", "{:.0f}"), ("scalarized", "{:.3f}")):
        txt = fmt.format(r[key])
        cells.append(f"**{txt}**" if abs(r[key] - best[key]) < 1e-12 else txt)
    return f"| {LABELS.get(name, name)} | " + " | ".join(cells) + " |"


def comparison_table(rows):
    keys = ("p95_latency", "slo_violation_rate", "mean_cost", "adaptation_time_s", "scalarized")
    best = {k: min(r[k] for r in rows.values()) for k in keys}
    head = ("| controller | p95 ms | SLO violations | cost | adapt s | score |\n"
            "|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(_row(n, r, best) for n, r in rows.items())


def block_results(res):
    if not res:
        return "_(no results yet - run `bash scripts/09_run_ablation.sh`)_"
    m = res["meta"]
    out = [f"{m['episodes']} matched-seed episodes (burst / diurnal / ramp-up / normal), default preference "
           f"(SLO, cost, balance) = {tuple(m['preference'])}. Lower is better everywhere; best per column in bold.",
           f"The tuned HPA is the best (CPU size, utilisation target) found on a held-out seed range: "
           f"**{m['hpa_tuned']['cpu']} cores, {m['hpa_tuned']['target']:.0%} target**.", "",
           "**6-service application (the one deployed on the cluster and calibrated against it):**", "",
           comparison_table(res["demo6"])]
    seeds = res.get("demo6_seeds") or {}
    if seeds:
        out += ["", "**Across training seeds** (mean ± std; one RL run is an anecdote):", "",
                "| arm | seeds | p95 ms | SLO violations | cost | score |", "|---|---|---|---|---|---|"]
        for arm, d in seeds.items():
            f = lambda k, fmt: f"{fmt.format(np.mean(d[k]))} ± {fmt.format(np.std(d[k]))}"
            out.append(f"| {LABELS.get(arm, arm)} | {len(d['seeds'])} | {f('p95_latency', '{:.0f}')} | "
                       f"{f('slo_violation_rate', '{:.1%}')} | {f('mean_cost', '{:.2f}')} | {f('scalarized', '{:.3f}')} |")
    if res.get("trainticket14"):
        out += ["", "**14-service application with Train-Ticket's call structure - never seen in training:**", "",
                comparison_table(res["trainticket14"])]
    return "\n".join(out)


def block_pareto(blob):
    if not blob:
        return "_(no sweep yet - run `bash scripts/12_run_pareto.sh`)_"
    pts = blob["points"]
    fam = blob["families"]
    lines = [f"One MO-TD3 network evaluated at {sum(p['family'] == 'mo_td3' for p in pts)} preferences vs. "
             f"{sum(p['family'] == 'hpa' for p in pts)} HPA configurations (CPU size × target), plus static / VPA / threshold. "
             f"Non-dominated points over all controllers on (SLO-violation rate, p95, cost):", "",
             "| point | p95 ms | SLO violations | cost | Pareto-optimal |", "|---|---|---|---|---|"]
    for p in sorted([p for p in pts if p["pareto_optimal"]], key=lambda p: p["mean_cost"]):
        lines.append(f"| {p['name']} | {p['p95_latency']:.0f} | {p['slo_violation_rate']:.1%} | {p['mean_cost']:.2f} | yes |")
    lines += ["", f"Hypervolume in (violation rate, cost/budget), larger is better: "
              f"**MO-TD3 {fam['mo_td3']['hypervolume']:.3f}** ({fam['mo_td3']['n_pareto_optimal']} of {fam['mo_td3']['n_points']} points Pareto-optimal) "
              f"vs. **HPA tuning sweep {fam['hpa']['hypervolume']:.3f}** ({fam['hpa']['n_pareto_optimal']} of {fam['hpa']['n_points']})."]
    return "\n".join(lines)


def block_calibration(cal):
    if not cal:
        return "_(no calibration yet - run `bash scripts/13_calibrate.sh`)_"
    rep, caps = cal["fit_report"], cal["capacity_check"]
    st, ho = rep.get("steady (fitted region)"), rep.get("near-saturation (held out)")
    lines = ["| CPU limit | measured saturation (req/s) | model (req/s) |", "|---|---|---|"]
    for cpu, c in caps.items():
        lines.append(f"| {float(cpu):.2f} core | {c['observed_max_rps']:.1f} | {c['model_capacity_rps']:.1f} |")
    lines += ["", f"Latency model vs. {cal['n_points']} measurements: steady region (n={st['n']}) mean-latency MAPE "
              f"{st['mean_MAPE_%']:.0f} %, p95 MAPE {st['p95_MAPE_%']:.0f} %, {st['p95_within_2x']:.0%} of p95 predictions within 2×"
              + (f"; near saturation (n={ho['n']}, held out) p95 MAPE {ho['p95_MAPE_%']:.0f} %." if ho else ".")]
    return "\n".join(lines)


def block_live(live):
    if not live:
        return "_(no live run yet - run `bash scripts/15_live_experiment.sh`)_"
    m = live["meta"]
    lines = [f"Real kind cluster, {m['steps']} control steps of {m['step_seconds']:.0f} s per run, client-observed latency from an "
             f"in-cluster open-loop load generator, SLO = {m['slo_ms']:.0f} ms p95. Each real run is paired with the identical simulated run.", "",
             "| pattern | controller | live p95 ms | sim p95 ms | live SLO viol. | sim SLO viol. | live cost | sim cost |",
             "|---|---|---|---|---|---|---|---|"]
    for r in live["runs"]:
        l, s = r["live"], r["sim"]
        lines.append(f"| {r['pattern']} | {LABELS.get(r['controller'], r['controller'])} | {l['p95_latency']:.0f} | "
                     f"{s.get('p95_latency', float('nan')):.0f} | {l['slo_violation_rate']:.1%} | "
                     f"{s.get('slo_violation_rate', float('nan')):.1%} | {l['mean_cost']:.2f} | {s.get('mean_cost', float('nan')):.2f} |")
    return "\n".join(lines)


def render():
    return {
        "results": block_results(_load(config.RESULTS_PATH)),
        "pareto": block_pareto(_load(config.PARETO_PATH)),
        "calibration": block_calibration(_load(config.SIM_PARAMS_PATH)),
        "live": block_live(_load(os.path.join(config.RESULTS_DIR, "live_results.json"))),
    }


def apply(text, blocks):
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md has no <!-- BEGIN:{name} --> ... <!-- END:{name} --> block")
        text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    old = open(README).read()
    new = apply(old, render())
    if args.check:
        if new != old:
            print("README.md is out of date with results/*.json - run: python -m src.evaluation.report")
            sys.exit(1)
        print("README.md matches the committed results")
        return
    open(README, "w").write(new)
    print("README.md updated")


if __name__ == "__main__":
    main()
