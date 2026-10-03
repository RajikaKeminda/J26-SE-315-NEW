#!/bin/bash
# Reward-weight sweep: re-trains rl_full at several W_COST values and plots each
# against HPA and static on the (latency, cost, SLO-violation) Pareto front.
# ~1.5 min per weight on a laptop CPU. Writes autoscaler/results/pareto_results.json.
#
#   bash scripts/12_run_pareto.sh                    # defaults (55 iterations)
#   bash scripts/12_run_pareto.sh --iterations 100
set -e
cd "$(dirname "$0")/../autoscaler"
python3 -m src.evaluation.pareto "$@"
