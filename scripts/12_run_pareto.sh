#!/bin/bash
# Preference sweep: ONE trained MO-TD3 network evaluated at 15 preferences over
# (SLO, cost, balance), against the HPA tuning sweep (CPU size x utilisation target), VPA,
# threshold and static. Reports the Pareto-optimal set and the hypervolume of each front.
# Writes autoscaler/results/pareto_results.json (served by the dashboard).
#
#   bash scripts/12_run_pareto.sh                    # defaults (16 episodes per point)
#   bash scripts/12_run_pareto.sh --episodes 32
set -e
cd "$(dirname "$0")/../autoscaler"
python3 -m src.evaluation.pareto "$@"
