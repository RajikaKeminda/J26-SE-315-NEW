#!/bin/bash
# Evaluates every controller - baselines (static, threshold, HPA, tuned HPA, VPA) and every
# trained MO-TD3 arm - on IDENTICAL traffic and noise, on the 6-service app and on the
# 14-service Train-Ticket-shaped app the agent never saw. Writes
# autoscaler/results/ablation_results.json (served by the dashboard).
#
#   bash scripts/09_run_ablation.sh [episodes]     # default 24
set -e
cd "$(dirname "$0")/../autoscaler"
python3 -m src.evaluation.evaluate --episodes "${1:-24}"
