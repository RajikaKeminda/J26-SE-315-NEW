#!/bin/bash
# Re-creates every number in the README from scratch on the simulator (no cluster needed):
# train all arms (3 seeds for the headline agent and the topology ablations), evaluate every controller,
# trace the Pareto fronts, draw the figures and regenerate the README tables.
#
#   JOBS=4 bash scripts/17_reproduce_results.sh          # ~2-3 h on a laptop with JOBS=4
#   EPISODES=600 SEEDS_FULL="0" bash scripts/17_reproduce_results.sh   # a quick smoke version
#
# (The live-cluster experiment is separate: scripts/13_calibrate.sh and scripts/15_live_experiment.sh.)
set -e
cd "$(dirname "$0")/.."
export JOBS="${JOBS:-4}"
export EPISODES="${EPISODES:-1000}"
SEEDS_FULL="${SEEDS_FULL:-0 1 2}"

echo "=== 1/4 training ==="
SEEDS="$SEEDS_FULL" ARMS="mo_td3_full mo_td3_no_topology mo_td3_sdg_only" bash scripts/06_train_all_arms.sh
SEEDS="0" ARMS="mo_td3_single_obj mo_td3_replica_only mo_td3_no_demo mo_td3_categorical" bash scripts/06_train_all_arms.sh

echo "=== 2/4 evaluation ==="
bash scripts/09_run_ablation.sh "${EVAL_EPISODES:-24}"

echo "=== 3/4 Pareto sweep ==="
bash scripts/12_run_pareto.sh --episodes "${PARETO_EPISODES:-16}"

echo "=== 4/4 figures and README ==="
cd autoscaler
python3 -m src.evaluation.plots
python3 -m src.evaluation.report
