#!/bin/bash
# Trains the MO-TD3 agent and every ablation arm on the calibrated simulator. No cluster needed.
#
#   bash scripts/06_train_all_arms.sh                 # 1 seed, 1000 episodes
#   SEEDS="0 1 2" EPISODES=1000 bash scripts/06_train_all_arms.sh
#   ARMS="mo_td3_full" bash scripts/06_train_all_arms.sh      # just the headline agent
#   JOBS=4 bash scripts/06_train_all_arms.sh          # run 4 trainings in parallel (1 CPU thread each)
#
# Arms: mo_td3_full (dual-graph GCN, hybrid actions, multi-objective), mo_td3_sdg_only,
# mo_td3_no_topology, mo_td3_single_obj, mo_td3_replica_only, mo_td3_no_demo, mo_td3_categorical.
# Seed 0 -> autoscaler/models/<arm>.pt, other seeds -> autoscaler/models/seeds/<arm>_s<k>.pt.
#
# Memory note: each training process holds a replay buffer (~150-300 MB). On an 8 GB laptop that also runs
# a Docker VM, JOBS=4..5 is the practical ceiling before the machine starts swapping.
set -e
cd "$(dirname "$0")/../autoscaler"
export PY="${PYTHON:-python3}"
export EPISODES="${EPISODES:-${1:-1000}}"
SEEDS="${SEEDS:-0}"
JOBS="${JOBS:-1}"
ARMS="${ARMS:-mo_td3_full mo_td3_sdg_only mo_td3_no_topology mo_td3_single_obj mo_td3_replica_only mo_td3_no_demo mo_td3_categorical}"

# one "arm seed" pair per line -> xargs runs them JOBS at a time (portable: macOS ships bash 3.2, which has no `wait -n`)
for seed in $SEEDS; do for arm in $ARMS; do echo "$arm $seed"; done; done |
  xargs -P "$JOBS" -L 1 bash -c '
    arm="$0"; seed="$1"
    echo "==> training $arm (seed $seed, $EPISODES episodes)"
    if [ "'"$JOBS"'" -gt 1 ]; then
      "$PY" -u -m src.evaluation.train_agent "$arm" --episodes "$EPISODES" --seed "$seed" > "/tmp/train_${arm}_s${seed}.log" 2>&1
      tail -3 "/tmp/train_${arm}_s${seed}.log"
    else
      "$PY" -u -m src.evaluation.train_agent "$arm" --episodes "$EPISODES" --seed "$seed"
    fi'

echo ""
echo "All arms trained -> autoscaler/models/"
echo "Next: bash scripts/09_run_ablation.sh"
