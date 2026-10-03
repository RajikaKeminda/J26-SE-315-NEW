#!/bin/bash
# Trains every ablation arm (5, incl. the HPA-warm-started rl_hpa_init) against the fast simulator. No cluster needed.
# Each arm is trained separately so you can re-run just one if needed.
#
#   bash scripts/06_train_all_arms.sh            # default iterations
#   bash scripts/06_train_all_arms.sh 100        # more iterations
set -e
cd "$(dirname "$0")/../autoscaler"
ITER="${1:-55}"

for arm in rl_full rl_sdg_only rl_no_topology rl_single_obj rl_hpa_init; do
  echo ""
  echo "=============================================="
  echo " training arm: $arm  ($ITER iterations)"
  echo "=============================================="
  python3 -m src.evaluation.train_arm "$arm" --iterations "$ITER"
done

echo ""
echo "All arms trained -> autoscaler/models/"
echo "Next: bash scripts/09_run_ablation.sh"
