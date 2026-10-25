#!/bin/bash
# Trains the BOTTLENECK-AWARE agent: in most episodes one random service is made the only bottleneck (the request mix is shifted towards it and the traffic
# scaled until it needs more CPU than its small starting limit; every other service stays well below its limit), so the only way to score well is to read
# where the load is and add capacity THERE. Fine-tunes from an existing checkpoint, validates on bottleneck scenarios of every service, and stores the run.
#
#   export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler
#   bash scripts/24_train_bottleneck_aware.sh                         # demo-app regime (fast pods), from mo_td3_full  -> models/mo_td3_bn.pt
#   REGIME=trainticket OUT=mo_td3_bn_tt FROM=models/mo_td3_bn.pt bash scripts/24_train_bottleneck_aware.sh    # Train-Ticket regime (95 s pods, CPU only)
#   then:  python -m src.evaluation.bottleneck_eval --topology demo6 --agent bn=models/mo_td3_bn.pt --agent full=models/mo_td3_full.pt
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
REGIME="${REGIME:-demo}"; OUT="${OUT:-mo_td3_bn}"; FROM="${FROM:-models/mo_td3_full.pt}"; EPISODES="${EPISODES:-1500}"; SEED="${SEED:-31}"
HOT_PROB="${HOT_PROB:-0.7}"; HOT_VAL="${HOT_VAL:-6}"; TARGET_ARGS=()
if [ "$REGIME" = trainticket ]; then
  set -a; source profiles/trainticket.env; set +a
  export ALLOW_MEMORY_RESIZE=0 SIM_PARAMS_PATH="$PWD/autoscaler/src/simulation/sim_params_trainticket.json"
  TARGET_ARGS=(--target trainticket-sim --target-prob "${TARGET_PROB:-0.6}")
fi
RECORD=(); [ -n "$AUTOSCALER_DATABASE_URL" ] && RECORD=(--record --cluster-name "${CLUSTER:-train-ticket}" --note "bottleneck-aware ($REGIME) from $FROM, hot_prob=$HOT_PROB")
cd autoscaler
exec python -u -m src.evaluation.train_agent mo_td3_full --episodes "$EPISODES" --seed "$SEED" --demo-episodes 0 \
  --init-from "$FROM" --sizes 4 14 "${TARGET_ARGS[@]}" --val-sizes 8 --hot-val "$HOT_VAL" \
  --hp actor_lr=5e-5 critic_lr=2e-4 randomize_initial=1 hot_prob="$HOT_PROB" val_episodes=48 \
  --out "models/$OUT.pt" --curve-out "results/training/$OUT.json" "${RECORD[@]}"
