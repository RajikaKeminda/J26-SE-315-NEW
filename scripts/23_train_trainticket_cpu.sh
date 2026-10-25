#!/bin/bash
# Fine-tunes the FULL agent (replicas + CPU; memory held fixed) for the real Train-Ticket's regime and stores the whole run in the history database:
# the real inputs it is built from, every simulated episode, step-by-step trajectories of a sample, the validation curve, the checkpoint's sha256.
#
#   export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler     (bash scripts/22_db_up.sh prints it)
#   bash scripts/23_train_trainticket_cpu.sh                       # 800 episodes, seed 11 -> models/mo_td3_trainticket_cpu.pt
#   EPISODES=1200 SEED=12 OUT=mo_td3_trainticket_cpu2 bash scripts/23_train_trainticket_cpu.sh
#   HOT_PROB=0.9 HOT_VAL=3 OUT=mo_td3_trainticket_hot bash scripts/23_train_trainticket_cpu.sh     # BOTTLENECK-AWARE: in 90 % of the target episodes one random
#                                                       # service is made the bottleneck (topologies.hot_variant), and the checkpoint is chosen on 3 such variants too
#
# Starts from models/mo_td3_large.pt. Regime: profiles/trainticket.env (30 s steps, 95 s pod start-up, 1-2 replicas, CPU 0.25-1.0), the measured
# Train-Ticket simulator parameters and call graph, services starting at random CPU limits. ALLOW_MEMORY_RESIZE=0: the agent learns to size CPU and
# replicas but never memory (the node's memory is the scarce resource). Look at the result: python -m src.persistence.training summary
set -e
cd "$(dirname "$0")/.."
[ -n "$AUTOSCALER_DATABASE_URL" ] || { echo "set AUTOSCALER_DATABASE_URL (bash scripts/22_db_up.sh prints it)"; exit 1; }
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -a; source profiles/trainticket.env; set +a
export ALLOW_MEMORY_RESIZE=0
CLUSTER="${CLUSTER:-train-ticket}"; OUT="${OUT:-mo_td3_trainticket_cpu}"; EPISODES="${EPISODES:-800}"; SEED="${SEED:-11}"
HOT_PROB="${HOT_PROB:-0}"; HOT_VAL="${HOT_VAL:-0}"; FROM="${FROM:-models/mo_td3_large.pt}"; TARGET_PROB="${TARGET_PROB:-0.7}"
cd autoscaler
export SIM_PARAMS_PATH="$PWD/src/simulation/sim_params_trainticket.json"
python -m src.persistence.training register --cluster-name "$CLUSTER"
exec python -u -m src.evaluation.train_agent mo_td3_full --episodes "$EPISODES" --seed "$SEED" --demo-episodes 0 \
  --init-from "$FROM" --sizes 4 14 --target trainticket-sim --target-prob "$TARGET_PROB" --val-sizes 14 --hot-val "$HOT_VAL" \
  --hp actor_lr=5e-5 critic_lr=2e-4 randomize_initial=1 hot_prob="$HOT_PROB" \
  --out "models/$OUT.pt" --curve-out "results/training/$OUT.json" \
  --record --cluster-name "$CLUSTER" --note "replicas + CPU (memory fixed) for the Train-Ticket regime, from $FROM, hot_prob=$HOT_PROB hot_val=$HOT_VAL"
