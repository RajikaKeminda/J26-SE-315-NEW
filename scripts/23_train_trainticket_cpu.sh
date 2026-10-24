#!/bin/bash
# Fine-tunes the FULL agent (replicas + CPU; memory held fixed) for the real Train-Ticket's regime and stores the whole run in the history database:
# the real inputs it is built from, every simulated episode, step-by-step trajectories of a sample, the validation curve, the checkpoint's sha256.
#
#   export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:autoscaler-local@127.0.0.1:5432/autoscaler     (bash scripts/22_db_up.sh prints it)
#   bash scripts/23_train_trainticket_cpu.sh                       # 800 episodes, seed 11 -> models/mo_td3_trainticket_cpu.pt
#   EPISODES=1200 SEED=12 OUT=mo_td3_trainticket_cpu2 bash scripts/23_train_trainticket_cpu.sh
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
cd autoscaler
export SIM_PARAMS_PATH="$PWD/src/simulation/sim_params_trainticket.json"
python -m src.persistence.training register --cluster-name "$CLUSTER"
exec python -u -m src.evaluation.train_agent mo_td3_full --episodes "$EPISODES" --seed "$SEED" --demo-episodes 0 \
  --init-from models/mo_td3_large.pt --sizes 4 14 --target trainticket-sim --target-prob 0.7 --val-sizes 14 \
  --hp actor_lr=5e-5 critic_lr=2e-4 randomize_initial=1 \
  --out "models/$OUT.pt" --curve-out "results/training/$OUT.json" \
  --record --cluster-name "$CLUSTER" --note "replicas + CPU (memory fixed) for the Train-Ticket regime, from mo_td3_large"
