#!/bin/bash
# Head-to-head on the REAL cluster with in-cluster open-loop load: each controller drives the
# real Kubernetes API through the same traffic patterns, latency is measured by the load
# generator, and every run is paired with the identical simulated run (sim-to-real gap).
# Writes autoscaler/results/live_results.json.
#
#   bash scripts/15_live_experiment.sh                                   # static, tuned HPA, MO-TD3
#   CONTROLLERS="static hpa_tuned mo_td3 k8s_hpa" PATTERNS="burst diurnal" bash scripts/15_live_experiment.sh
#
# ~8 minutes per (controller, pattern). The in-cluster autoscaler is scaled to 0 for the duration
# (this script is the control loop) and restored afterwards.
set -e
cd "$(dirname "$0")/../autoscaler"
NS=autoscaler-poc
CONTROLLERS="${CONTROLLERS:-static hpa_tuned mo_td3}"
PATTERNS="${PATTERNS:-burst diurnal}"

kubectl scale deploy/autoscaler -n $NS --replicas=0 2>/dev/null || true
trap 'kubectl scale deploy/autoscaler -n $NS --replicas=1 2>/dev/null || true' EXIT
python3 -m src.evaluation.live_experiment --controllers $CONTROLLERS --patterns $PATTERNS "$@"
