#!/bin/bash
# Starts the live control-room dashboard.
#
#   bash scripts/10_run_dashboard.sh                      # demo mode, no cluster needed
#   bash scripts/10_run_dashboard.sh live                 # real Prometheus/Jaeger/K8s
#   bash scripts/10_run_dashboard.sh live 8080 --shadow   # live, but never scale
#   bash scripts/10_run_dashboard.sh demo 8080 --scenario diurnal
#
# Then open http://localhost:8080
#
# live mode (from your laptop) requires scripts/05_port_forward.sh running in
# another terminal, or deploy the autoscaler in-cluster (scripts/03_deploy_all.sh).
set -e
cd "$(dirname "$0")/../autoscaler"
MODE="${1:-demo}"; PORT="${2:-8080}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true
python3 -m src.dashboard.server --mode "$MODE" --port "$PORT" "$@"
