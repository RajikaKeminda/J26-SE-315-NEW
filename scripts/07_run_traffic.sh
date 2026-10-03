#!/bin/bash
# Sends REAL open-loop traffic to the deployed gateway-service through a port-forward.
# Requires scripts/05_port_forward.sh running in another terminal.
# (For measurements use the in-cluster generator instead: scripts/15_live_experiment.sh.)
#
# Usage:
#   scripts/07_run_traffic.sh                      # burst pattern, 400 s (40 control steps)
#   scripts/07_run_traffic.sh diurnal 200 12       # pattern duration base_rps
set -e
cd "$(dirname "$0")/../autoscaler"
PATTERN="${1:-burst}"
DURATION="${2:-400}"
BASE_RPS="${3:-12}"
PYTHONPATH=. python3 ../traffic-generator/generate_traffic.py --url http://localhost:8000 \
  --pattern "$PATTERN" --duration "$DURATION" --base-rps "$BASE_RPS"
