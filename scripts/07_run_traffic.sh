#!/bin/bash
# Sends REAL traffic to the deployed gateway-service. Requires
# scripts/05_port_forward.sh to be running in another terminal.
#
# Usage:
#   scripts/07_run_traffic.sh                     # burst pattern, 2 minutes
#   scripts/07_run_traffic.sh diurnal 300 15       # pattern duration base_rps
set -e
cd "$(dirname "$0")/../traffic-generator"

PATTERN="${1:-burst}"
DURATION="${2:-120}"
BASE_RPS="${3:-20}"

python3 generate_traffic.py --url http://localhost:8000 --pattern "$PATTERN" --duration "$DURATION" --base-rps "$BASE_RPS"
