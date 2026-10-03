#!/bin/bash
# Measure real per-replica capacity and print the MAX_CONCURRENCY_PER_REPLICA
# value for your cluster (closes the sim-to-real gap; see README).
#
#   kubectl port-forward deploy/inventory-service 8004:8000 -n autoscaler-poc   # leaf service
#   bash scripts/13_calibrate.sh http://localhost:8004
set -e
cd "$(dirname "$0")/../traffic-generator"
python3 calibrate.py --url "${1:-http://localhost:8004}"
