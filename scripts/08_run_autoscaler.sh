#!/bin/bash
# Headless autoscaler control loop against the REAL cluster.
# Requires scripts/05_port_forward.sh running in another terminal.
#
#   bash scripts/08_run_autoscaler.sh --shadow --iterations 20   # observe only
#   bash scripts/08_run_autoscaler.sh --iterations 60            # act for real
set -e
cd "$(dirname "$0")/../autoscaler"
python3 -m src.inference.live_loop "$@"
