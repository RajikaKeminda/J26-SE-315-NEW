#!/bin/bash
# Unit + integration tests for every component. No cluster needed.
#   bash scripts/11_run_tests.sh            # full suite (pytest)
#   bash scripts/11_run_tests.sh -k live    # any pytest args pass through
set -e
cd "$(dirname "$0")/../autoscaler"
python3 -m pytest -q "$@"
