#!/bin/bash
# Evaluates every trained arm plus the HPA and static baselines on IDENTICAL
# topologies and traffic, then writes results/ablation_results.json.
# This is the head-to-head comparison that supports the research claim.
set -e
cd "$(dirname "$0")/../autoscaler"
python3 -m src.evaluation.evaluate --episodes "${1:-20}"
