#!/bin/bash
# Waits for every Deployment in the namespace to become Ready, then prints
# a summary. Useful right after scripts/03_deploy_all.sh.
set -e
NAMESPACE="autoscaler-poc"

echo "Waiting for all deployments in '$NAMESPACE' to become available (timeout 180s)..."
kubectl wait --for=condition=available --timeout=180s deployment --all -n "$NAMESPACE"

echo
echo "=== Pods ==="
kubectl get pods -n "$NAMESPACE" -o wide

echo
echo "=== Services ==="
kubectl get svc -n "$NAMESPACE"

echo
echo "Everything is up. Next: scripts/05_port_forward.sh (in a separate terminal)"
