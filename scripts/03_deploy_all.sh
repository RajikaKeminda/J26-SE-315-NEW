#!/bin/bash
# Deploys everything: namespace -> observability (Prometheus, Jaeger) -> the 6
# microservices -> the in-cluster autoscaler + dashboard.
#
#   bash scripts/03_deploy_all.sh                     # everything
#   WITH_AUTOSCALER=0 bash scripts/03_deploy_all.sh   # skip the in-cluster autoscaler
#                                                     # (run it from your laptop instead)
set -e
cd "$(dirname "$0")/../k8s"

echo "=== 1/4: namespace ==="
kubectl apply -f 00-namespace.yaml

echo "=== 2/4: observability (Prometheus + Jaeger) ==="
kubectl apply -f 02-observability/

echo "=== 3/4: the 6 microservices ==="
kubectl apply -f 01-services/

if [ "${WITH_AUTOSCALER:-1}" = "1" ]; then
  echo "=== 4/4: autoscaler + dashboard (in-cluster, RBAC-scoped) ==="
  kubectl apply -f 03-autoscaler/
else
  echo "=== 4/4: skipped (WITH_AUTOSCALER=0) ==="
fi

echo
echo "Deployed. Check status with: kubectl get pods -n autoscaler-poc"
echo "Or run scripts/04_verify_deployment.sh to wait for everything to be ready."
