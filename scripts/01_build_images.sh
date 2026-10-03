#!/bin/bash
# Builds the two images:
#   k8s-autoscaler/generic-service  - the ONE image used by all 6 microservices
#   k8s-autoscaler/autoscaler       - the autoscaler + dashboard (in-cluster deployment)
#
# minikube: run `eval $(minikube docker-env)` BEFORE this script so the images are
# built directly into minikube's Docker daemon (no registry push needed).
# kind: build normally, then run scripts/02_load_image_kind.sh.
set -e
cd "$(dirname "$0")/.."

echo "Building k8s-autoscaler/generic-service:latest ..."
docker build -t k8s-autoscaler/generic-service:latest services/generic-service

echo "Building k8s-autoscaler/autoscaler:latest ..."
docker build -t k8s-autoscaler/autoscaler:latest autoscaler

echo "Done. Verify with: docker images | grep k8s-autoscaler"
