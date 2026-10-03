#!/bin/bash
# Creates a local kind cluster (Docker-in-Docker Kubernetes) for the project.
#   bash scripts/00_kind_up.sh [cluster-name]
# minikube users can skip this: `minikube start --cpus=4 --memory=6144`.
set -e
NAME="${1:-autoscaler}"
if kind get clusters 2>/dev/null | grep -qx "$NAME"; then
  echo "kind cluster '$NAME' already exists"
else
  kind create cluster --name "$NAME" --wait 120s
fi
kubectl config use-context "kind-$NAME"
kubectl get nodes
