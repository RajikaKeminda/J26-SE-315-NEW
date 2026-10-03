#!/bin/bash
# ONLY needed if you're using kind (not minikube). kind runs Docker in an
# isolated environment, so images built on your host must be explicitly
# loaded into the kind cluster's node.
#
# minikube users: skip this script entirely - instead run
#   eval $(minikube docker-env)
# BEFORE scripts/01_build_images.sh, so the images are built directly where
# minikube can see them.
set -e

CLUSTER_NAME="${1:-autoscaler}"
echo "Loading images into kind cluster '$CLUSTER_NAME'..."
kind load docker-image k8s-autoscaler/generic-service:latest k8s-autoscaler/autoscaler:latest --name "$CLUSTER_NAME"
echo "Done."
