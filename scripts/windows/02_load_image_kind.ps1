# 02_load_image_kind.ps1
# ONLY needed if you're using kind (not minikube). kind runs Docker in an
# isolated environment, so images built on your host must be explicitly
# loaded into the kind cluster's node.
#
# minikube users: skip this entirely - instead run
#   minikube docker-env | Invoke-Expression
# BEFORE 01_build_images.ps1, so the image is built directly where
# minikube can see it.

param(
    [string]$ClusterName = "autoscaler"
)

Write-Host "Loading image into kind cluster '$ClusterName'..."
kind load docker-image k8s-autoscaler/generic-service:latest k8s-autoscaler/autoscaler:latest --name $ClusterName
Write-Host "Done."
