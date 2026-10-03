# 01_build_images.ps1
# Builds the two images: generic-service (used by all 6 microservices) and autoscaler.
#
# IMPORTANT: run this in the SAME PowerShell window where you already ran:
#   minikube docker-env | Invoke-Expression
# so the image is built directly into minikube's Docker daemon (no registry
# push needed). See WINDOWS_SETUP.md for details.

Set-Location "$PSScriptRoot\..\.."

Write-Host "Building k8s-autoscaler/generic-service:latest ..."
docker build -t k8s-autoscaler/generic-service:latest services/generic-service

Write-Host "Building k8s-autoscaler/autoscaler:latest ..."
docker build -t k8s-autoscaler/autoscaler:latest autoscaler

Write-Host "Done. Verify with: docker images | Select-String k8s-autoscaler"
