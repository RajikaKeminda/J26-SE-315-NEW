# 03_deploy_all.ps1
# Deploys everything: namespace -> observability (Prometheus, Jaeger) ->
# the 6 microservices -> the in-cluster autoscaler + dashboard.
# Set $env:WITH_AUTOSCALER = "0" to skip the in-cluster autoscaler.

Set-Location "$PSScriptRoot\..\..\k8s"

Write-Host "=== 1/4: namespace ==="
kubectl apply -f 00-namespace.yaml

Write-Host "=== 2/4: observability (Prometheus + Jaeger) ==="
kubectl apply -f 02-observability/

Write-Host "=== 3/4: the 6 microservices ==="
kubectl apply -f 01-services/

if ($env:WITH_AUTOSCALER -ne "0") {
    Write-Host "=== 4/4: autoscaler + dashboard (in-cluster) ==="
    kubectl apply -f 03-autoscaler/
}

Write-Host ""
Write-Host "Deployed. Check status with: kubectl get pods -n autoscaler-poc"
Write-Host "Or run 04_verify_deployment.ps1 to wait for everything to be ready."
