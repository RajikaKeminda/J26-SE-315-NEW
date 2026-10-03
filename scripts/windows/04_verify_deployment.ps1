# 04_verify_deployment.ps1
# Waits for every Deployment in the namespace to become Ready, then prints
# a summary.

$Namespace = "autoscaler-poc"

Write-Host "Waiting for all deployments in '$Namespace' to become available (timeout 180s)..."
kubectl wait --for=condition=available --timeout=180s deployment --all -n $Namespace

Write-Host ""
Write-Host "=== Pods ==="
kubectl get pods -n $Namespace -o wide

Write-Host ""
Write-Host "=== Services ==="
kubectl get svc -n $Namespace

Write-Host ""
Write-Host "Everything is up. Next: 05_port_forward.ps1 (in a separate PowerShell window)"
