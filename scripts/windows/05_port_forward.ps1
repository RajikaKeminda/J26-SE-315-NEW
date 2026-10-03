# 05_port_forward.ps1
# Opens port-forwards for everything you need to reach from outside the
# cluster:
#   gateway-service  -> localhost:8000  (traffic generator target)
#   prometheus       -> localhost:9090  (Prometheus UI + autoscaler reads)
#   jaeger           -> localhost:16686 (Jaeger UI + autoscaler reads)
#
# Runs all three as background jobs and stays alive until you press Ctrl+C,
# at which point it stops all three.
#
# NOTE: if you find background jobs unreliable on your machine, the more
# bulletproof alternative is to just open THREE separate PowerShell windows
# and run one command in each (no script needed):
#   kubectl port-forward svc/gateway-service 8000:8000 -n autoscaler-poc
#   kubectl port-forward svc/prometheus 9090:9090 -n autoscaler-poc
#   kubectl port-forward svc/jaeger 16686:16686 -n autoscaler-poc

$Namespace = "autoscaler-poc"

$jobs = @()
$jobs += Start-Job -ScriptBlock { kubectl port-forward svc/gateway-service 8000:8000 -n autoscaler-poc }
$jobs += Start-Job -ScriptBlock { kubectl port-forward svc/prometheus 9090:9090 -n autoscaler-poc }
$jobs += Start-Job -ScriptBlock { kubectl port-forward svc/jaeger 16686:16686 -n autoscaler-poc }

Start-Sleep -Seconds 2

Write-Host "Port-forwards active:"
Write-Host "  gateway-service -> http://localhost:8000"
Write-Host "  Prometheus UI   -> http://localhost:9090"
Write-Host "  Jaeger UI       -> http://localhost:16686"
Write-Host ""
Write-Host "Leave this window running. Press Ctrl+C to stop all port-forwards."

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping port-forwards..."
    $jobs | Stop-Job
    $jobs | Remove-Job
}
