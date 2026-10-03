# Starts the live control-room dashboard, then open http://localhost:8080
#   .\10_run_dashboard.ps1              # demo mode, no cluster needed
#   .\10_run_dashboard.ps1 -Mode live   # real Prometheus/Jaeger/Kubernetes
param([string]$Mode = "demo", [int]$Port = 8080)
Set-Location "$PSScriptRoot\..\..\autoscaler"
python -m src.dashboard.server --mode $Mode --port $Port
