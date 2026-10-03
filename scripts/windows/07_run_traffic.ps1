# 07_run_traffic.ps1
# Sends REAL open-loop traffic to the deployed gateway-service through a port-forward. Requires
# 05_port_forward.ps1 to be running in another window.
#
# Usage:
#   .\07_run_traffic.ps1                                  # burst, 400s (40 control steps), 12 rps base
#   .\07_run_traffic.ps1 -Pattern diurnal -Duration 200 -BaseRps 12

param(
    [string]$Pattern = "burst",
    [int]$Duration = 400,
    [double]$BaseRps = 12
)

Set-Location "$PSScriptRoot\..\..\autoscaler"
$env:PYTHONPATH = "."
python ..\traffic-generator\generate_traffic.py --url http://localhost:8000 --pattern $Pattern --duration $Duration --base-rps $BaseRps
