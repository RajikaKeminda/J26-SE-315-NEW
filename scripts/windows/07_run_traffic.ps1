# 07_run_traffic.ps1
# Sends REAL traffic to the deployed gateway-service. Requires
# 05_port_forward.ps1 to be running in another window.
#
# Usage:
#   .\07_run_traffic.ps1                                  # burst, 120s, 20 rps
#   .\07_run_traffic.ps1 -Pattern diurnal -Duration 300 -BaseRps 15

param(
    [string]$Pattern = "burst",
    [int]$Duration = 120,
    [double]$BaseRps = 20
)

Set-Location "$PSScriptRoot\..\..\traffic-generator"
python generate_traffic.py --url http://localhost:8000 --pattern $Pattern --duration $Duration --base-rps $BaseRps
