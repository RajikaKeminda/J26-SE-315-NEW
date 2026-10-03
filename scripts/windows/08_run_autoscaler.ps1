# 08_run_autoscaler.ps1
# Runs the real autoscaler control loop against the live cluster. Requires
# 05_port_forward.ps1 running in another window, and models already
# trained (06_train_models.ps1).
#
# Usage:
#   .\08_run_autoscaler.ps1                          # live mode, 30 iterations
#   .\08_run_autoscaler.ps1 --shadow                 # log actions without applying them
#   .\08_run_autoscaler.ps1 --iterations 60

Set-Location "$PSScriptRoot\..\..\autoscaler"
python -m src.inference.live_loop @args
