# Head-to-head comparison of every controller (baselines + every trained MO-TD3 arm).
param([int]$Episodes = 24)
Set-Location "$PSScriptRoot\..\..\autoscaler"
python -m src.evaluation.evaluate --episodes $Episodes
