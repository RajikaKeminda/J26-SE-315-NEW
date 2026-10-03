# Head-to-head comparison of every arm plus HPA and static baselines.
param([int]$Episodes = 20)
Set-Location "$PSScriptRoot\..\..\autoscaler"
python -m src.evaluation.evaluate --episodes $Episodes
