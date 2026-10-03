# Trains every ablation arm against the fast simulator. No cluster needed.
param([int]$Iterations = 55)
Set-Location "$PSScriptRoot\..\..\autoscaler"
foreach ($arm in @("rl_full","rl_sdg_only","rl_no_topology","rl_single_obj")) {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host " training arm: $arm  ($Iterations iterations)"
    Write-Host "=============================================="
    python -m src.evaluation.train_arm $arm --iterations $Iterations
}
Write-Host ""
Write-Host "All arms trained. Next: 09_run_ablation.ps1"
