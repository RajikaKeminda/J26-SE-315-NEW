# Trains the MO-TD3 agent and every ablation arm against the calibrated simulator. No cluster needed.
#   .\06_train_all_arms.ps1                 # 1500 episodes, seed 0
#   .\06_train_all_arms.ps1 -Episodes 800 -Seeds 0,1,2
param([int]$Episodes = 1000, [int[]]$Seeds = @(0))
Set-Location "$PSScriptRoot\..\..\autoscaler"
$arms = @("mo_td3_full","mo_td3_sdg_only","mo_td3_no_topology","mo_td3_single_obj","mo_td3_replica_only","mo_td3_no_demo","mo_td3_categorical")
foreach ($seed in $Seeds) {
    foreach ($arm in $arms) {
        Write-Host ""
        Write-Host "=============================================="
        Write-Host " training arm: $arm  (seed $seed, $Episodes episodes)"
        Write-Host "=============================================="
        python -u -m src.evaluation.train_agent $arm --episodes $Episodes --seed $seed
    }
}
Write-Host ""
Write-Host "All arms trained. Next: 09_run_ablation.ps1"
