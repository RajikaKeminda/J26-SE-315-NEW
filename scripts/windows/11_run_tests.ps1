# Unit + integration tests for every component. No cluster needed.
Set-Location "$PSScriptRoot\..\..\autoscaler"
python -m pytest -q
