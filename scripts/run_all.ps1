# Runs the full experiment matrix: every config in configs/exp x 3 seeds, then evaluation.
#
# Runs are sequential on purpose. Each already uses 6 parallel environments, and stacking
# runs on top of that would contend for the same 8 cores and the single Showdown server,
# making wall-clock timings meaningless as a reported result.
#
# The Showdown server must already be running (.\scripts\start_showdown.ps1).
#
# Usage:
#   .\scripts\run_all.ps1                 # full matrix, seeds 0 1 2
#   .\scripts\run_all.ps1 -Seeds 0        # single seed, for a quick pass
#   .\scripts\run_all.ps1 -EvalOnly       # skip training, re-evaluate existing runs

param(
    [int[]]$Seeds = @(0, 1, 2),
    [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

# Fail fast rather than 40 runs deep.
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8000" -TimeoutSec 5 -UseBasicParsing
} catch {
    Write-Error "Showdown server is not responding on http://localhost:8000. Start it with .\scripts\start_showdown.ps1"
}

$configs = Get-ChildItem (Join-Path $root "configs\exp\*.yaml") |
    Where-Object { $_.BaseName -notin @("smoke", "sanity_ppo_masked") }

Write-Host "$($configs.Count) configs x $($Seeds.Count) seeds = $($configs.Count * $Seeds.Count) runs" -ForegroundColor Cyan

# Seeds are the OUTER loop on purpose: if the run is interrupted, every cell has been
# covered at the seeds completed so far, rather than a few cells having all three seeds.
$runDirs = @()
foreach ($seed in $Seeds) {
    foreach ($cfg in $configs) {
        $runId = "$($cfg.BaseName)_seed$seed"
        $runDir = Join-Path $root "results\$runId"
        $runDirs += $runDir

        if (-not $EvalOnly) {
            Write-Host "`n=== TRAIN $runId ===" -ForegroundColor Green
            & $python -m pokerl.train --config $cfg.FullName --seed $seed
            if ($LASTEXITCODE -ne 0) { Write-Warning "Training failed for $runId; continuing."; continue }
        }

        Write-Host "`n=== EVAL $runId ===" -ForegroundColor Green
        & $python -m pokerl.evaluate --run $runDir
        if ($LASTEXITCODE -ne 0) { Write-Warning "Evaluation failed for $runId" }
    }
}

Write-Host "`n=== CROSS-EVALUATION ===" -ForegroundColor Green
$seed0 = $runDirs | Where-Object { $_ -like "*_seed0" }
& $python -m pokerl.evaluate --cross $seed0 --n 100

Write-Host "`n=== ANALYSIS ===" -ForegroundColor Green
& $python (Join-Path $root "scripts\analyse.py")

Write-Host "`nDone. Results in results\, figures in report\figures\." -ForegroundColor Cyan
