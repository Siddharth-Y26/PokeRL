# Launch the three coach training arms concurrently.
#
# These are not study cells -- see COACH_ARMS in scripts/gen_configs.py for why they are kept
# out of the experiment matrix. Each runs 5M steps with the v1 encoder.
#
# Concurrency notes:
#   * Runs are staggered so that account registration does not collide. poke-env mints
#     usernames from uuid4 (see _unique_accounts in env/battle_env.py), but three processes
#     opening 8 environments each in the same instant still makes a burst the server handles
#     better spread out.
#   * The Node simulator is the throughput ceiling, not the GPU or the learner, so
#     config/config.js is set to network: 2 / simulator: 2 for this batch.
#   * torch_threads stays at 1. On this machine the learner otherwise burns ~7 cores on a
#     small MLP and starves the env workers that are actually on the critical path.
#
# Usage:  .\scripts\train_coach_arms.ps1
#         .\scripts\train_coach_arms.ps1 -Seed 1

param(
    [int]$Seed = 0,
    [string[]]$Arms = @("ppo_v1_long", "ppo_v1_selfplay", "ppo_v1_entropy"),
    [int]$StaggerSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) { Write-Error "venv python not found at $python" }

try {
    $null = Invoke-WebRequest -Uri "http://localhost:8000" -TimeoutSec 5 -UseBasicParsing
} catch {
    Write-Error "Showdown does not appear to be running. Start it with .\scripts\start_showdown.ps1"
}

$logDir = Join-Path $root "results\_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$procs = @()
foreach ($arm in $Arms) {
    $cfg = Join-Path $root "configs\coach\$arm.yaml"
    if (-not (Test-Path $cfg)) {
        Write-Error "Missing config $cfg. Run: $python scripts\gen_configs.py"
    }
    $log = Join-Path $logDir "$arm`_seed$Seed.log"
    Write-Host "Launching $arm (seed $Seed) -> $log" -ForegroundColor Cyan

    $p = Start-Process -FilePath $python `
        -ArgumentList "-u", "-m", "pokerl.train", "--config", $cfg, "--seed", $Seed `
        -WorkingDirectory $root -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err" -NoNewWindow -PassThru
    # Touching .Handle caches the process handle. Without it .ExitCode reads back as $null
    # after WaitForExit, and the summary below reports successful runs as "FAILED ()".
    $null = $p.Handle
    $procs += [pscustomobject]@{ Arm = $arm; Process = $p; Log = $log }

    if ($arm -ne $Arms[-1]) { Start-Sleep -Seconds $StaggerSeconds }
}

Write-Host "`n$($procs.Count) runs in flight. Follow one with:" -ForegroundColor Green
foreach ($r in $procs) { Write-Host "  Get-Content -Wait '$($r.Log)'" }
Write-Host "`nWaiting for all runs to finish..." -ForegroundColor Yellow

foreach ($r in $procs) {
    $r.Process.WaitForExit()
    $code = $r.Process.ExitCode
    $state = if ($code -eq 0) { "OK" } elseif ($null -eq $code) { "UNKNOWN (no exit code)" } else { "FAILED ($code)" }
    Write-Host "  $($r.Arm): $state"
}

Write-Host "`nEvaluate with:" -ForegroundColor Green
foreach ($r in $procs) {
    Write-Host "  $python -m pokerl.evaluate --run results\$($r.Arm)_seed$Seed"
}
