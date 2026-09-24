# Unattended completion of the study.
#
# Waits for the running matrix to finish, then does the work that must happen afterwards:
#   1. re-runs the two seed-0 cells that predate bug fixes
#        - ppo_masked_v1_seed0  (crashed on the Move.priority KeyError)
#        - tabular_q_v0_seed0   (ran with the index-biased argmax tie-break)
#   2. re-runs the full cross-evaluation over the corrected set
#   3. regenerates summary.csv and the figures
#   4. writes report/RESULTS.md
#
# The re-runs use --torch-threads 1 so the tuned setting gets measured against the
# 15.2 min/run baseline on an otherwise idle machine.
#
# Everything is local: no internet required at any point.
#
# Usage:  .\scripts\finish_up.ps1

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

$log = Join-Path $root "results\finish_up.log"
function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

Say "finish_up started; waiting for the matrix to drain"

# --- 1. Wait for the matrix ------------------------------------------------------------
# Require sustained silence: between runs a new python starts within seconds, so a single
# empty poll is not evidence the matrix is done. Six consecutive quiet polls (~3 min) is.
$quiet = 0
while ($quiet -lt 6) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
              Where-Object { $_.CommandLine -and $_.CommandLine -match 'pokerl\.(train|evaluate)' })
    if ($busy.Count -gt 0) { $quiet = 0 } else { $quiet++ }
    Start-Sleep -Seconds 30
}
Say "matrix idle; proceeding"

# --- 2. Showdown must still be up ------------------------------------------------------
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8000" -TimeoutSec 5 -UseBasicParsing
    Say "Showdown server responding"
} catch {
    Say "ERROR: Showdown server is down; cannot re-run. Skipping to analysis."
    & $python (Join-Path $root "scripts\analyse.py")   2>&1 | Tee-Object -Append $log
    & $python (Join-Path $root "scripts\make_report.py") 2>&1 | Tee-Object -Append $log
    exit 1
}

# --- 3. Corrective re-runs -------------------------------------------------------------
# Derived from what is actually missing on disk rather than a hardcoded list, so any run
# that crashed for any reason is picked up automatically. Also re-runs tabular_q seed 0,
# which completed but predates the argmax tie-break fix and so is not comparable with its
# own later seeds.
$seeds = @(0, 1, 2)
$configs = Get-ChildItem (Join-Path $root "configs\exp\*.yaml") |
           Where-Object { $_.BaseName -notin @("smoke", "sanity_ppo_masked") }

$todo = @()
foreach ($cfg in $configs) {
    foreach ($seed in $seeds) {
        $runDir = Join-Path $root "results\$($cfg.BaseName)_seed$seed"
        if (-not (Test-Path (Join-Path $runDir "eval.csv"))) {
            $todo += [PSCustomObject]@{ Cfg = $cfg; Seed = $seed; Why = "missing" }
        }
    }
}
# Stale: ran before the tie-break fix.
$stale = Join-Path $root "results\tabular_q_v0_seed0"
if (Test-Path (Join-Path $stale "eval.csv")) {
    $todo += [PSCustomObject]@{
        Cfg  = ($configs | Where-Object BaseName -eq "tabular_q_v0")
        Seed = 0; Why = "predates tie-break fix"
    }
}

Say "corrective re-runs queued: $($todo.Count)"
foreach ($t in $todo) { Say "   $($t.Cfg.BaseName) seed $($t.Seed)  [$($t.Why)]" }

foreach ($t in $todo) {
    $name = $t.Cfg.BaseName
    $runDir = Join-Path $root "results\${name}_seed$($t.Seed)"
    Say "RE-RUN $name seed $($t.Seed) (torch-threads 1)"
    $t0 = Get-Date
    & $python -m pokerl.train --config $t.Cfg.FullName --seed $t.Seed --torch-threads 1 2>&1 |
        Tee-Object -Append $log
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($LASTEXITCODE -ne 0) { Say "WARNING: training failed for $name seed $($t.Seed)"; continue }
    Say "$name seed $($t.Seed) trained in $mins min (baseline ~13 min at torch default)"
    & $python -m pokerl.evaluate --run $runDir 2>&1 | Tee-Object -Append $log
    if ($LASTEXITCODE -ne 0) { Say "WARNING: evaluation failed for $name seed $($t.Seed)" }
}

# --- 4. Cross-evaluation over the corrected set ----------------------------------------
Say "cross-evaluation"
$seed0 = Get-ChildItem (Join-Path $root "results") -Directory |
         Where-Object { $_.Name -like "*_seed0" -and $_.Name -notlike "sanity*" -and
                        (Test-Path (Join-Path $_.FullName "config.yaml")) } |
         Select-Object -ExpandProperty FullName
& $python -m pokerl.evaluate --cross $seed0 --n 100 2>&1 | Tee-Object -Append $log
if ($LASTEXITCODE -ne 0) { Say "WARNING: cross-evaluation failed" }

# --- 5. Analysis and report ------------------------------------------------------------
Say "analysis"
& $python (Join-Path $root "scripts\analyse.py") 2>&1 | Tee-Object -Append $log
Say "report"
& $python (Join-Path $root "scripts\make_report.py") 2>&1 | Tee-Object -Append $log

Say "DONE. See report\RESULTS.md, report\figures\, results\summary.csv"
