# Starts the local Pokemon Showdown server for training.
#
# --no-security disables rate limiting and authentication. Without it, parallel training
# environments get throttled into connection timeouts. This server must only ever be
# reachable on localhost.
#
# Usage:  .\scripts\start_showdown.ps1
# Then:   http://localhost:8000

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "pokemon-showdown"

if (-not (Test-Path $server)) {
    Write-Error "Showdown not found at $server. Run .\scripts\setup_showdown.ps1 first."
}

Set-Location $server
Write-Host "Starting Pokemon Showdown on http://localhost:8000 (--no-security)" -ForegroundColor Cyan
node pokemon-showdown start --no-security
