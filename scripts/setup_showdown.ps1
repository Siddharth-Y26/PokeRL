# One-time setup of the local Pokemon Showdown server.
#
# Usage:  .\scripts\setup_showdown.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "pokemon-showdown"

if (-not (Test-Path $server)) {
    Write-Host "Cloning Pokemon Showdown..." -ForegroundColor Cyan
    git clone --depth 1 https://github.com/smogon/pokemon-showdown.git $server
} else {
    Write-Host "Showdown already cloned at $server" -ForegroundColor Yellow
}

Set-Location $server
Write-Host "Installing npm dependencies..." -ForegroundColor Cyan
npm install --no-audit --no-fund

$cfg = Join-Path $server "config\config.js"
if (-not (Test-Path $cfg)) {
    Copy-Item (Join-Path $server "config\config-example.js") $cfg
    Write-Host "Created config/config.js" -ForegroundColor Green
}

Write-Host "`nDone. Start the server with: .\scripts\start_showdown.ps1" -ForegroundColor Green
