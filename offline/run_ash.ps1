<#
.SYNOPSIS
  Start A.S.H on the offline machine.

.EXAMPLE
  .\run_ash.ps1            # Flask + Socket.IO web UI on http://127.0.0.1:5000
.EXAMPLE
  .\run_ash.ps1 -Cli       # terminal REPL
.EXAMPLE
  .\run_ash.ps1 -Daemon    # always-on background daemon (ashd.py)
.EXAMPLE
  .\run_ash.ps1 -Verify    # re-run the offline self-test
#>
[CmdletBinding()]
param(
    [switch] $Cli,
    [switch] $Daemon,
    [switch] $Verify,
    [string] $AshDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $AshDir) { $AshDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $AshDir

$venvPy = Join-Path $AshDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "No venv at $venvPy -- falling back to system python" -ForegroundColor Yellow
    $venvPy = "python"
}

# Make sure Ollama is up before ASH tries to talk to it; ASH imports fine
# without it but every turn would come back as a connection error.
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    $up = $false
    try { Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null; $up = $true } catch { }
    if (-not $up) {
        Write-Host "Starting ollama serve..." -ForegroundColor Cyan
        Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            try { Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null; $up = $true; break } catch { }
        }
    }
    if ($up) { Write-Host "Ollama is up." -ForegroundColor Green }
    else     { Write-Host "Ollama did not come up -- ASH will start but cannot generate." -ForegroundColor Yellow }
} else {
    Write-Host "ollama not on PATH -- ASH will start but cannot generate." -ForegroundColor Yellow
}

if     ($Verify) { & $venvPy verify_offline.py }
elseif ($Cli)    { & $venvPy -m src.py.ash }
elseif ($Daemon) { & $venvPy ashd.py }
else {
    Write-Host "Web UI -> http://127.0.0.1:5000" -ForegroundColor Green
    & $venvPy app.py
}
