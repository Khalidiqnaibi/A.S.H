<#
.SYNOPSIS
  Rehearse the whole flash-drive workflow on ONE machine, with the network
  blocked, before you walk anything over.

.DESCRIPTION
  Three stages:

    1. Stage the kit into a folder that stands in for the flash drive.
    2. Run install_offline.ps1 out of that folder into a scratch install dir,
       with every outbound-HTTP environment variable pointed at a dead port so
       that ANY accidental network call fails instantly instead of silently
       succeeding because this machine happens to be online.
    3. Run the offline self-test under the same blocked environment.

  Loopback is exempted via NO_PROXY, because Ollama is supposed to be reachable
  on 127.0.0.1 -- that is the whole point.

  This does not need a real flash drive and does not touch your real install.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File offline\simulate_airgap.ps1

.EXAMPLE
  .\offline\simulate_airgap.ps1 -Kit D:\ASH_OFFLINE_KIT -FakeDrive D:\SIM_USB -InstallDir D:\ASH_SIM
#>
[CmdletBinding()]
param(
    [string] $Kit        = "D:\ASH_OFFLINE_KIT",
    [string] $FakeDrive  = "D:\SIM_USB",
    [string] $InstallDir = "D:\ASH_SIM",
    [switch] $SkipCopy,           # reuse an already-staged FakeDrive
    [switch] $KeepInstall         # do not wipe InstallDir first
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Banner($msg) {
    Write-Host ""
    Write-Host ("#" * 70) -ForegroundColor Magenta
    Write-Host "# $msg" -ForegroundColor Magenta
    Write-Host ("#" * 70) -ForegroundColor Magenta
}

# -----------------------------------------------------------------------------
Banner "STAGE 1 -- copy the kit onto the stand-in flash drive"
# -----------------------------------------------------------------------------
if (-not (Test-Path $Kit)) { throw "no kit at $Kit -- run build_usb.ps1 first" }

$staged = Join-Path $FakeDrive "ASH_OFFLINE_KIT"
if ($SkipCopy -and (Test-Path $staged)) {
    Write-Host "    reusing $staged" -ForegroundColor Yellow
} else {
    New-Item -ItemType Directory -Force -Path $staged | Out-Null
    Write-Host "    $Kit  ->  $staged"
    $t0 = Get-Date
    # NO /MT here, deliberately. Multithreaded robocopy interleaves writes, and
    # on flash media that collapses sequential throughput: measured 0.4 MB/s
    # with /MT:8 versus 3.9 MB/s single-threaded on the same USB stick. /MT is
    # a win on SSDs and a disaster on thumb drives.
    robocopy $Kit $staged /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
    $mb = [math]::Round(((Get-ChildItem $staged -Recurse -File | Measure-Object Length -Sum).Sum)/1MB, 1)
    Write-Host "    copied $mb MB in $([math]::Round(((Get-Date) - $t0).TotalSeconds,1))s" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
Banner "STAGE 2 -- install from the drive, with the network blocked"
# -----------------------------------------------------------------------------
# Port 9 is the discard port: nothing listens, so every proxied request fails
# immediately rather than hanging. If any step in the installer still needs the
# internet, it fails here -- which is exactly what we want to find out now.
# Only the uppercase spellings are listed: Windows environment variables are
# case-insensitive, so setting HTTP_PROXY also answers a lookup for http_proxy.
$blocked = @{
    HTTP_PROXY               = "http://127.0.0.1:9"
    HTTPS_PROXY              = "http://127.0.0.1:9"
    ALL_PROXY                = "http://127.0.0.1:9"
    NO_PROXY                 = "127.0.0.1,localhost"
    HF_HUB_OFFLINE           = "1"
    TRANSFORMERS_OFFLINE     = "1"
    PIP_NO_INDEX             = "1"
    PIP_DISABLE_PIP_VERSION_CHECK = "1"
}
$saved = @{}
foreach ($k in $blocked.Keys) {
    $saved[$k] = [Environment]::GetEnvironmentVariable($k, "Process")
    [Environment]::SetEnvironmentVariable($k, $blocked[$k], "Process")
}
Write-Host "    outbound HTTP -> 127.0.0.1:9 (dead), loopback exempt via NO_PROXY"

try {
    if (-not $KeepInstall -and (Test-Path $InstallDir)) {
        Write-Host "    wiping $InstallDir"
        Remove-Item $InstallDir -Recurse -Force
    }

    # Prove the block is real before trusting the run that follows.
    #
    # Probe with PYTHON, not Invoke-WebRequest. PowerShell 5.1's web cmdlets go
    # through .NET/WinINET and ignore HTTP_PROXY/HTTPS_PROXY entirely, so they
    # would sail straight past this block and report "still online" no matter
    # what we set. pip, requests and urllib -- which are the only things in the
    # install that could reach out -- do honour these variables. Probe the layer
    # that is actually being constrained.
    # The probe swallows its own exception and reports on stdout. Letting Python
    # raise would work too, except that Windows PowerShell 5.1 turns a native
    # command's stderr into a terminating NativeCommandError under
    # ErrorActionPreference='Stop' -- so a *successful* block would abort the run.
    Write-Host "    sanity check: is outbound HTTP blocked for Python/pip?"
    $probeFile = Join-Path $env:TEMP "ash_netprobe.py"
@'
import sys
try:
    import requests
    requests.get("https://pypi.org/simple/", timeout=8)
    sys.stdout.write("REACHED")
except Exception:
    sys.stdout.write("BLOCKED")
'@ | Set-Content -Path $probeFile -Encoding ascii
    $probe = (& python $probeFile)
    Remove-Item $probeFile -Force -ErrorAction SilentlyContinue
    if ("$probe".Trim() -ne "BLOCKED") {
        throw "python still reached pypi.org -- the block did not take effect, so this run would prove nothing"
    }
    Write-Host "    + blocked (good)" -ForegroundColor Green

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $staged "scripts\install_offline.ps1") `
        -KitRoot $staged -InstallDir $InstallDir -Force
    $installCode = $LASTEXITCODE
}
finally {
    foreach ($k in $blocked.Keys) {
        [Environment]::SetEnvironmentVariable($k, $saved[$k], "Process")
    }
    Write-Host ""
    Write-Host "    network environment restored" -ForegroundColor DarkGray
}

# -----------------------------------------------------------------------------
Banner "RESULT"
# -----------------------------------------------------------------------------
if ($installCode -eq 0) {
    Write-Host " Install + self-test PASSED with the network blocked." -ForegroundColor Green
    Write-Host " The kit at $Kit is good to copy to a real flash drive."
} else {
    Write-Host " Install or self-test FAILED (exit $installCode). Scroll up for the FAIL lines." -ForegroundColor Red
}
Write-Host ""
Write-Host " simulated drive : $staged"
Write-Host " simulated target: $InstallDir"
Write-Host ""
Write-Host " Clean up with:" -ForegroundColor DarkGray
Write-Host "   Remove-Item '$FakeDrive','$InstallDir' -Recurse -Force" -ForegroundColor DarkGray
exit $installCode
