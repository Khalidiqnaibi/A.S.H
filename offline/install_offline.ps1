<#
.SYNOPSIS
  Install A.S.H on an AIR-GAPPED Windows machine from the flash-drive kit.

.DESCRIPTION
  Run this from the kit folder on the flash drive. It never touches the
  network: pip is called with --no-index, HuggingFace is forced offline, and
  the Ollama model is imported by copying blobs rather than pulling.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File E:\ASH_OFFLINE_KIT\scripts\install_offline.ps1

.EXAMPLE
  # Install somewhere other than C:\ASH
  .\install_offline.ps1 -InstallDir D:\ASH
#>
[CmdletBinding()]
param(
    [string] $KitRoot    = "",
    [string] $InstallDir = "C:\ASH",
    [string] $Model      = "",      # default: first model in build_manifest.json
    [switch] $SkipVenv,
    [switch] $Force                  # overwrite an existing InstallDir
)

$ErrorActionPreference = "Stop"

# $ScriptDir is not reliably populated inside param() defaults when the
# script is launched with a relative -File path, so resolve it here instead.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $KitRoot) { $KitRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path }

function Step($n, $msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
    Write-Host "[$n] $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
}
function Info($m) { Write-Host "    $m" }
function Warn($m) { Write-Host "    ! $m" -ForegroundColor Yellow }
function Ok($m)   { Write-Host "    + $m" -ForegroundColor Green }
function Fail($m) { Write-Host "    X $m" -ForegroundColor Red; throw $m }

# =============================================================================
Step 1 "Check the kit"
# =============================================================================
Info "kit root: $KitRoot"
foreach ($d in @("wheels","repo","scripts")) {
    if (-not (Test-Path (Join-Path $KitRoot $d))) { Fail "kit is incomplete -- missing $d" }
}
$manifestPath = Join-Path $KitRoot "build_manifest.json"
if (Test-Path $manifestPath) {
    $bm = Get-Content $manifestPath -Raw | ConvertFrom-Json
    Info "built $($bm.built_at) on $($bm.built_by) for Python $($bm.build_python)"
    if (-not $Model -and $bm.ollama_models) { $Model = @($bm.ollama_models)[0] }
} else {
    Warn "no build_manifest.json -- continuing anyway"
}
if (-not $Model) { $Model = "mistral:latest" }
Info "target model: $Model"

# =============================================================================
Step 2 "Python"
# =============================================================================
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
    $pyInstaller = Get-ChildItem (Join-Path $KitRoot "installers") -Filter "python-*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $pyInstaller) { Fail "no python on PATH and no installer in the kit" }
    Warn "python not found -- installing $($pyInstaller.Name) (takes a minute)"
    # PrependPath=1 is what makes `python` resolvable in NEW shells afterwards.
    Start-Process -FilePath $pyInstaller.FullName -Wait -ArgumentList @(
        "/quiet","InstallAllUsers=1","PrependPath=1","Include_pip=1","Include_test=0"
    )
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
    $python = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $python) { Fail "python still not on PATH -- open a NEW PowerShell and re-run" }
}
$pyVer = (& python -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
Ok "python $pyVer at $($python.Source)"
if ($bm -and $bm.build_python -and $pyVer -ne $bm.build_python) {
    Fail "wheel/interpreter mismatch: kit was built for Python $($bm.build_python), this machine has $pyVer. Install Python $($bm.build_python) from installers\ first."
}

# =============================================================================
Step 3 "Ollama"
# =============================================================================
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue)
if (-not $ollama) {
    $ollamaInstaller = Join-Path $KitRoot "installers\OllamaSetup.exe"
    if (-not (Test-Path $ollamaInstaller)) { Fail "no ollama on PATH and no OllamaSetup.exe in the kit" }
    Warn "ollama not found -- installing"
    Start-Process -FilePath $ollamaInstaller -Wait -ArgumentList "/VERYSILENT","/NORESTART"
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
    $ollama = (Get-Command ollama -ErrorAction SilentlyContinue)
    if (-not $ollama) { Fail "ollama still not on PATH -- open a NEW PowerShell and re-run" }
}
Ok "ollama at $($ollama.Source)"

# =============================================================================
Step 4 "Copy the application"
# =============================================================================
if ((Test-Path $InstallDir) -and -not $Force) {
    Warn "$InstallDir exists -- updating in place (pass -Force to wipe it first)"
} elseif (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
robocopy (Join-Path $KitRoot "repo") $InstallDir /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Fail "robocopy failed ($LASTEXITCODE)" }
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "logs")   | Out-Null
Ok "application at $InstallDir"

# =============================================================================
Step 5 "Kokoro TTS model files"
# =============================================================================
$kokoroSrc = Join-Path $KitRoot "kokoro"
if (Test-Path $kokoroSrc) {
    Get-ChildItem $kokoroSrc -File | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $InstallDir "models\$($_.Name)") -Force
        Ok "$($_.Name) ($([math]::Round($_.Length/1MB,1)) MB)"
    }
} else { Warn "no kokoro\ in the kit -- TTS will be disabled (ASH still runs)" }

# =============================================================================
Step 6 "Virtualenv + wheels (offline)"
# =============================================================================
$venv    = Join-Path $InstallDir ".venv"
$venvPy  = Join-Path $venv "Scripts\python.exe"
if ($SkipVenv) {
    Warn "skipping venv -- installing into the system interpreter"
    $venvPy = $python.Source
} else {
    if (-not (Test-Path $venvPy)) {
        Info "creating venv"
        & python -m venv $venv
        if ($LASTEXITCODE -ne 0) { Fail "venv creation failed" }
    }
    Ok "venv at $venv"
}

$wheels = Join-Path $KitRoot "wheels"

# Same pins the kit was built against. Applied to EVERY install below, because
# pip resolves each invocation independently: without this, an optional extra
# installed later (opencv-python is the offender) pulls numpy 2.x and
# uninstalls the 1.26.4 that transformers 4.38.2 was just installed against,
# leaving a broken environment that the earlier step reported as successful.
$constraints = Join-Path $KitRoot "scripts\build_constraints.txt"
$pin = @()
if (Test-Path $constraints) {
    $pin = @("-c", $constraints)
    Info "applying build constraints: $((Get-Content $constraints) -join ', ')"
} else {
    Warn "no build_constraints.txt in the kit -- optional extras may change numpy/torch"
}

# --no-index is the whole point: pip must never reach for PyPI.
Info "upgrading pip from the local wheelhouse"
& $venvPy -m pip install --no-index --find-links $wheels --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Warn "pip self-upgrade failed -- continuing with the bundled pip" }

Info "installing requirements (this is the slow step, ~2-5 min)"
& $venvPy -m pip install --no-index --find-links $wheels @pin -r (Join-Path $KitRoot "scripts\requirements-offline.txt")
if ($LASTEXITCODE -ne 0) { Fail "offline pip install failed -- see the error above" }

# kokoro-onnx's declared numpy>=2.0.2 floor is stricter than what it needs, and
# honouring it would pull the whole stack off numpy 1.26.4. Its real deps are
# already installed above, so --no-deps loses nothing.
$nodepsReq = Join-Path $KitRoot "scripts\requirements-offline-nodeps.txt"
if (Test-Path $nodepsReq) {
    Info "installing no-deps packages (kokoro-onnx)"
    & $venvPy -m pip install --no-index --find-links $wheels --no-deps -r $nodepsReq
    if ($LASTEXITCODE -ne 0) { Warn "kokoro-onnx did not install -- local TTS will be unavailable" }
}

$daemonReq = Join-Path $KitRoot "scripts\requirements-offline-daemon.txt"
if (Test-Path $daemonReq) {
    Info "installing daemon sensor extras (optional -- failures are not fatal)"
    & $venvPy -m pip install --no-index --find-links $wheels @pin -r $daemonReq
    if ($LASTEXITCODE -ne 0) { Warn "some daemon extras did not install -- ashd.py will perceive less" }
}

Info "installing spaCy language models"
Get-ChildItem (Join-Path $KitRoot "spacy") -Filter "*.whl" -ErrorAction SilentlyContinue | ForEach-Object {
    & $venvPy -m pip install --no-index --find-links $wheels @pin $_.FullName
    if ($LASTEXITCODE -ne 0) { Fail "failed to install $($_.Name)" }
    Ok $_.Name
}
Ok "python environment ready"

# =============================================================================
Step 7 "HuggingFace cache"
# =============================================================================
# Placed in the real user cache so HF_HUB_OFFLINE=1 finds it with no extra
# config. Copied, not symlinked -- the flash drive will be pulled out.
$hfSrc = Join-Path $KitRoot "hf_cache"
if (Test-Path $hfSrc) {
    $hfDst = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
    New-Item -ItemType Directory -Force -Path $hfDst | Out-Null
    robocopy $hfSrc $hfDst /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "HF cache copy failed" }
    Ok "HF cache -> $hfDst"
} else { Warn "no hf_cache\ in the kit -- embeddings will fail offline" }

# =============================================================================
Step 8 "Import Ollama model"
# =============================================================================
# Importing = dropping blobs and the manifest into Ollama's store. No pull, no
# network. Ollama picks them up on its next request; no restart needed.
$modelSrc = Join-Path $KitRoot "ollama_models"
if (-not (Test-Path $modelSrc)) {
    Warn "no ollama_models\ in the kit -- import a model yourself before ASH can talk"
} else {
    # Resolving this correctly matters more than it looks. The Ollama app keeps
    # a custom model directory in its own settings database, NOT in an
    # environment variable -- so copying blindly into ~/.ollama/models can drop
    # 4 GB of blobs somewhere the server never reads. The server logs its
    # resolved environment at startup, which is the reliable readout.
    $store = $env:OLLAMA_MODELS
    if (-not $store) {
        $logDir = Join-Path $env:LOCALAPPDATA "Ollama"
        if (Test-Path $logDir) {
            $logs = Get-ChildItem $logDir -Filter "server*.log" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending
            foreach ($f in $logs) {
                $hit = Select-String -Path $f.FullName -Pattern 'OLLAMA_MODELS:(\S+)' -ErrorAction SilentlyContinue |
                       Select-Object -Last 1
                if ($hit) {
                    # paths are escaped in the log ("D:\\models")
                    $store = $hit.Matches[0].Groups[1].Value.Replace('\\', '\').Trim('"')
                    Info "detected custom model store from ollama's log"
                    break
                }
            }
        }
    }
    if (-not $store) { $store = Join-Path $env:USERPROFILE ".ollama\models" }
    Info "ollama store: $store"
    New-Item -ItemType Directory -Force -Path (Join-Path $store "blobs") | Out-Null
    robocopy $modelSrc $store /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "model import failed ($LASTEXITCODE)" }
    Ok "model blobs + manifests imported"

    # Make sure the server is up, then confirm it can see what we dropped in.
    $tags = $null
    try { $tags = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 } catch { }
    if (-not $tags) {
        Info "starting ollama serve"
        Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            try { $tags = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3; break } catch { }
        }
    }
    if ($tags) {
        $names = @($tags.models | ForEach-Object { $_.name })
        Ok "ollama sees: $($names -join ', ')"
        if ($names -notcontains $Model) {
            Warn "'$Model' is not in that list. If OLLAMA_MODELS points elsewhere on this machine, set it and re-run step 8."
        }
    } else {
        Warn "could not reach the ollama API on 11434 -- start it manually, then re-run verify"
    }
}

# =============================================================================
Step 9 "Configuration"
# =============================================================================
$envFile = Join-Path $InstallDir ".env"
if (Test-Path $envFile) {
    Warn ".env already exists -- leaving it alone"
} else {
    $tpl = Get-Content (Join-Path $KitRoot "scripts\ash.env.template") -Raw
    # No backslash escaping here: python-dotenv takes UNQUOTED values literally,
    # so C:\ASH\data must be written exactly as that.
    $tpl = $tpl.Replace("__MODEL__", $Model).Replace("__ASH_DIR__", $InstallDir)
    Set-Content -Path $envFile -Value $tpl -Encoding utf8
    Ok "wrote $envFile"
}

# =============================================================================
Step 10 "Verify"
# =============================================================================
Copy-Item (Join-Path $KitRoot "scripts\verify_offline.py") (Join-Path $InstallDir "verify_offline.py") -Force
Copy-Item (Join-Path $KitRoot "scripts\run_ash.ps1")       (Join-Path $InstallDir "run_ash.ps1")       -Force

Push-Location $InstallDir
try {
    & $venvPy verify_offline.py
    $verifyCode = $LASTEXITCODE
} finally { Pop-Location }

Write-Host ""
if ($verifyCode -eq 0) {
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host " A.S.H INSTALLED AND VERIFIED" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green
} else {
    Write-Host ("=" * 70) -ForegroundColor Yellow
    Write-Host " INSTALLED, BUT VERIFY REPORTED PROBLEMS (see above)" -ForegroundColor Yellow
    Write-Host ("=" * 70) -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Start it with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File $InstallDir\run_ash.ps1        # web UI on http://127.0.0.1:5000" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File $InstallDir\run_ash.ps1 -Cli   # terminal REPL" -ForegroundColor Yellow
exit $verifyCode
