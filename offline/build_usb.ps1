<#
.SYNOPSIS
  Build the A.S.H offline install kit on an ONLINE Windows machine.

.DESCRIPTION
  Produces a single self-contained folder you copy to a flash drive. The
  offline machine then runs scripts\install_offline.ps1 from that folder and
  needs no network at any point.

  Everything this script collects is pinned to:
      Windows x64  +  CPython 3.11  +  Ollama
  Wheels are resolved for the interpreter running this script, so build with
  the same Python minor version the offline machine will use.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File offline\build_usb.ps1 -Out D:\ASH_OFFLINE_KIT

.EXAMPLE
  # Ship two models and skip the big installers (offline box already has them)
  .\offline\build_usb.ps1 -Models mistral:latest,qwen2.5:3b -SkipInstallers
#>
[CmdletBinding()]
param(
    [string]   $Out         = "D:\ASH_OFFLINE_KIT",
    [string[]] $Models      = @("mistral:latest"),
    [string]   $RepoRoot    = "",
    [switch]   $SkipInstallers,
    [switch]   $SkipWheels,
    [switch]   $SkipModels,
    [switch]   $SkipHfCache,
    [switch]   $SkipKokoro,
    [switch]   $SkipDaemonExtras
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"   # much faster Invoke-WebRequest

# $ScriptDir is not reliably populated inside param() defaults when the
# script is launched with a relative -File path, so resolve it here instead.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path }

# --- pinned download sources -------------------------------------------------
$PYTHON_URL  = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$OLLAMA_URL  = "https://ollama.com/download/OllamaSetup.exe"
$KOKORO_MODEL_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
$KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
$SPACY_MODELS = @(
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"
)
# HuggingFace repos the runtime loads by name. Air-gapped, these must already
# be in the HF cache or every embed/sentiment call falls back or fails.
$HF_REPOS = @(
    "models--sentence-transformers--all-MiniLM-L6-v2",
    "models--distilbert-base-uncased-finetuned-sst-2-english"
)

function Step($n, $msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
    Write-Host "[$n] $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
}
function Info($m)  { Write-Host "    $m" }
function Warn($m)  { Write-Host "    ! $m" -ForegroundColor Yellow }
function Ok($m)    { Write-Host "    + $m" -ForegroundColor Green }
function Fail($m)  { Write-Host "    X $m" -ForegroundColor Red; throw $m }

function Get-DirSizeMB($path) {
    if (-not (Test-Path $path)) { return 0 }
    $b = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
          Measure-Object -Property Length -Sum).Sum
    if (-not $b) { return 0 }
    return [math]::Round($b / 1MB, 1)
}

function Fetch($url, $dest) {
    $leaf = Split-Path $dest -Leaf
    if (Test-Path $dest) {
        Ok "already have $leaf ($([math]::Round((Get-Item $dest).Length/1MB,1)) MB)"
        return
    }
    # Download to .part and rename only on success. A build killed mid-download
    # would otherwise leave a truncated file that the next run happily accepts
    # as "already have" -- and the offline machine finds out the hard way.
    $part = "$dest.part"
    if (Test-Path $part) { Remove-Item $part -Force }
    Info "downloading $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing
    } catch {
        if (Test-Path $part) { Remove-Item $part -Force }
        throw
    }
    if ((Get-Item $part).Length -lt 1024) {
        Remove-Item $part -Force
        Fail "$leaf came back suspiciously small -- check the URL"
    }
    Move-Item $part $dest -Force
    Ok "$leaf ($([math]::Round((Get-Item $dest).Length/1MB,1)) MB)"
}

# =============================================================================
Step 1 "Preflight"
# =============================================================================
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { Fail "python not on PATH" }
$pyVer  = (& python -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
$pyArch = (& python -c "import platform;print(platform.machine())").Trim()
Info "python $pyVer ($pyArch) at $($py.Source)"
if ($pyVer -ne "3.11") {
    Warn "building wheels for Python $pyVer -- the OFFLINE machine must run Python $pyVer too."
}

$ollamaExe = (Get-Command ollama -ErrorAction SilentlyContinue)
if ($ollamaExe) { Info "ollama at $($ollamaExe.Source)" } else { Warn "ollama not on PATH -- model export needs -SkipModels" }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
foreach ($d in @("installers","wheels","spacy","kokoro","hf_cache","ollama_models","repo","scripts")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Out $d) | Out-Null
}
Ok "staging at $Out"

# =============================================================================
Step 2 "Installers (Python + Ollama)"
# =============================================================================
if ($SkipInstallers) { Warn "skipped" } else {
    Fetch $PYTHON_URL (Join-Path $Out "installers\python-3.11.9-amd64.exe")
    Fetch $OLLAMA_URL (Join-Path $Out "installers\OllamaSetup.exe")
}

# =============================================================================
Step 3 "Python wheels"
# =============================================================================
if ($SkipWheels) { Warn "skipped" } else {
    $wheelDir = Join-Path $Out "wheels"
    $reqSrc   = Join-Path $ScriptDir "requirements-offline.txt"
    if (-not (Test-Path $reqSrc)) { Fail "missing $reqSrc" }

    # torch first, from the CPU-only index. Plain PyPI hands Windows the CUDA
    # wheel (~2.4 GB) that ASH never uses -- it runs the encoders on CPU.
    Info "torch (CPU-only index)"
    & python -m pip download torch --index-url https://download.pytorch.org/whl/cpu -d $wheelDir
    if ($LASTEXITCODE -ne 0) { Fail "pip download torch failed" }

    # Constraints that every later resolve must respect.
    #
    # torch: pin the exact CPU build we just fetched, or the next resolve goes
    # back to PyPI and quietly adds the ~2.4 GB CUDA wheel alongside it.
    #
    # numpy: transformers 4.38.2 and sentence-transformers 2.6.1 are known-good
    # against 1.26.4. Without this pin, a later requirement (opencv-python is
    # the one that does it) drags in numpy 2.x and *uninstalls* 1.26.4 out from
    # under the stack that was already installed correctly.
    $constraintLines = @("numpy==1.26.4")
    $torchWhl = Get-ChildItem $wheelDir -Filter "torch-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    $constraints = Join-Path $Out "build_constraints.txt"
    if ($torchWhl) {
        $torchVer = ($torchWhl.Name -split "-")[1]
        $constraintLines += "torch==$torchVer"
        Info "pinning torch==$torchVer, numpy==1.26.4"
    } else {
        Warn "no torch wheel found to pin"
    }
    Set-Content -Path $constraints -Value $constraintLines -Encoding ascii

    Info "remaining requirements"
    & python -m pip download -r $reqSrc -d $wheelDir --find-links $wheelDir -c $constraints
    if ($LASTEXITCODE -ne 0) { Fail "pip download -r failed" }

    # kokoro-onnx declares numpy>=2.0.2, which would drag the whole stack off
    # the numpy 1.26.4 that transformers 4.38.2 is known-good against. Fetch it
    # without its metadata; its real deps are in requirements-offline.txt.
    $nodepsReq = Join-Path $ScriptDir "requirements-offline-nodeps.txt"
    if (Test-Path $nodepsReq) {
        Info "no-deps packages (kokoro-onnx)"
        & python -m pip download -r $nodepsReq -d $wheelDir --no-deps
        if ($LASTEXITCODE -ne 0) { Warn "no-deps download failed -- local TTS will be unavailable" }
    }

    if (-not $SkipDaemonExtras) {
        $daemonReq = Join-Path $ScriptDir "requirements-offline-daemon.txt"
        if (Test-Path $daemonReq) {
            Info "daemon sensor extras"
            & python -m pip download -r $daemonReq -d $wheelDir --find-links $wheelDir -c $constraints
            if ($LASTEXITCODE -ne 0) { Warn "some daemon extras could not be downloaded -- ashd.py will perceive less" }
        }
    }

    # pip itself, so the offline box can bootstrap a fresh venv.
    & python -m pip download pip setuptools wheel -d $wheelDir | Out-Null

    # Any sdist left in the wheelhouse is a landmine: `pip install --no-index`
    # on the offline box would have to COMPILE it, and an air-gapped machine
    # rarely has MSVC. Build the wheel here, where a toolchain may exist; drop
    # the package if we cannot, rather than shipping something that explodes
    # halfway through the install. (webrtcvad is the usual culprit.)
    $sdists = Get-ChildItem $wheelDir -File |
              Where-Object { $_.Name -match '\.(tar\.gz|zip)$' }
    $script:DroppedSdists = @()
    foreach ($s in $sdists) {
        Info "source-only package: $($s.Name) -- building a wheel"
        & python -m pip wheel $s.FullName -w $wheelDir --no-deps --find-links $wheelDir
        if ($LASTEXITCODE -eq 0) {
            Remove-Item $s.FullName -Force
            Ok "wheel built for $($s.Name)"
        } else {
            Remove-Item $s.FullName -Force
            $script:DroppedSdists += $s.Name
            Warn "no compiler here -- DROPPED $($s.Name) from the kit"
        }
    }
    if ($script:DroppedSdists.Count) {
        Warn "dropped source-only packages: $($script:DroppedSdists -join ', ')"
        Warn "if any of these are REQUIRED, install MSVC Build Tools here and re-run"
    }

    Ok "$((Get-ChildItem $wheelDir -File).Count) wheels, $(Get-DirSizeMB $wheelDir) MB"
}

# =============================================================================
Step 4 "spaCy language models"
# =============================================================================
foreach ($u in $SPACY_MODELS) {
    Fetch $u (Join-Path $Out ("spacy\" + (Split-Path $u -Leaf)))
}

# =============================================================================
Step 5 "Kokoro TTS model files"
# =============================================================================
if ($SkipKokoro) { Warn "skipped" } else {
    Fetch $KOKORO_MODEL_URL  (Join-Path $Out "kokoro\kokoro-v1.0.onnx")
    Fetch $KOKORO_VOICES_URL (Join-Path $Out "kokoro\voices-v1.0.bin")
}

# =============================================================================
Step 6 "HuggingFace model cache"
# =============================================================================
if ($SkipHfCache) { Warn "skipped" } else {
    $hfHub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
    if (-not (Test-Path $hfHub)) {
        Warn "no HF cache at $hfHub -- run the app once online to populate it, then re-run"
    } else {
        foreach ($repo in $HF_REPOS) {
            $src = Join-Path $hfHub $repo
            if (-not (Test-Path $src)) {
                Warn "MISSING from cache: $repo"
                continue
            }
            $dst = Join-Path $Out "hf_cache\$repo"
            robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP | Out-Null
            Ok "$repo ($(Get-DirSizeMB $dst) MB)"
        }
        $v = Join-Path $hfHub "version.txt"
        if (Test-Path $v) { Copy-Item $v (Join-Path $Out "hf_cache\version.txt") -Force }
    }
}

# =============================================================================
Step 7 "Ollama models"
# =============================================================================
# GOTCHA, found by simulating this: the Ollama app can be pointed at a model
# store that is NOT ~/.ollama/models, and it keeps that choice in its own
# settings -- not in an env var this shell can see. So resolve the LIVE store
# from the running runner process before falling back to the default path.
function Get-OllamaLogStores {
    # The Ollama server prints its whole resolved environment at startup, so
    # its log is the one place that reliably reveals a custom model directory.
    # Paths appear escaped there ("D:\\models"), hence the unescape.
    $out = New-Object System.Collections.Generic.List[string]
    $dir = Join-Path $env:LOCALAPPDATA "Ollama"
    if (-not (Test-Path $dir)) { return $out }
    $logs = Get-ChildItem $dir -Filter "server*.log" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending
    foreach ($f in $logs) {
        $hit = Select-String -Path $f.FullName -Pattern 'OLLAMA_MODELS:(\S+)' -ErrorAction SilentlyContinue |
               Select-Object -Last 1
        if ($hit) {
            $p = $hit.Matches[0].Groups[1].Value.Replace('\\', '\').Trim('"')
            if ($p -and (Test-Path $p) -and -not $out.Contains($p)) { $out.Add($p) }
        }
    }
    return $out
}

function Resolve-OllamaStores {
    $stores = New-Object System.Collections.Generic.List[string]
    if ($env:OLLAMA_MODELS -and (Test-Path $env:OLLAMA_MODELS)) {
        $stores.Add((Resolve-Path $env:OLLAMA_MODELS).Path)
    }
    foreach ($p in (Get-OllamaLogStores)) {
        if (-not $stores.Contains($p)) { $stores.Add($p) }
    }
    Get-CimInstance Win32_Process -Filter "Name like 'ollama%'" -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.CommandLine -match '(?<p>[A-Za-z]:\\[^"]*?)\\blobs\\sha256-') {
            $p = $Matches['p']
            if ((Test-Path $p) -and -not $stores.Contains($p)) { $stores.Add($p) }
        }
    }
    $default = Join-Path $env:USERPROFILE ".ollama\models"
    if ((Test-Path $default) -and -not $stores.Contains($default)) { $stores.Add($default) }
    return $stores
}

if ($SkipModels) { Warn "skipped" } else {
    $stores = Resolve-OllamaStores
    if ($stores.Count -eq 0) { Fail "no Ollama model store found" }
    Info "model stores found: $($stores -join ' | ')"

    $outModels = Join-Path $Out "ollama_models"
    New-Item -ItemType Directory -Force -Path (Join-Path $outModels "blobs") | Out-Null

    foreach ($m in $Models) {
        $parts = $m.Split(":", 2)
        $name  = $parts[0]
        $tag   = if ($parts.Count -gt 1 -and $parts[1]) { $parts[1] } else { "latest" }
        if ($name -notmatch "/") { $relManifest = "manifests\registry.ollama.ai\library\$name\$tag" }
        else                     { $relManifest = "manifests\registry.ollama.ai\$name\$tag" }

        $manifest  = $null
        $storeRoot = $null
        foreach ($s in $stores) {
            $cand = Join-Path $s $relManifest
            if (Test-Path $cand) { $manifest = $cand; $storeRoot = $s; break }
        }

        if (-not $manifest) {
            if (-not $ollamaExe) { Fail "$m not present locally and ollama is not on PATH" }
            Info "$m not in any store -- pulling (this is the big download)"
            & ollama pull $m
            if ($LASTEXITCODE -ne 0) { Fail "ollama pull $m failed" }
            $stores = Resolve-OllamaStores
            foreach ($s in $stores) {
                $cand = Join-Path $s $relManifest
                if (Test-Path $cand) { $manifest = $cand; $storeRoot = $s; break }
            }
            if (-not $manifest) { Fail "pulled $m but still cannot find its manifest" }
        }

        Info "$m  <-  $storeRoot"

        $dstManifest = Join-Path $outModels $relManifest
        New-Item -ItemType Directory -Force -Path (Split-Path $dstManifest -Parent) | Out-Null
        Copy-Item $manifest $dstManifest -Force

        # copy every blob the manifest references (config + all layers)
        $j = Get-Content $manifest -Raw | ConvertFrom-Json
        $digests = @()
        if ($j.config.digest) { $digests += $j.config.digest }
        foreach ($l in $j.layers) { $digests += $l.digest }

        foreach ($d in ($digests | Select-Object -Unique)) {
            $blobName = $d -replace ":", "-"
            $srcBlob  = Join-Path $storeRoot "blobs\$blobName"
            $dstBlob  = Join-Path $outModels "blobs\$blobName"
            if (-not (Test-Path $srcBlob)) { Fail "manifest references missing blob $blobName" }
            if (Test-Path $dstBlob) { continue }
            Copy-Item $srcBlob $dstBlob -Force
            Info "  blob $($blobName.Substring(0,19))... $([math]::Round((Get-Item $dstBlob).Length/1MB,1)) MB"
        }
        Ok "$m exported"
    }
    Ok "ollama_models total $(Get-DirSizeMB $outModels) MB"
}

# =============================================================================
Step 8 "Repository snapshot"
# =============================================================================
$repoOut = Join-Path $Out "repo"
# Caches and the local model dir ship separately, so they are excluded here.
# .env is excluded on purpose -- it holds the online API keys.
robocopy $RepoRoot $repoOut /E /NFL /NDL /NJH /NJS /NP /XD ".git" ".claude" "__pycache__" ".venv" "models" "chroma_data" "node_modules" /XF "*.pyc" ".env" | Out-Null
if ($LASTEXITCODE -ge 8) { Fail "robocopy repo failed ($LASTEXITCODE)" }
New-Item -ItemType Directory -Force -Path (Join-Path $repoOut "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repoOut "logs")   | Out-Null
Ok "repo snapshot $(Get-DirSizeMB $repoOut) MB"

# =============================================================================
Step 9 "Scripts + manifest"
# =============================================================================
Copy-Item (Join-Path $ScriptDir "install_offline.ps1")       (Join-Path $Out "scripts\") -Force
Copy-Item (Join-Path $ScriptDir "verify_offline.py")         (Join-Path $Out "scripts\") -Force
Copy-Item (Join-Path $ScriptDir "run_ash.ps1")               (Join-Path $Out "scripts\") -Force
Copy-Item (Join-Path $ScriptDir "ash.env.template")          (Join-Path $Out "scripts\") -Force
Copy-Item (Join-Path $ScriptDir "requirements-offline.txt")  (Join-Path $Out "scripts\") -Force
# The installer applies these same pins, so a later optional package cannot
# swap numpy or torch out from under the stack.
$constraintsOut = Join-Path $Out "build_constraints.txt"
if (Test-Path $constraintsOut) { Copy-Item $constraintsOut (Join-Path $Out "scripts\") -Force }
$nodepsReq = Join-Path $ScriptDir "requirements-offline-nodeps.txt"
if (Test-Path $nodepsReq) { Copy-Item $nodepsReq (Join-Path $Out "scripts\") -Force }
$daemonReq = Join-Path $ScriptDir "requirements-offline-daemon.txt"
if ((Test-Path $daemonReq) -and -not $SkipDaemonExtras) {
    Copy-Item $daemonReq (Join-Path $Out "scripts\") -Force
}
Copy-Item (Join-Path $ScriptDir "OFFLINE_README.md")         (Join-Path $Out "00_READ_ME_FIRST.md") -Force

$sections = [ordered]@{}
foreach ($d in @("installers","wheels","spacy","kokoro","hf_cache","ollama_models","repo","scripts")) {
    $sections[$d] = [ordered]@{
        size_mb = Get-DirSizeMB (Join-Path $Out $d)
        files   = (Get-ChildItem (Join-Path $Out $d) -Recurse -File -ErrorAction SilentlyContinue).Count
    }
}
$manifest = [ordered]@{
    built_at      = (Get-Date).ToString("s")
    built_by      = "$env:COMPUTERNAME\$env:USERNAME"
    build_python  = $pyVer
    build_arch    = $pyArch
    target        = "Windows x64, Python $pyVer, Ollama"
    ollama_models = $Models
    dropped_source_only = @($script:DroppedSdists)
    sections      = $sections
    total_mb      = (Get-DirSizeMB $Out)
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $Out "build_manifest.json") -Encoding utf8

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host " KIT READY: $Out" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
foreach ($k in $sections.Keys) {
    "{0,-16} {1,10} MB  {2,6} files" -f $k, $sections[$k].size_mb, $sections[$k].files | Write-Host
}
"{0,-16} {1,10} MB" -f "TOTAL", $manifest.total_mb | Write-Host -ForegroundColor Green
Write-Host ""
Write-Host "Copy the whole folder to the flash drive, then on the offline machine run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File <DRIVE>\ASH_OFFLINE_KIT\scripts\install_offline.ps1" -ForegroundColor Yellow
