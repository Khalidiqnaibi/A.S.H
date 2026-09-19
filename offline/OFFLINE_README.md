# A.S.H — Offline (air-gapped) install kit

This folder is everything the offline machine needs. Once it is on the flash
drive, **no further internet access is required at any point.**

---

## 0. The short version

**On the online machine** (this repo checked out, Python 3.11 x64, Ollama installed):

```powershell
powershell -ExecutionPolicy Bypass -File offline\build_usb.ps1 -Out D:\ASH_OFFLINE_KIT
```

Copy `D:\ASH_OFFLINE_KIT` to the flash drive. Walk it over.

**On the offline machine**, plug in the drive and run:

```powershell
powershell -ExecutionPolicy Bypass -File E:\ASH_OFFLINE_KIT\scripts\install_offline.ps1
```

(`E:` = whatever letter the drive got.) Then:

```powershell
powershell -ExecutionPolicy Bypass -File C:\ASH\run_ash.ps1
```

---

## 1. What is on the drive

```
ASH_OFFLINE_KIT\
├── 00_READ_ME_FIRST.md          this file
├── build_manifest.json          what was captured, when, for which Python
├── build_constraints.txt        the exact torch build the wheels were pinned to
│
├── installers\
│   ├── python-3.11.9-amd64.exe  only needed if the offline box has no Python 3.11
│   └── OllamaSetup.exe          only needed if it has no Ollama
│
├── wheels\                      every Python dependency as a .whl, incl. torch CPU
│
├── spacy\
│   ├── en_core_web_sm-3.8.0-py3-none-any.whl    NER (required)
│   └── en_core_web_md-3.8.0-py3-none-any.whl    larger vectors (optional)
│
├── kokoro\
│   ├── kokoro-v1.0.onnx         local TTS voice model
│   └── voices-v1.0.bin          voice embeddings
│
├── hf_cache\                    pre-warmed HuggingFace cache
│   ├── models--sentence-transformers--all-MiniLM-L6-v2       embeddings (required)
│   └── models--distilbert-base-uncased-finetuned-sst-2-english  sentiment (required)
│
├── ollama_models\               the LLM itself, as raw blobs + manifest
│   ├── blobs\sha256-...
│   └── manifests\registry.ollama.ai\library\mistral\latest
│
├── repo\                        the A.S.H source tree (no .git, no .env, no caches)
│
└── scripts\
    ├── install_offline.ps1      run this on the offline machine
    ├── verify_offline.py        self-test; install runs it automatically
    ├── run_ash.ps1              launcher (web UI / CLI / daemon)
    ├── ash.env.template         becomes C:\ASH\.env
    ├── requirements-offline.txt
    ├── requirements-offline-nodeps.txt   kokoro-onnx, installed with --no-deps
    └── requirements-offline-daemon.txt   optional sensor extras for ashd.py
```

> **Why `--no-deps` for kokoro-onnx:** it declares `numpy>=2.0.2`, which makes
> pip reject the whole requirement set, because the rest of the stack is held at
> the numpy 1.26.4 that `transformers 4.38.2` and `sentence-transformers 2.6.1`
> are known-good against. That floor is stricter than what the package actually
> needs. Its real dependencies (`espeakng-loader`, `onnxruntime`,
> `phonemizer-fork`) are listed explicitly in `requirements-offline.txt`, so
> nothing is missing — only the metadata is bypassed.

Sizes are recorded in `build_manifest.json`. These are the measured figures from
an actual build (`mistral:latest`, Python 3.11 x64, daemon extras included):

| Section | Size | Files |
|---|---:|---:|
| `ollama_models\` (mistral:latest) | 4170.3 MB | 6 |
| `installers\` (Python + Ollama) | 1522.3 MB | 2 |
| `hf_cache\` | 430.3 MB | 34 |
| `wheels\` | 374.6 MB | 154 |
| `kokoro\` | 337.4 MB | 2 |
| `spacy\` | 44.1 MB | 2 |
| `repo\` | 4.5 MB | 173 |
| `scripts\` | <1 MB | 7 |
| **Total** | **6883.4 MB (6.7 GB)** | |

Drop `-SkipInstallers` if the offline box already has Python 3.11 and Ollama and
the kit falls to about **5.2 GB**.

**Use a 16 GB drive or larger**, formatted **exFAT or NTFS**. FAT32 will not
work: the mistral weights are a single **4.07 GB** blob and FAT32 caps files at
4 GB. An 8 GB drive is also too tight once formatting overhead is counted.

### Copying to the drive — do not use `/MT`

Copy the folder **single-threaded**. Multithreaded robocopy interleaves writes,
which collapses sequential throughput on flash media. Measured on the same USB
stick with the same 6.9 GB kit:

| Method | Throughput | Time for 6.9 GB |
|---|---:|---:|
| `robocopy /MT:8` | 0.4 MB/s | ~4.5 hours |
| plain `robocopy` / Explorer drag | 3.9 MB/s | ~30 min |

```powershell
robocopy D:\ASH_OFFLINE_KIT E:\ASH_OFFLINE_KIT /E
```

Cheap drives also burst fast and then collapse once their SLC cache fills, so
judge the speed from the *second* gigabyte, not the first. If a copy is crawling
at well under 1 MB/s, check that nothing passed `/MT`.

---

## 2. Requirements on the offline machine

| | |
|---|---|
| OS | Windows 10/11, **x64** |
| Python | **3.11** — the wheels are compiled for cp311 and will not install on 3.12/3.13 |
| Disk | ~12 GB free (app + venv + model) |
| RAM | 8 GB minimum for a 7B model; 16 GB comfortable |
| Admin rights | only if Python or Ollama still need installing |

The wheel/interpreter match is checked by `install_offline.ps1` and it will
stop rather than install a broken environment.

---

## 3. Building the kit (online machine), step by step

The build machine must be the **same OS and Python minor version** as the
offline machine — Windows x64 + Python 3.11.

1. Install Ollama and pull the model you want to ship:
   ```powershell
   ollama pull mistral:latest
   ```
2. Warm the HuggingFace cache once, so the build has something to copy:
   ```powershell
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"
   ```
3. Build:
   ```powershell
   powershell -ExecutionPolicy Bypass -File offline\build_usb.ps1 -Out D:\ASH_OFFLINE_KIT
   ```

Useful flags:

```powershell
# offline box already has Python + Ollama: save ~1.3 GB
.\offline\build_usb.ps1 -SkipInstallers

# ship a smaller model instead
.\offline\build_usb.ps1 -Models qwen2.5:3b

# ship two, so you can A/B on the offline box
.\offline\build_usb.ps1 -Models mistral:latest,qwen2.5:3b

# no voice output wanted: save ~340 MB
.\offline\build_usb.ps1 -SkipKokoro

# no always-on daemon wanted: drop the sensor extras (opencv, mss, ...)
.\offline\build_usb.ps1 -SkipDaemonExtras
```

The build is re-runnable. Anything already downloaded is kept, so a failed run
resumes cheaply — downloads land in a `.part` file and are only renamed on
success, so an interrupted build never leaves a truncated installer behind.

### Rehearse it before you walk it over

`offline\simulate_airgap.ps1` runs the entire drive workflow on the build
machine with outbound HTTP pointed at a dead port, so any step that still
secretly needs the internet fails here instead of on the machine you cannot fix:

```powershell
powershell -ExecutionPolicy Bypass -File offline\simulate_airgap.ps1
```

It stages the kit into a stand-in drive folder, installs from it into a scratch
directory, and runs the self-test — all with the network blocked. It aborts if
the block did not take effect, so a green run means something.

### Which model to ship

Ship a **non-reasoning** model. `mistral:latest` is the default and is what
`src/py/ash.py` expects.

Reasoning models (`qwen3`, `deepseek-r1`) work but will bite you: ASH's system
prompt is long, and the model can spend its entire `num_predict` budget on
chain-of-thought and return **empty content**. `tools/LLM.py` now detects this,
warns, and falls back to the reasoning text — but the output quality is worse.
If you must ship one, raise `max_tokens`.

---

## 4. Installing on the offline machine, step by step

1. Plug the drive in. Note its letter (`E:` below).
2. Open PowerShell **as Administrator** (needed only if Python/Ollama must be
   installed; otherwise a normal shell is fine).
3. ```powershell
   powershell -ExecutionPolicy Bypass -File E:\ASH_OFFLINE_KIT\scripts\install_offline.ps1
   ```

The installer does, in order:

| Step | What it does |
|---|---|
| 1 | Sanity-checks the kit and reads `build_manifest.json` |
| 2 | Installs Python 3.11 if missing; **hard-fails on a version mismatch** |
| 3 | Installs Ollama if missing |
| 4 | Copies `repo\` to `C:\ASH` |
| 5 | Drops the Kokoro TTS files into `C:\ASH\models\` |
| 6 | Creates `C:\ASH\.venv` and `pip install --no-index` from `wheels\` |
| 7 | Copies `hf_cache\` into `%USERPROFILE%\.cache\huggingface\hub` |
| 8 | Copies the Ollama blobs + manifest into Ollama's store, starts the server, confirms it sees the model |
| 9 | Writes `C:\ASH\.env` from the template |
| 10 | Runs `verify_offline.py` and reports |

If you installed Python or Ollama in this run, **open a new PowerShell** and
re-run the installer — PATH changes do not reach an already-running shell. The
installer is idempotent, so re-running is safe.

To install somewhere else: `-InstallDir D:\ASH`.
To wipe and reinstall: `-Force`.

---

## 5. Verifying

`verify_offline.py` runs automatically at the end of install. Re-run any time:

```powershell
C:\ASH\.venv\Scripts\python C:\ASH\verify_offline.py
C:\ASH\.venv\Scripts\python C:\ASH\verify_offline.py --quick   # skip generation
```

It checks, in order:

1. Python 3.11 x64, running inside the venv
2. Every required package imports; optional ones reported but not fatal
3. HF cache present; `SentenceTransformer` actually loads **with
   `HF_HUB_OFFLINE=1`**; spaCy models load; Kokoro files present
4. Ollama API reachable, the configured model is listed, and a real
   generation round-trip through `tools/LLM.py` returns non-empty text
5. `src.py.ash` imports, `ASH_LLM_MODE` is `ollama`, and one full cognitive
   turn produces a response
6. Confirms the machine genuinely has no outbound route (a warning, not a
   failure — but if this one passes, "it works offline" is proven rather than
   assumed)

Exit code 0 = ready. Non-zero = the FAIL lines tell you what to fix.

---

## 6. Running

```powershell
powershell -ExecutionPolicy Bypass -File C:\ASH\run_ash.ps1           # web UI, http://127.0.0.1:5000
powershell -ExecutionPolicy Bypass -File C:\ASH\run_ash.ps1 -Cli      # terminal REPL
powershell -ExecutionPolicy Bypass -File C:\ASH\run_ash.ps1 -Daemon   # always-on daemon
powershell -ExecutionPolicy Bypass -File C:\ASH\run_ash.ps1 -Verify   # re-run self-test
```

`run_ash.ps1` starts `ollama serve` first if it is not already up.

To run it at logon, from `C:\ASH`:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
```

> **Note:** `app.py` binds `0.0.0.0:5000`, so the web UI is reachable from the
> local network, not just localhost. On an isolated machine that is usually
> fine; if it is not, change the `host` argument at the bottom of `app.py` to
> `127.0.0.1`.

---

## 7. What works offline, and what does not

**Fully offline:**

- LLM generation — Ollama, local
- Intent classification and routing — `all-MiniLM-L6-v2` from the local cache
- Sentiment — local DistilBERT
- NER / entity memory — local spaCy
- Episodic, core and entity memory, concept graph, consolidation — local files
- Text-to-speech — Kokoro ONNX, local
- All file tools, calculator, date/time
- The Flask web UI and the daemon

**Does not work offline (by design, and it degrades rather than crashes):**

- `tools/mcp_servers.json` configures a weather MCP server on
  `http://localhost:8931/sse`. In practice `ash.py` calls
  `load_mcp_servers("mcp_servers.json")` with a path relative to the working
  directory, so the file at `tools/` is never found and the loader logs
  "No mcp_servers.json found -- skipping". Nothing to do; if you ever fix that
  path, nothing will be listening on 8931 offline
  and moves on. Delete the entry to silence it.
- `utils/google.py`, `utils/weather.py`, `utils/yt.py` are network tools. They
  are dead code — nothing in the runtime imports them — but do not wire them in.
- Speech-to-text falls back to Google's recognizer, which **uploads audio**.
  Install `faster-whisper` for local ASR. It is not in the kit by default
  because it pulls a large model of its own; add it with:
  ```powershell
  # on the ONLINE machine
  python -m pip download faster-whisper -d D:\ASH_OFFLINE_KIT\wheels
  ```
  and fetch a Whisper CT2 model into the HF cache before building.
- Vision (`google/vit-base-patch16-224`) is not in the kit. Without it the
  `VisionExtractor` silently falls back to a color/edge descriptor. To include
  it, load it once online and add its repo to `$HF_REPOS` in `build_usb.ps1`.

---

## 8. Configuration

`C:\ASH\.env`, written from `scripts\ash.env.template`:

| Variable | Default | Why it matters |
|---|---|---|
| `ASH_LLM_MODE` | `ollama` | `openrouter`/`local` need the network. Leave this alone. |
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/chat` | Ollama's real default port. |
| `OLLAMA_MODEL` | `mistral:latest` | Must match a name in `ollama list`. |
| `HF_HUB_OFFLINE` | `1` | Without it, every model load stalls trying to reach huggingface.co. |
| `TRANSFORMERS_OFFLINE` | `1` | Same, for the `transformers` code paths. |
| `ASH_EMBED_MODEL` | `all-MiniLM-L6-v2` | Must be in the shipped HF cache. |
| `ASH_AI_BASE` | `C:\ASH\data` | Where `intents.json` and the embedding cache live. |
| `ASH_CONCEPTS` | `1` | `0` disables the associative concept layer. |

`OPENROUTER_API_KEY` is deliberately absent. If you add it **and** set
`ASH_LLM_MODE=openrouter`, ASH will try to reach the internet.

---

## 9. Troubleshooting

**`pip install` fails with "No matching distribution found"**
The offline machine's Python is not 3.11, or not x64. Check
`build_manifest.json` → `build_python` and install the matching version from
`installers\`.

**`ollama list` does not show the model after install**
Ollama is reading a different store than the installer wrote to. The Ollama app
can be pointed at a custom model directory and keeps that choice in **its own
settings database**, not in an environment variable any shell can see — so
`~/.ollama/models` is not necessarily where the blobs belong.

`install_offline.ps1` handles this automatically by reading the store out of
Ollama's own startup log, but if it still lands wrong, find the live store
yourself:

```powershell
Select-String "$env:LOCALAPPDATA\Ollama\server*.log" -Pattern 'OLLAMA_MODELS:(\S+)' | Select-Object -Last 1
```

(Paths are escaped in that log, so `D:\\models` means `D:\models`.) Or, if a
model is currently loaded, look for the `...\blobs\sha256-...` path in the
runner's command line:

```powershell
Get-CimInstance Win32_Process -Filter "Name like 'ollama%'" | Select-Object CommandLine
```

Then copy the kit's `ollama_models\*` into that directory, or set
`OLLAMA_MODELS` to it and restart Ollama.

**ASH starts but every reply is a connection error**
Ollama is not running. `run_ash.ps1` starts it, but if you launched `app.py`
directly, run `ollama serve` first.

**Replies come back empty**
A reasoning model spent its whole token budget thinking. Switch
`OLLAMA_MODEL` to `mistral:latest`, or raise the token budget.

**Model loading hangs for 30+ seconds then works**
`HF_HUB_OFFLINE` is not set. Confirm `C:\ASH\.env` exists and that you are
launching from `C:\ASH` so `load_dotenv()` finds it.

**`ImportError: DLL load failed` from onnxruntime, torch or opencv**
The Microsoft Visual C++ 2015-2022 redistributable is missing. The bundled
Python installer normally brings it, but a stripped Windows image may not have
it. Grab `vc_redist.x64.exe` from
`https://aka.ms/vs/17/release/vc_redist.x64.exe` on the online machine, drop it
in `installers\`, and run it on the offline box before re-running the install.

**`verify_offline.py` warns "this machine CAN reach the internet"**
Exactly what it says — the run did not prove offline operation, because the
network was there to fall back on. Pull the cable and re-run.

---

## 10. Updating an already-installed machine

Rebuild the kit, copy it over, and re-run `install_offline.ps1`. It updates in
place and leaves `C:\ASH\.env` and the memory files alone. Pass `-Force` only
if you want a clean wipe — that discards accumulated memory.

To ship **only** new code (no models, no wheels), rebuild with:

```powershell
.\offline\build_usb.ps1 -SkipInstallers -SkipWheels -SkipModels -SkipHfCache -SkipKokoro -SkipDaemonExtras
```

The kit drops to about **5 MB** and fits on any drive. The installer skips every
section that is not present, so this works as a code-only update.
