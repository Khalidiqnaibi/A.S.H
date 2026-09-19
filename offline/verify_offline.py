#!/usr/bin/env python
"""
A.S.H offline self-test.

Run this from the A.S.H install directory on the air-gapped machine:

    .venv\\Scripts\\python verify_offline.py          # full check, incl. a real LLM turn
    .venv\\Scripts\\python verify_offline.py --quick  # skip the slow generation checks

Exit code 0 means every REQUIRED check passed and ASH can hold a conversation
with no network. Optional checks (TTS, vision, daemon sensors) are reported
but never fail the run -- ASH degrades gracefully without them.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

# Force every HuggingFace code path to use the local cache only. Without this,
# transformers/sentence-transformers try to reach huggingface.co to revalidate
# the snapshot and hang for the full connect timeout on an air-gapped box.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

RESULTS: list[tuple[str, str, str, str]] = []  # (status, kind, name, detail)
PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
REQUIRED, OPTIONAL = "required", "optional"


def _enable_colour() -> bool:
    """The classic Windows console does not render ANSI unless we ask it to.
    If we cannot turn it on, print plain text rather than escape-code soup."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


COLOUR = _enable_colour()


def record(status: str, kind: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, kind, name, detail))
    colour = ""
    if COLOUR:
        colour = {PASS: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m", SKIP: "\033[90m"}.get(status, "")
    reset = "\033[0m" if colour else ""
    print(f"  {colour}[{status:4}]{reset} {name}" + (f"  --  {detail}" if detail else ""), flush=True)


def section(title: str) -> None:
    print(f"\n{'-' * 68}\n{title}\n{'-' * 68}", flush=True)


def check(name: str, kind: str = REQUIRED):
    """Decorator-ish helper: run fn, record PASS/FAIL, never raise."""
    def run(fn):
        try:
            detail = fn() or ""
            record(PASS, kind, name, str(detail))
            return True
        except Exception as e:
            record(FAIL if kind == REQUIRED else WARN, kind, name, f"{type(e).__name__}: {e}")
            return False
    return run


# =============================================================================
def check_interpreter() -> None:
    section("1. Interpreter")

    @check("python 3.11 x64")
    def _():
        import platform
        v = sys.version_info
        if (v.major, v.minor) != (3, 11):
            raise RuntimeError(f"expected 3.11, got {v.major}.{v.minor} -- wheels in the kit will not match")
        if platform.machine().lower() not in ("amd64", "x86_64"):
            raise RuntimeError(f"expected x64, got {platform.machine()}")
        return f"{v.major}.{v.minor}.{v.micro} {platform.machine()}"

    @check("running inside the venv", OPTIONAL)
    def _():
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("not in a venv -- packages went to the system interpreter")
        return sys.prefix


# =============================================================================
def check_packages() -> None:
    section("2. Python packages")

    required = [
        ("numpy", "numpy"), ("requests", "requests"), ("dotenv", "python-dotenv"),
        ("langchain_core", "langchain_core"), ("torch", "torch"),
        ("transformers", "transformers"), ("sentence_transformers", "sentence-transformers"),
        ("spacy", "spacy"), ("flask", "flask"), ("flask_socketio", "flask-socketio"),
        ("speech_recognition", "SpeechRecognition"), ("pydub", "pydub"),
        ("PyPDF2", "PyPDF2"), ("PIL", "Pillow"), ("mcp", "mcp"),
    ]
    optional = [
        ("kokoro_onnx", "kokoro-onnx (local TTS)"), ("soundfile", "soundfile"),
        ("psutil", "psutil (daemon sensors)"), ("mss", "mss (screen capture)"),
        ("cv2", "opencv-python (camera)"), ("sounddevice", "sounddevice (mic)"),
        ("faster_whisper", "faster-whisper (local ASR)"),
    ]

    for mod, label in required:
        @check(label)
        def _(mod=mod):
            m = __import__(mod)
            return getattr(m, "__version__", "")
    for mod, label in optional:
        @check(label, OPTIONAL)
        def _(mod=mod):
            m = __import__(mod)
            return getattr(m, "__version__", "")

    @check("torch is the CPU build", OPTIONAL)
    def _():
        import torch
        if "+cu" in torch.__version__:
            raise RuntimeError(f"{torch.__version__} is a CUDA build -- works, but ~2 GB larger than needed")
        return torch.__version__

    @check("numpy matches the pinned build")
    def _():
        # transformers 4.38.2 / sentence-transformers 2.6.1 are pinned against
        # numpy 1.26.x. An optional extra installed afterwards (opencv-python)
        # will happily pull numpy 2.x and uninstall 1.26.4, leaving an
        # environment that pip called a success and that fails at import time.
        import numpy
        major = int(numpy.__version__.split(".")[0])
        if major != 1:
            raise RuntimeError(
                f"numpy {numpy.__version__} -- expected 1.26.x. Something installed "
                f"after the main requirements replaced it; re-run the install with "
                f"scripts/build_constraints.txt present."
            )
        return numpy.__version__


# =============================================================================
def check_local_models(quick: bool) -> None:
    section("3. Local model assets (no network)")

    @check("HF cache present")
    def _():
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        if not hub.is_dir():
            raise RuntimeError(f"{hub} does not exist -- copy hf_cache\\ from the kit")
        repos = [p.name for p in hub.iterdir() if p.name.startswith("models--")]
        if not repos:
            raise RuntimeError(f"{hub} is empty")
        return ", ".join(repos)

    if quick:
        record(SKIP, REQUIRED, "sentence-transformers loads offline", "--quick")
    else:
        @check("sentence-transformers loads offline")
        def _():
            from sentence_transformers import SentenceTransformer
            name = os.environ.get("ASH_EMBED_MODEL", "all-MiniLM-L6-v2")
            t0 = time.time()
            m = SentenceTransformer(name)
            dim = m.get_sentence_embedding_dimension()
            v = m.encode(["offline smoke test"])
            return f"{name}, dim={dim}, {time.time() - t0:.1f}s, vec={v.shape}"

    @check("spaCy en_core_web_sm")
    def _():
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Khalid built ASH in Amman.")
        return f"{len(doc.ents)} entities on the probe sentence"

    @check("spaCy en_core_web_md", OPTIONAL)
    def _():
        import spacy
        spacy.load("en_core_web_md")
        return "loaded"

    @check("Kokoro TTS model files", OPTIONAL)
    def _():
        need = [Path("models/kokoro-v1.0.onnx"), Path("models/voices-v1.0.bin")]
        missing = [str(p) for p in need if not p.is_file()]
        if missing:
            raise RuntimeError("missing " + ", ".join(missing))
        return ", ".join(f"{p.name} {p.stat().st_size / 1e6:.0f} MB" for p in need)


# =============================================================================
def _ollama_base() -> str:
    url = os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434/api/chat"
    return url.split("/api/")[0]


def check_ollama(quick: bool) -> str | None:
    section("4. Ollama")
    model = (os.environ.get("OLLAMA_MODEL") or "mistral:latest").strip()
    base = _ollama_base()
    available: list[str] = []

    @check(f"ollama API reachable at {base}")
    def _():
        with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as r:
            data = json.load(r)
        available.extend(m["name"] for m in data.get("models", []))
        if not available:
            raise RuntimeError("server is up but has no models -- import ollama_models\\ from the kit")
        return ", ".join(available)

    @check(f"model '{model}' is present")
    def _():
        if model not in available:
            raise RuntimeError(
                f"not in {available}. Either import it from the kit, or point "
                f"OLLAMA_MODEL in .env at one of those."
            )
        return "ok"

    if quick:
        record(SKIP, REQUIRED, "ollama generates through tools.LLM", "--quick")
        return model

    @check("ollama generates through tools.LLM")
    def _():
        from langchain_core.messages import HumanMessage
        from tools.LLM import LLM
        llm = LLM(
            mode="ollama", temperature=0.0, max_tokens=2048,
            ollama_model=model, ollama_url=f"{base}/api/chat", timeout=300,
        )
        t0 = time.time()
        out = llm.invoke([HumanMessage(content="Reply with exactly one word: PONG")]).content
        if not out.strip():
            raise RuntimeError(
                "empty response. Reasoning models (qwen3, deepseek-r1) can spend the "
                "whole token budget thinking -- raise max_tokens or use mistral."
            )
        return f"{time.time() - t0:.1f}s -> {out.strip()[:60]!r}"

    return model


# =============================================================================
def check_ash(quick: bool) -> None:
    section("5. A.S.H runtime")

    @check("import src.py.ash")
    def _():
        t0 = time.time()
        from src.py.ash import ash, llm, ASH_LLM_MODE
        if llm is None:
            raise RuntimeError(f"no LLM backend was built (ASH_LLM_MODE={ASH_LLM_MODE})")
        if ASH_LLM_MODE != "ollama":
            raise RuntimeError(f"ASH_LLM_MODE={ASH_LLM_MODE} -- that backend needs the network")
        globals()["_ash"] = ash
        return f"mode={ASH_LLM_MODE}, {time.time() - t0:.1f}s cold start"

    if quick:
        record(SKIP, REQUIRED, "one full cognitive turn", "--quick")
        return

    @check("one full cognitive turn")
    def _():
        ash = globals().get("_ash")
        if ash is None:
            raise RuntimeError("ASH did not import, skipping")
        t0 = time.time()
        out = ash.run("Say hello in exactly five words.")
        text = getattr(out, "content", out)
        if not str(text).strip():
            raise RuntimeError("ASH produced an empty response")
        return f"{time.time() - t0:.1f}s -> {str(text).strip()[:80]!r}"


# =============================================================================
def check_isolation() -> None:
    section("6. Network isolation (informational)")

    @check("machine has no internet route", OPTIONAL)
    def _():
        try:
            s = socket.create_connection(("1.1.1.1", 443), timeout=3)
            s.close()
        except OSError:
            return "no outbound route -- this is the expected air-gapped state"
        raise RuntimeError("this machine CAN reach the internet; the checks above did not prove offline operation")

    @check("HF offline flags set")
    def _():
        flags = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        if flags["HF_HUB_OFFLINE"] != "1":
            raise RuntimeError(f"HF_HUB_OFFLINE={flags['HF_HUB_OFFLINE']} -- set it to 1 in .env")
        return str(flags)


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="A.S.H offline self-test")
    ap.add_argument("--quick", action="store_true", help="skip model loading and generation")
    args = ap.parse_args()

    # .env lives next to this file after install; load it so OLLAMA_* apply.
    try:
        from dotenv import load_dotenv
        if Path(".env").is_file():
            load_dotenv(".env", override=False)
    except Exception:
        pass

    print("=" * 68)
    print(" A.S.H OFFLINE VERIFICATION")
    print(f" cwd: {HERE}")
    print("=" * 68)

    check_interpreter()
    check_packages()
    check_local_models(args.quick)
    check_ollama(args.quick)
    check_ash(args.quick)
    check_isolation()

    failed = [r for r in RESULTS if r[0] == FAIL]
    warned = [r for r in RESULTS if r[0] == WARN]
    passed = [r for r in RESULTS if r[0] == PASS]

    print("\n" + "=" * 68)
    print(f" {len(passed)} passed   {len(warned)} warnings (optional)   {len(failed)} FAILED")
    print("=" * 68)
    for _, _, name, detail in warned:
        print(f"  warn: {name} -- {detail}")
    for _, _, name, detail in failed:
        print(f"  FAIL: {name} -- {detail}")

    if failed:
        print("\nASH is NOT ready. Fix the FAIL lines above and re-run.")
        return 1
    print("\nASH is ready to run fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
