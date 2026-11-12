"""
tests/run_all.py -- run the test files one after another and say how it went.

    python tests/run_all.py                    every tst_*.py
    python tests/run_all.py vision daemon      only files whose name contains one of these words
    python tests/run_all.py --skip soak routing    everything except those
    python tests/run_all.py --list             what would run
    python tests/run_all.py --timeout 300      seconds allowed per file (default 900)
    python tests/run_all.py --logs DIR         keep each file's output in DIR (default: a temp folder)

Your memory is never touched: tests that load the real brain would write the memory files in the project folder
(entities.json, episodic_memory.jsonl, core_memory.json), so each test gets its own temp copy through
ASH_MEMORY_DIR (src/memory/paths.py). As a second guard the runner notes those three files before it starts and
puts them back byte for byte if anything changed them anyway. One at a time, so the timings mean something.

A file passes when it exits 0. "fail-lines" counts the checks a file printed as FAIL (or, for the end-to-end
evaluation tst_e2e, MISS -- its misses are findings, not failures; only its safety gate sets its exit code).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MEMORY_FILES = ("entities.json", "episodic_memory.jsonl", "core_memory.json")


def discover(only=(), skip=()):
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "tst_*.py")))
    if only:
        names = [n for n in names if any(w in n for w in only)]
    return [n for n in names if not any(w in n for w in skip)]


def snapshot():
    out = {}
    for f in MEMORY_FILES:
        p = os.path.join(ROOT, f)
        out[f] = open(p, "rb").read() if os.path.exists(p) else None
    return out


def put_back(before):
    """Restore any memory file a test left different. Returns the names it had to restore."""
    fixed = []
    for f, data in before.items():
        p = os.path.join(ROOT, f)
        now = open(p, "rb").read() if os.path.exists(p) else None
        if now != data:
            fixed.append(f)
            if data is None:
                os.remove(p)
            else:
                with open(p, "wb") as fh:
                    fh.write(data)
    return fixed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the A.S.H test files one after another.")
    ap.add_argument("words", nargs="*", help="only files whose name contains one of these")
    ap.add_argument("--skip", nargs="*", default=[], help="skip files whose name contains one of these")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--logs", default="")
    args = ap.parse_args(argv)

    names = discover(args.words, args.skip)
    if args.list:
        print("\n".join(names))
        return 0
    if not names:
        print("no test files match")
        return 2
    logs = args.logs or tempfile.mkdtemp(prefix="ash-tests-")
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    before = snapshot()
    rows = []
    t_all = time.time()
    try:
        for n in names:
            t0 = time.time()
            log = os.path.join(logs, n + ".log")
            # Each test gets its own copy of the memory files (ASH_MEMORY_DIR, src/memory/paths.py): the brain writes
            # there, never to yours -- even if the test is killed half-way.
            mem = tempfile.mkdtemp(prefix="ash-mem-")
            for f, data in before.items():
                if data is not None:
                    with open(os.path.join(mem, f), "wb") as fh:
                        fh.write(data)
            try:
                with open(log, "wb") as fh:
                    rc = subprocess.run([sys.executable, os.path.join(HERE, n)], cwd=ROOT, env=dict(env, ASH_MEMORY_DIR=mem),
                                        stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = 124
            text = open(log, "rb").read().decode("utf-8", "replace")
            fails = sum(1 for line in text.splitlines() if re.match(r"\s*(?:FAIL|MISS)\s", line))   # not "FAILED: none"
            secs = time.time() - t0
            rows.append((n, rc, fails, secs))
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {n:<26} rc={rc:<3} fail-lines={fails:<3} {secs:5.0f}s", flush=True)
    finally:
        fixed = put_back(before)
    bad = [r for r in rows if r[1] != 0]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} files passed in {time.time() - t_all:.0f}s; logs in {logs}")
    if fixed:
        print(f"restored {', '.join(fixed)} to how they were before the run")
    for n, rc, fails, _ in bad:
        print(f"  failed: {n} (rc={rc}, {fails} fail line(s)) -- see {os.path.join(logs, n + '.log')}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
