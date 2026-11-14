"""
tst_atomic.py -- the edges of the crash-safe write (src/memory/atomic.py).

  A1  Two writers on one file take turns: no shared temp name, no stolen replace
  A2  The file is never missing: the previous copy is a second name for it, not a rename
  A3  A busy destination is retried (Windows: a scan or a reader has it open)
  A4  An empty file never becomes the backup; a locked file is not "damaged"

Run:  PYTHONIOENCODING=utf-8 python tests/tst_atomic.py
"""

import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.memory import atomic  # noqa: E402

check = tw.check


def a1_two_writers():
    print("\n=== A1. Two writers ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.json")
    errors = []

    def writer(tag):
        for i in range(150):
            try:
                atomic.write_json(p, {"who": tag, "i": i, "pad": "x" * 2000})
            except Exception as e:                      # noqa: BLE001
                errors.append(repr(e))
    ts = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b", "c")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check("three threads writing one file: no errors", not errors, str(errors[:2]))
    got = json.load(open(p))
    check("...and what is there is one whole write", got["who"] in ("a", "b", "c") and got["i"] == 149 and len(got["pad"]) == 2000)
    left = [f for f in os.listdir(d) if f.endswith(".tmp")]
    check("...and no temp file is left over", not left, str(left))


def a2_never_missing():
    print("\n=== A2. Never missing ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "y.json")
    atomic.write_json(p, {"v": 1})
    seen_missing = []
    real = atomic._replace

    def spying(src, dst, tries=6):
        if dst == p and not os.path.exists(p):
            seen_missing.append(1)
        return real(src, dst, tries)
    atomic._replace = spying
    try:
        atomic.write_json(p, {"v": 2})
    finally:
        atomic._replace = real
    check("while the backup is made the file is still there (it used to be renamed away first)", not seen_missing)
    check("the backup holds the previous write and the file the new one", json.load(open(p + ".bak"))["v"] == 1
          and json.load(open(p))["v"] == 2)

    # a write that fails half way leaves the file as it was
    class Bad:
        def __float__(self):
            raise ValueError("x")
    try:
        atomic.write_with(p, lambda f: (f.write("{\"v\": 3"), (_ for _ in ()).throw(RuntimeError("disk gone"))), "w")
    except RuntimeError:
        pass
    check("a write that dies half way leaves the file untouched", json.load(open(p))["v"] == 2)
    check("...and no temp file", not [f for f in os.listdir(d) if f.endswith(".tmp")])


def a3_busy():
    print("\n=== A3. A busy destination ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "z.json")
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2 and str(dst) == p:
            raise PermissionError(13, "in use")
        return real(src, dst)
    os.replace = flaky
    try:
        atomic.write_json(p, {"v": 1})
    finally:
        os.replace = real
    check("a destination that is busy twice is written on the third try", json.load(open(p))["v"] == 1, str(calls))

    def always(src, dst):
        raise PermissionError(13, "in use")
    os.replace = always
    try:
        try:
            atomic.write_json(os.path.join(d, "q.json"), {"v": 1})
            ok = False
        except PermissionError:
            ok = True
    finally:
        os.replace = real
    check("one that never frees up still raises in the end (and the caller says so once a minute)", ok)


def a4_loading():
    print("\n=== A4. Loading ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "w.json")
    atomic.write_json(p, {"v": 1})
    atomic.write_json(p, {"v": 2})
    open(p, "w").close()                                # an empty file (the power went)
    atomic.write_json(p, {"v": 3})                      # the next save
    check("an empty file is not rotated over the last good backup", json.load(open(p + ".bak")) == {"v": 2}
          or json.load(open(p + ".bak")) == {"v": 1}, str(open(p + ".bak").read()))

    atomic.write_json(p, {"v": 4})
    real_open = open

    def locked(path, mode="r", *a, **k):
        if str(path) == p and "r" in mode:
            raise PermissionError(13, "locked by a scan")
        return real_open(path, mode, *a, **k)
    import builtins
    builtins.open = locked
    try:
        got = atomic.load_json(p, default="none")
    finally:
        builtins.open = real_open
    check("a file that is locked for a moment is read from the backup this once", got in ({"v": 3}, {"v": 2}), str(got))
    check("...and is NOT moved aside as damaged", os.path.exists(p) and not os.path.exists(p + ".bad"))
    check("...and reads fine afterwards", atomic.load_json(p) == {"v": 4})


if __name__ == "__main__":
    a1_two_writers()
    a2_never_missing()
    a3_busy()
    a4_loading()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
