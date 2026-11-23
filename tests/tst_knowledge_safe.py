"""
tst_knowledge_safe.py -- what ASH knows (people, relations, tastes, facts) survives the power going.

Knowledge was saved with a plain temp-and-replace (no fsync, no backup, a temp name only per process) and loaded with a bare json.load, so a
torn or empty file after a power cut meant starting empty -- exactly what memory/atomic.py was written for after it happened to the others.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_knowledge_safe.py
"""

import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def main():
    from src.daemon.wondering import Knowledge
    print("\n=== K1. Saving and loading ===")
    p = os.path.join(tempfile.mkdtemp(), "knowledge.json")
    k = Knowledge(p)
    k.facts["wifi"] = "hunter2"
    k.tastes["khalid"] = {"name": "Khalid", "user": True, "items": {"jazz": {"item": "Jazz", "score": 0.8}}}
    k.save()
    check("it is saved", os.path.exists(p) and json.load(open(p, encoding="utf-8"))["facts"]["wifi"] == "hunter2")
    k.facts["door"] = "left"
    k.save()
    check("the last good copy is kept as .bak", os.path.exists(p + ".bak") and "door" not in json.load(open(p + ".bak", encoding="utf-8"))["facts"])
    with open(p, "w", encoding="utf-8") as fh:
        fh.write('{"people": {"omar"')                              # torn by a power cut
    again = Knowledge(p)
    check("a torn file: loaded from the backup, not empty", again.facts.get("wifi") == "hunter2" and again.tastes.get("khalid"), str(again.facts))
    check("...and the torn one is set aside, not left to repeat", os.path.exists(p + ".bad"))
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("")
    check("an empty file too", Knowledge(p).facts.get("wifi") == "hunter2")
    check("no file at all: empty, no error", Knowledge(os.path.join(tempfile.mkdtemp(), "none.json")).facts == {})

    print("\n=== K2. Saving from two threads ===")
    q = os.path.join(tempfile.mkdtemp(), "k.json")
    kk = Knowledge(q)
    errors = []

    def spam(tag):
        for i in range(60):
            try:
                kk.facts[f"{tag}{i}"] = str(i)
                kk.save()
            except Exception as e:                       # noqa: BLE001
                errors.append(repr(e))
    ts = [threading.Thread(target=spam, args=(t,)) for t in "abc"]
    [t.start() for t in ts]
    [t.join() for t in ts]
    kk.save()
    check("no error", not errors, str(errors[:2]))
    check("the file is whole and has everything", len(json.load(open(q, encoding="utf-8"))["facts"]) == 180)
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
