"""
tst_candidates_persist.py -- a name seen once before a restart and once after becomes an entity.

A name has to come up in two different sentences before it becomes an entity (one-off capitalised words had been 309 of 362 entities), within
30 days. The candidates were only in RAM, and the daemon restarts often -- so the count started again at every restart. They are kept now,
with a hash that means the same in every process (the built-in one does not).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_candidates_persist.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def router(path):
    from src.memory.memory_router import MemoryRouter
    r = MemoryRouter.__new__(MemoryRouter)            # the candidate counting needs nothing else
    r.config = {"entity_min_mentions": 2, "entity_candidate_days": 30.0, "entity_candidates_path": path}
    r._candidates = {}
    r._cand_saved = 0.0
    r._cand_path = path
    r._load_candidates()
    return r


def main():
    print("\n=== C1. Across a restart ===")
    p = os.path.join(tempfile.mkdtemp(), "c.json")
    a = router(p)
    check("the first sighting is not enough", a._recurs("Lieth", "I met Lieth at the lab") is False)
    a._save_candidates(time.time() + 100)             # (the save is throttled to once in 20 s: force it past that)
    check("it was written", os.path.exists(p) and "lieth" in json.load(open(p, encoding="utf-8")))
    import shutil
    first_only = p + ".first"
    shutil.copyfile(p, first_only)                    # the file as it was after one sighting
    b = router(p)                                     # a restart
    check("the second, in a different sentence, after a restart, is", b._recurs("Lieth", "Lieth is coming on Friday") is True)
    c = router(first_only)
    check("the SAME sentence again is not a second sighting", c._recurs("Lieth", "I met Lieth at the lab") is False)
    check("the hash is stable (the same in every process)", json.load(open(p, encoding="utf-8"))["lieth"][0][1] == __import__("hashlib").sha1(b"i met lieth at the lab").hexdigest()[:12])
    old = os.path.join(tempfile.mkdtemp(), "o.json")
    json.dump({"omar": [[time.time() - 40 * 86400, "abc"]]}, open(old, "w"))
    check("a sighting older than 30 days is forgotten on load", router(old)._candidates == {})
    open(old, "w").write("{ torn")
    check("a torn file is not a crash", router(old)._candidates == {})
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
