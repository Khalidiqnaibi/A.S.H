"""
tst_salience.py -- a surprising turn is laid down stronger, a routine one a little weaker.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_salience.py
"""

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain.salience import remembered_importance as r  # noqa: E402

check = tw.check


def s1_values():
    print("\n=== S1. The weights ===")
    check("a surprise raises it", r(0.5, 0.9) > 0.6, str(r(0.5, 0.9)))
    check("a novel input raises it a little", 0.5 < r(0.5, 0.0, 0.8) < 0.68, str(r(0.5, 0.0, 0.8)))
    check("a routine, familiar turn is a little lower", r(0.5, 0.02, 0.05) == 0.4, str(r(0.5, 0.02, 0.05)))
    check("a middling one is left alone", r(0.5, 0.2, 0.3) >= 0.5)
    check("never past the cap, so it is not forced into every prompt", r(0.5, 1.0, 1.0) <= 0.68)
    check("asked for on purpose (0.9): never lowered", r(0.9, 0.0, 0.0) == 0.9 and r(0.9, 1.0, 1.0) >= 0.9)
    check("junk in, sane out", 0.2 <= r(0.5, None, None) <= 0.8 and r(0.5, 9, -3) <= 0.68)


def s2_wired():
    print("\n=== S2. The brain uses it when it lays a turn down ===")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "brain", "brain.py"), encoding="utf-8").read()
    check("commit_turn is given the surprise-weighted importance",
          re.search(r"importance=remembered_importance\(importance,\s*prediction\.effective_surprise\(\)", src) is not None)


if __name__ == "__main__":
    s1_values()
    s2_wired()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
