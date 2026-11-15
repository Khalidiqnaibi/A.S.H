"""
tst_phrasing.py -- the same acknowledgement twice in a row changes its opening word; the first time it is exactly as written.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_phrasing.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.phrasing import Fresh  # noqa: E402
from src.daemon.reminders import Reminders  # noqa: E402

check = tw.check


def f1_fresh():
    print("\n=== F1. Fresh ===")
    f = Fresh()
    check("the first time: as written", f.apply("Opening Chrome.", 0) == "Opening Chrome.")
    a = f.apply("Opening Word.", 30)
    b = f.apply("Opening Excel.", 60)
    check("the next ones change their opening, each differently", a != "Opening Word." and b != "Opening Excel."
          and a.split()[0] != b.split()[0] and a.endswith("Word.") and b.endswith("Excel."), f"{a} / {b}")
    check("after a quiet spell, back to the plain line", f.apply("Opening Notepad.", 60 + 400) == "Opening Notepad.")
    check("a line it has no alternative for is left alone", f.apply("Hello there.", 1) == "Hello there.")
    check("each kind of line is counted on its own", Fresh().apply("Switching to Word.", 0) == "Switching to Word.")
    g = Fresh()
    g.apply("Okay, one.", 0)
    seen = {g.apply("Okay, again.", 10 * i).split()[0] for i in range(1, 8)}
    check("it cycles through the alternatives rather than picking at random", len(seen) == 3, str(seen))


def f2_reminders():
    print("\n=== F2. Reminders and the desktop use it ===")
    t = [1_800_000_000.0]
    r = Reminders(path=None, clock=lambda: t[0])
    a = r.hear("set a timer for 5 minutes")
    t[0] += 20
    b = r.hear("set a timer for 7 minutes")
    check("two timers in a row do not both open the same way", a.startswith("Okay, a 5-minute timer") and not b.startswith("Okay,")
          and "7-minute timer" in b, f"{a} / {b}")
    t[0] += 1000
    c = r.hear("set a timer for 9 minutes")
    check("later, plain again", c.startswith("Okay, a 9-minute timer"), c)


if __name__ == "__main__":
    f1_fresh()
    f2_reminders()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
