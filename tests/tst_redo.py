"""
tst_redo.py -- "do that again": the last thing a rule did, run once more.

Follow-ups with no noun in them ("again", "one more time", "do that again") have nothing to be resolved against, so they reached the model, which has
no idea what was meant. The line a rule just answered is kept (ten minutes), and put back through the same rules.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_redo.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check


def main():
    import tst_modellast as tm
    import tst_desktop as td
    print("\n=== R1. Again ===")
    rt = tm.rig()
    rt.desktop = td.drig()
    used, said = rt.say("open outlook")
    check("(the first time it opens)", rt.desktop.ex_.opened == ["outlook"] and not used)
    for phrase in ("do that again", "one more time", "again", "same again", "Ash, do it again please"):
        n = len(rt.desktop.ex_.opened)
        used, said = rt.say(phrase)
        check(f"{phrase!r} does it once more, no model", not used and len(rt.desktop.ex_.opened) == n + 1 and rt.desktop.ex_.opened[-1] == "outlook",
              f"{used} {rt.desktop.ex_.opened[-2:]} {said[-1:]}")
    fresh = tm.rig()
    used, said = fresh.say("do that again")
    check("with nothing done yet it is not taken (the model or a game's own 'again' answers)", used or said, str(said))
    rt2 = tm.rig()
    rt2.desktop = td.drig()
    rt2.say("open outlook")
    rt2.clock = (lambda c=rt2.clock: c() + 3600)
    n = len(rt2.desktop.ex_.opened)
    rt2.say("do that again")
    check("an hour later it is stale: not redone", len(rt2.desktop.ex_.opened) == n)
    rt3 = tm.rig()
    rt3.desktop = td.drig()
    rt3.say("open outlook")
    rt3.say("do that again")
    n = len(rt3.desktop.ex_.opened)
    rt3.say("do that again")
    check("again and again", len(rt3.desktop.ex_.opened) == n + 1)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
