"""tst_undoapps.py -- "undo that" takes back the app ASH just opened or closed. Fake executors only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_undoapps.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402

check = tw.check


def rig(**cfg):
    rt = tm.rig()
    rt.desktop = td.drig(**cfg)
    rt.desktop.on_act = lambda label, fn: rt.undo.push(label, fn)
    return rt


def main():
    print("\n=== U1. Opening ===")
    rt = rig()
    used, said = rt.say("open Outlook")
    check("opened", rt.desktop.ex_.opened == ["outlook"] and not used)
    used, said = rt.say("undo that")
    check("undo closes what it opened (politely), by rule", rt.desktop.ex_.closed == ["outlook.exe"] and not used and said and said[-1] == "Undone: opening Outlook.", f"{rt.desktop.ex_.closed} {said}")
    used, said = rt.say("undo")
    check("and there is nothing more to undo", said and said[-1] == "There's nothing recent to undo.", str(said))

    print("\n=== U2. Closing ===")
    rt = rig()
    rt.desktop.ex_.procs.add("outlook.exe")
    rt.say("close Outlook")
    rt.say("yes")
    check("closed after the yes", rt.desktop.ex_.closed == ["outlook.exe"], str(rt.desktop.ex_.closed))
    used, said = rt.say("take that back")
    check("undo opens it again", rt.desktop.ex_.opened == ["outlook"] and said and said[-1] == "Undone: closing Outlook.", f"{rt.desktop.ex_.opened} {said}")

    print("\n=== U3. A failed action leaves nothing to undo; a routine is not one step ===")
    rt = rig()
    rt.desktop.ex_.fail = True
    rt.say("open Outlook")
    used, said = rt.say("undo that")
    check("it did not open: nothing to undo", said and said[-1] == "There's nothing recent to undo.", str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
