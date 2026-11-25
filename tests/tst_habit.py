"""tst_habit.py -- open Outlook, be offered Word if the two usually go together; yes opens it. Fake executors only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_habit.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon.routines import Routines, RoutinesConfig  # noqa: E402
from src.senses.base import Modality, SensorEvent  # noqa: E402

check = tw.check


def win(process):
    return SensorEvent(source="active_window", modality=Modality.TEXT, text="", meta={"process": process})


def seeded(days=3):
    rt = tm.rig()
    rt.desktop = td.drig()
    rt.desktop.ex_.procs.discard("winword.exe")              # Word is not open: that is why it is offered
    rt.routines = Routines(RoutinesConfig(), clock=lambda: rt.clock())
    rt.routines.together = {"Outlook + Word": {"apps": ["Outlook", "Word"], "days": [f"2026-11-{10 + i}" for i in range(days)], "asked": False}}
    return rt


def main():
    print("\n=== H1. The offer ===")
    rt = seeded()
    sunk = []
    rt.response_sink = lambda t, m: sunk.append(t)
    rt.submit(win("outlook.exe"))
    rt._tick()
    check("a sitting that begins with Outlook is offered Word", sunk and sunk[-1] == "You usually have Word going with Outlook. Open it?", str(sunk))
    used, said = rt.say("yes")
    check("a yes opens it (through the desktop), no model", not used and rt.desktop.ex_.opened == ["winword"], f"{used} {rt.desktop.ex_.opened}")
    sunk.clear()
    rt.submit(win("outlook.exe"))
    rt._tick()
    check("not offered again the same day", not sunk, str(sunk))

    rt = seeded()
    rt.response_sink = lambda t, m: sunk.append(t)
    sunk.clear()
    rt.submit(win("outlook.exe"))
    rt._tick()
    used, said = rt.say("no")
    check("and a no opens nothing", rt.desktop.ex_.opened == [])

    rt = seeded()
    rt.desktop.ex_.procs.add("winword.exe")
    sunk.clear()
    rt.response_sink = lambda t, m: sunk.append(t)
    rt.submit(win("outlook.exe"))
    rt._tick()
    check("Word already open: nothing to offer", not sunk, str(sunk))

    rt = seeded(days=2)
    sunk.clear()
    rt.response_sink = lambda t, m: sunk.append(t)
    rt.submit(win("outlook.exe"))
    rt._tick()
    check("two days is not a habit: nothing said", not sunk, str(sunk))

    rt = seeded()
    sunk.clear()
    rt.response_sink = lambda t, m: sunk.append(t)
    rt.gate.muted = True
    rt.submit(win("outlook.exe"))
    rt._tick()
    check("muted: nothing said", not sunk, str(sunk))
    rt = seeded()
    rt.desktop = None
    sunk.clear()
    rt.response_sink = lambda t, m: sunk.append(t)
    rt.submit(win("outlook.exe"))
    rt._tick()
    check("no desktop control: nothing offered that cannot be done", not sunk, str(sunk))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
