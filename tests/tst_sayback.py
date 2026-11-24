"""tst_sayback.py -- a command the recogniser was unsure of is said back before it opens anything.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_sayback.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_desktop as td  # noqa: E402

check = tw.check


def s1_desktop():
    print("\n=== S1. The desktop ===")
    d = td.drig()
    check("a sure command just happens", d.hear("open Outlook", conf=-0.2) == "Opening Outlook." and d.ex_.opened == ["outlook"])
    d = td.drig()
    r = d.hear("open Outlook", conf=-0.9)
    check("an unsure one is said back, nothing opened", r == "Did you say “open Outlook”?" and d.ex_.opened == [], f"{r} {d.ex_.opened}")
    check("'yes' does it", d.hear("yes") == "Opening Outlook." and d.ex_.opened == ["outlook"])
    d = td.drig()
    d.hear("open Outlook", conf=-0.9)
    check("'no' drops it", d.hear("no, never mind") == "Okay, never mind." and d.ex_.opened == [])
    check("and it is asked only once, the answer does not linger", d.hear("yes") is None and d.ex_.opened == [])
    d = td.drig()
    d.hear("open Outlook", conf=-0.9)
    d.t[0] += 500
    check("a late 'yes' is not an answer", d.hear("yes") is None and d.ex_.opened == [])
    d = td.drig()
    check("not an app of ours: nothing to ask", d.hear("open the flibbertigibbet", conf=-0.9) is None)
    d = td.drig()
    check("no confidence known (typed, console): never asked", d.hear("open Outlook") == "Opening Outlook.")


def s2_runtime():
    print("\n=== S2. In the runtime ===")
    import tst_modellast as tm
    from src.senses.base import Modality, SensorEvent
    rt = tm.rig()
    rt.desktop = td.drig()

    def say(text, conf=None):
        meta = {"asr_conf": conf} if conf is not None else {}
        rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text=text, addressed=True, salience=0.9, urgency=0.6, meta=meta))
        rt._tick()

    used, said = rt.say("hello there")                        # warm the rig
    n = len(rt.desktop.ex_.opened)
    say("open Outlook", -1.1)
    check("an unsure voice command is not acted on", len(rt.desktop.ex_.opened) == n)
    say("yes")
    check("'yes' to the question does it", rt.desktop.ex_.opened[n:] == ["outlook"], str(rt.desktop.ex_.opened))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    s1_desktop()
    s2_runtime()
