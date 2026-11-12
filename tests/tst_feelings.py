"""
tst_feelings.py -- BMO's moods: feelings that last, from what happened, by rule.

  M1  what moves them: thanks, praise, laughing up; "you're useless", "shut up" down; ordinary talk not
  M2  they last, and fade: hours later a little is left, a day later it's gone; kept across a restart
  M3  energy: sleepy late at night; hours alone in the day -> bored
  M4  "how are you feeling?" answered from them, with the reason; what is not that question
  M5  in the runtime: by rule, no model; a guest's words don't move them; the face wears a feeling that has held

Run:  PYTHONIOENCODING=utf-8 python tests/tst_feelings.py
"""

import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.feelings import Feelings  # noqa: E402

check = tw.check


def at(h, m=0, day=12):
    return datetime(2026, 11, day, h, m).timestamp()


def feel(start=None, path=None):
    t = [start if start is not None else at(10)]
    f = Feelings(path, clock=lambda: t[0])
    f.t = t
    return f


def m1_moved():
    print("\n=== M1. What moves them ===")
    f = feel()
    base = f.cheer
    for said, kind in [("thank you so much", "thanks"), ("good job ash", "praise"), ("hahaha", "laugh"),
                       ("you're useless", "hurt"), ("shut up", "hushed"), ("what time is it", None)]:
        check(f"{said!r} -> {kind}", f.hear(said) == kind)
    g = feel()
    for _ in range(3):
        g.hear("thanks")
    check("thanks lift it", g.cheer > base + 0.3, str(g.cheer))
    h = feel()
    for _ in range(3):
        h.hear("you're useless")
    check("being told off lowers it", h.cheer < -0.5 and h.label() in ("low", "grumpy"), f"{h.cheer} {h.label()}")


def m2_last_and_fade():
    print("\n=== M2. They last, and fade ===")
    path = os.path.join(tempfile.mkdtemp(), "feelings.json")
    f = feel(path=path)
    for _ in range(5):
        f.hear("you're the best")
    f.save()
    high = f.cheer
    again = Feelings(path, clock=lambda: f.t[0] + 3600)
    again._drift(f.t[0] + 3600)
    check("kept across a restart, a little faded an hour later", 0.1 < again.cheer < high and again.cheer > 0.5, f"{high} -> {again.cheer}")
    again._drift(f.t[0] + 30 * 3600)
    check("a day later it's nearly back to normal", abs(again.cheer - 0.1) < 0.05, str(again.cheer))
    check("a new day starts its count again", again.today == {})


def m3_energy():
    print("\n=== M3. Energy ===")
    f = feel(start=at(1, 30))
    f.hear("hi")
    f.t[0] += 3600
    check("late at night: sleepy", f.tick() == "sleepy", f"{f.energy}")
    g = feel(start=at(10))
    g.hear("hi")
    g.t[0] = at(14, 30)
    check("hours alone in the day: bored", g.tick() == "bored", f"{g.energy}")
    g.hear("hello")
    g.t[0] += 120
    check("talking to it again: not bored any more", g.tick() != "bored", g.label())


def m4_asking():
    print("\n=== M4. How are you feeling? ===")
    for said in ("how are you", "Ash, how are you feeling?", "how are you doing today", "are you ok?", "what's your mood",
                 "hey, how are you, ash?"):
        check(f"asked: {said!r}", Feelings.wants(said))
    for said in ("how are you so smart", "how are you going to do that", "I am feeling ok", "how do I look"):
        check(f"not asking: {said!r}", not Feelings.wants(said))
    f = feel()
    f.hear("thanks")
    f.hear("thank you")
    f.hear("thanks a lot")
    out = f.say()
    check("happy, and why", out == "Really good -- you've thanked me 3 times today." or out.startswith("Pretty happy -- you've thanked me"), out)
    g = feel(start=at(10))
    g.hear("hi")
    g.t[0] = at(15)
    check("bored, and for how long", g.say() == "A bit bored, honestly. It's been quiet for 5 hours.", g.say())
    h = feel(start=at(1))
    h.hear("hi")
    h.t[0] += 3600
    check("sleepy", h.say().startswith("Sleepy."))
    k = feel()
    for _ in range(4):
        k.hear("shut up")
    check("grumpy, and why", "grumpy" in k.say(), k.say())
    n = feel()
    check("nothing much: steady", n.say().startswith("I'm good, thanks."), n.say())


def m5_runtime():
    print("\n=== M5. In the runtime ===")
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    rt = te.rig(feelings=True, feelings_path=os.path.join(tempfile.mkdtemp(), "f.json"), with_face=True)
    n = len(rt.ash.calls)
    rt.say("thank you ash")
    rt.say("Ash, how are you feeling?")
    check("answered by rule, from what happened", "thanked me once" in rt.out[-1] and len(rt.ash.calls) == n + 1, rt.out[-1])
    before = rt.feelings.cheer
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="Ash, you're useless", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's words don't move them", rt.feelings.cheer == before)
    for _ in range(6):
        rt.say("Ash, you're the best")
    shown = []
    rt.face.on_feeling = lambda anim: shown.append(anim)
    for _ in range(8):
        rt.t[0] += 61
        rt._tick()
    check("a feeling that has held for five minutes is shown on the face", shown and shown[-1] in ("love", "very_happy"), str(shown))
    check("status", rt.status()["feelings"]["feeling"] in ("loved", "delighted"), str(rt.status()["feelings"]))


if __name__ == "__main__":
    for fn in (m1_moved, m2_last_and_fade, m3_energy, m4_asking, m5_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
