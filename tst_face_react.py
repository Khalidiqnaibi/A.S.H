"""
tst_face_react.py -- the face shows what ASH is doing.

  F1  Which expression goes with which line
  F2  Quiet hours: sleepy, and awake again after; away and quiet together
  F3  The runtime with the face attached: a question looks curious, a
      reminder alerts, a game win is very happy, an answer nods, the quiet
      hours make it sleepy -- and the lines still reach the user and the log

Run:  PYTHONIOENCODING=utf-8 python tst_face_react.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.face import BUILTINS  # noqa: E402
from src.face.bridge import FaceBridge, attach_to_daemon  # noqa: E402

check = tw.check
E = FaceBridge.expression_for


class Player:
    def __init__(self):
        self.played, self.idle = [], []
        self.current = None

    def play(self, name, force=False):
        self.played.append(name)
        return True

    def set_idle(self, name):
        self.idle.append(name)

    def stop(self):
        pass

    def set_drives(self, d):
        pass

    def status(self):
        return {}


def f1_mapping():
    print("\n=== F1. Which expression ===")
    cases = [
        ("Reminder: call mom.", {"pathway": "reminder"}, "alert"),
        ("chrome has closed.", {"pathway": "watch"}, "alert"),
        ("Who is Omar to you?", {"attention": {"kind": "wonder"}}, "curious"),
        ("Got it -- Omar is your cousin. Next: What does ZXQ stand for?", {"outcome": "batch"}, "curious"),
        ("Good morning. Today: your exam.", {"attention": {"kind": "briefing"}}, "happy"),
        ("A SQL query walks into a bar...", {"attention": {"kind": "moment"}}, "happy"),
        ("Yes! 20 -- you got it in 3 tries. Another one?", {"outcome": "play"}, "very_happy"),
        ("Higher.", {"outcome": "play"}, "skeptical"),
        ("Hm, I have Lina down as your sister.", {"outcome": "play"}, "skeptical"),
        ("Okay, game over. That was fun.", {"outcome": "play"}, "happy"),
        ("Okay -- I'm thinking of a number from 1 to 50. Guess!", {"outcome": "play"}, "curious"),
        ("Got it -- Omar is your cousin.", {"outcome": "answered"}, "acknowledge"),
        ("Opening outlook.", {"outcome": "desktop"}, "acknowledge"),
        ("Do you want tea?", {"addressed": False}, "curious"),
        ("It's 14:05.", {"addressed": True, "pathway": "fast"}, None),
        ("", {}, None),
    ]
    for text, meta, want in cases:
        check(f"{text[:30]!r} -> {want}", E(text, meta) == want, str(E(text, meta)))
    used = {w for _, _, w in cases if w}
    check("every expression used exists", used <= set(BUILTINS), str(used - set(BUILTINS)))


def f2_quiet():
    print("\n=== F2. Quiet hours ===")
    p = Player()
    b = FaceBridge(p)
    b.on_quiet_hours(True)
    check("sleepy in quiet hours", p.idle == ["sleepy"])
    b.on_quiet_hours(True)
    check("not re-sent while it stays quiet", p.idle == ["sleepy"])
    b.on_quiet_hours(False)
    check("awake after", p.idle == ["sleepy", "idle"])
    b.on_user_away(True)
    b.on_quiet_hours(True)
    b.on_quiet_hours(False)
    check("still sleepy while away", p.idle[-1] == "sleepy")
    b.on_user_away(False)
    check("and awake when back", p.idle[-1] == "idle")
    b.on_quiet_hours(True)
    b.on_user_away(True)
    b.on_user_away(False)
    check("back during quiet hours: still sleepy", p.idle[-1] == "sleepy")
    b = FaceBridge(p, enabled=False)
    n = len(p.played)
    b.on_line("Who is Omar to you?", {"attention": {"kind": "wonder"}})
    check("a disabled bridge plays nothing", len(p.played) == n)


def f3_runtime():
    print("\n=== F3. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    T0 = time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(quiet_hours=(23, 7)),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0, wonder=True,
                             wonder_cfg={"min_gap_s": 0}, wonder_path=os.path.join(d, "w.json"),
                             knowledge_path=os.path.join(d, "k.json"), reminders=True, play=True,
                             remarks_dir=os.path.join(d, "remarks")),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    rt.reminders.clock = lambda: clock[0]
    p = Player()
    attach_to_daemon(rt, p)

    def say(text):
        clock[0] += 2
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    def step(dt, idle_s, ticks=1):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    say("set a timer for 1 minutes")
    check("a reply the face nods to", "acknowledge" in p.played, str(p.played))
    step(61, 300.0)
    check("a timer going off: alert", p.played[-1] == "alert" and out[-1] == "Your 1-minute timer is done.",
          str(p.played[-3:]))
    say("I told Omar about it")
    for _ in range(15):
        step(10, 0.0)
    step(15, 25.0, ticks=2)
    check("a question at a free moment: curious", "Who is Omar to you?" in out and "curious" in p.played,
          str(p.played[-4:]))
    say("let's play the number game")
    s = rt.play.game.secret
    say(str(s))
    check("a game win: very happy", p.played[-1] == "very_happy", str(p.played[-3:]))
    check("lines still reach the user", out[-1].startswith(f"Yes! {s}"))
    check("and the remark log", any(r.get("text", "").startswith("Yes!") for r in rt.remarks.load(1, clock[0])))
    clock[0] = time.mktime((2026, 3, 10, 23, 30, 0, 0, 0, -1))
    step(1, 5000.0)
    check("quiet hours: sleepy", p.idle[-1] == "sleepy", str(p.idle))
    clock[0] = time.mktime((2026, 3, 11, 8, 0, 0, 0, 0, -1))
    step(1, 30.0)
    check("morning: awake", p.idle[-1] == "idle", str(p.idle))


if __name__ == "__main__":
    for fn in (f1_mapping, f2_quiet, f3_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
