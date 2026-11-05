"""
tst_focus.py -- focus mode.

  F1  Turning it on: for a length, until a time, a plain "let me focus"; what
      is not a request; turning it off
  F2  The summary: nothing held, things held, extras
  F3  The runtime: nothing volunteered while focusing (questions, free
      moments, briefings, check-ins), what was asked for still happens, a
      question to ASH is answered, and the summary is said at the end -- on
      time, or when asked

Run:  PYTHONIOENCODING=utf-8 python tst_focus.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.focus import Focus  # noqa: E402

check = tw.check
T0 = time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))


def frig():
    t = [T0]
    f = Focus(clock=lambda: t[0])
    f.t = t
    return f


def f1_on_off():
    print("\n=== F1. On and off ===")
    f = frig()
    s = f.hear("I'm focusing for an hour")
    check("for an hour", s == "Focus mode until 15:00. I'll keep to myself; reminders still go off." and f.active(), s)
    f = frig()
    check("for 45 minutes", f.hear("focus mode for 45 minutes").startswith("Focus mode until 14:45"))
    f = frig()
    check("for half an hour", f.hear("i need to focus for half an hour").startswith("Focus mode until 14:30"))
    f = frig()
    check("until 4pm", f.hear("do not disturb until 4pm").startswith("Focus mode until 16:00"))
    f = frig()
    check("until 3 (the afternoon)", f.hear("don't disturb me until 3").startswith("Focus mode until 15:00"))
    f = frig()
    check("a plain 'let me focus' is an hour", f.hear("let me focus").startswith("Focus mode until 15:00"))
    f = frig()
    check("capped at eight hours", f.hear("focus for 20 hours") and f.until == T0 + 8 * 3600)
    for text in ("I'm focusing on the report", "focus on the left one", "what's the focus of this", "", None):
        f = frig()
        check(f"not a request: {str(text)[:30]!r}", f.hear(text) is None and not f.active())
    f = frig()
    f.hear("let me focus")
    f.t[0] += 600
    check("done focusing turns it off (the caller says the summary)",
          f.hear("I'm done focusing") is None and not f.active())
    check("off when it is not on: said", f.hear("focus off") == "Focus wasn't on.")
    f = frig()
    f.hear("focus for 30 minutes")
    f.t[0] += 31 * 60
    check("it ends on its own", not f.active())
    check("off: nothing", Focus(enabled=False).hear("let me focus") is None)


def f2_summary():
    print("\n=== F2. The summary ===")
    f = frig()
    f.hear("let me focus")
    f.t[0] += 25 * 60
    check("nothing held", f.summary([]) == "Focus done (25 min). Nothing came up.")
    f.hold("wonder")
    f.hold("moment")
    check("things held, and extras", f.summary(["2 questions are waiting"]) ==
          "Focus done (25 min). While you focused: I held back 2 things; 2 questions are waiting.")
    f.hear("let me focus")
    check("a new session starts a new count", sum(f.held.values()) == 0)


def rt_rig():
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=20),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0, wonder=True,
                             wonder_cfg={"min_gap_s": 0}, wonder_path=os.path.join(d, "w.json"),
                             knowledge_path=os.path.join(d, "k.json"), moments=True,
                             moments_cfg={"min_gap_s": 0}, reminders=True, focus=True),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    rt.reminders.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.clock, h.idle = rt, ash, out, clock, idle

    def say(text, addressed=True, source="console"):
        clock[0] += 2
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    def step(dt, idle_s, ticks=1):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def free():
        for _ in range(15):
            step(10, 0.0)
        step(15, 25.0, ticks=2)

    h.say, h.step, h.free = say, step, free
    h.said = lambda: [o[0] for o in out]
    return h


def f3_runtime():
    print("\n=== F3. The runtime ===")
    h = rt_rig()
    h.say("I told Omar about it")
    h.say("set a timer for 20 minutes")
    h.say("I'm focusing for an hour")
    check("on, without the model", h.said()[-1].startswith("Focus mode until") and len(h.ash.calls) == 1)
    n = len(h.out)
    for _ in range(3):
        h.free()
    check("nothing volunteered while focusing (no question, no joke, no check-in)", len(h.out) == n + 0
          or all(o[1].get("pathway") == "reminder" for o in h.out[n:]), str(h.said()[n:]))
    check("what it held is counted", h.rt.focus.stats["held"] >= 1, str(h.rt.focus.stats))
    from src.senses.base import Modality, SensorEvent
    before, n = h.rt.focus.stats["held"], len(h.out)
    h.rt.submit(SensorEvent(source="system", modality=Modality.TEXT, text="battery at 15% and unplugged",
                            salience=0.9, urgency=0.8))
    h.rt._tick()
    check("an unprompted remark the gate would allow is held, not said", len(h.out) == n
          and h.rt.focus.stats["held"] == before + 1, str(h.said()[n:]))
    h.step(20 * 60, 300.0)
    check("a timer asked for still goes off", any("Your 20-minute timer is done." in x for x in h.said()))
    h.say("what's the time")
    check("a question to ASH is answered", h.ash.calls[-1] == "what's the time")
    calls = len(h.ash.calls)
    h.say("I'm done focusing")
    s = h.said()[-1]
    check("'done focusing' is taken without the model", len(h.ash.calls) == calls)
    check("the summary when asked to stop", s.startswith("Focus done (") and "question" in s, s)
    h.free()
    check("and things are said again after", any(x.endswith("Who is Omar to you?") for x in h.said()),
          str(h.said()[-3:]))

    h = rt_rig()
    h.say("focus for 30 minutes")
    h.step(31 * 60, 300.0)
    check("the summary when it runs out on its own", h.said()[-1].startswith("Focus done (") and
          h.out[-1][1]["pathway"] == "focus", h.said()[-1])
    h.step(60, 300.0)
    check("only once", sum(x.startswith("Focus done") for x in h.said()) == 1)
    check("status carries focus", h.rt.status()["focus"]["sessions"] == 1)


if __name__ == "__main__":
    for fn in (f1_on_off, f2_summary, f3_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
