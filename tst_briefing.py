"""
tst_briefing.py -- morning and evening briefings.

  B1  Asking for one: "brief me", "what's my day look like", "what did I do
      today"; and what is not a request
  B2  When: the morning and evening windows, once a day, kept across restarts
  B3  What: today's / tomorrow's events, yesterday's / today's work, questions
      waiting, extra parts, a failing part, nothing to say
  B4  Activity memory keeps time per day
  B5  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tst_briefing.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.activity import ActivityMemory, Topic  # noqa: E402
from src.daemon.briefing import BriefingConfig, Briefings, asked_for_briefing  # noqa: E402
from src.daemon.followups import FollowUps  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
DAY, HOUR = 86400.0, 3600.0


def at(day, hour, minute=0):
    return time.mktime((2026, 3, 10 + day, hour, minute, 0, 0, 0, -1))


def d(day):
    return time.strftime("%Y-%m-%d", time.localtime(at(day, 12)))


class Act:
    def __init__(self, **by):
        self.topics = {}
        for name, days in by.items():
            t = Topic(key=name, name=name.replace("_", "."), kind="subject")
            t.by_day = days
            self.topics[name] = t


def b1_asking():
    print("\n=== B1. Asking for one ===")
    for text, want in (("brief me", ""), ("ash, brief me please", ""), ("what's my day look like?", ""),
                       ("what does my day look like", ""), ("give me my morning briefing", "morning"),
                       ("what did I do today", "evening"), ("how was my day", "evening"),
                       ("what's on tomorrow", "evening"), ("evening briefing", "evening")):
        check(f"asks: {text!r}", asked_for_briefing(text) == want, repr(asked_for_briefing(text)))
    for text in ("what's the weather", "brief", "I need to brief my team about the plan for today and tomorrow",
                 "can you brief me on what happened with the deploy and the database migration last night", "", None):
        check(f"not a request: {str(text)[:30]!r}", asked_for_briefing(text) is None)


def b2_when():
    print("\n=== B2. When ===")
    t = [at(0, 8)]
    b = Briefings(BriefingConfig(), clock=lambda: t[0])
    check("08:00 is the morning", b.due() == "morning")
    b.mark("morning")
    check("once a day", b.due() == "")
    t[0] = at(0, 14)
    check("the afternoon is neither", b.due() == "")
    t[0] = at(0, 19)
    check("19:00 is the evening", b.due() == "evening")
    t[0] = at(1, 3)
    check("the night is neither", b.due() == "")
    t[0] = at(1, 7)
    check("a new day, a new morning", b.due() == "morning")
    check("morning off", Briefings(BriefingConfig(morning=False), clock=lambda: at(0, 8)).due() == "")
    check("all off", Briefings(BriefingConfig(enabled=False), clock=lambda: at(0, 19)).due() == "")
    p = os.path.join(tempfile.mkdtemp(), "b.json")
    b = Briefings(BriefingConfig(), path=p, clock=lambda: at(0, 8))
    b.mark("morning")
    check("kept across a restart", Briefings(BriefingConfig(), path=p, clock=lambda: at(0, 9)).due() == "")


def b3_what():
    print("\n=== B3. What ===")
    t = [at(0, 20)]
    fu = FollowUps(clock=lambda: t[0])
    fu.observe("I have an interview tomorrow at 10am")
    fu.observe("my exam is tomorrow")
    fu.observe("I have a meeting on Friday at 3pm")
    act = Act(notes={d(0): 300}, ASH={d(0): 1200}, brain_py={d(0): 2.5 * HOUR, d(-1): 3600})
    w = tw.rig()
    w.observe("I told Omar about it")
    w.observe("I told Sara about it")
    b = Briefings(BriefingConfig(), followups=fu, activity=act, wonder=w, clock=lambda: t[0])
    s = b.compose("evening")
    check("evening: the day's work, with time", "Today you spent most of your time on brain.py (2h30) and ASH (20 min)."
          in s, s)
    check("a short stretch is left out", "notes" not in s)
    check("tomorrow's events, in order", "Tomorrow: your exam and your interview at 10:00." in s
          or "Tomorrow: your interview at 10:00 and your exam." in s, s)
    check("not events on other days", "meeting" not in s)
    check("questions waiting", s.endswith("I have 2 questions for you when you have a minute."), s)
    t[0] = at(1, 8)
    act.topics["brain_py"].by_day[d(0)] = 2.5 * HOUR
    s = b.compose("morning")
    check("morning: today's events", s.startswith("Good morning. Today: ") and "interview at 10:00" in s, s)
    check("yesterday's work", "Yesterday you worked mostly on brain.py and ASH." in s, s)
    b.add_provider(lambda kind, now: "It looks sunny." if kind == "morning" else None)
    b.add_provider(lambda kind, now: 1 / 0)
    s = b.compose("morning")
    check("an extra part is added", "It looks sunny." in s, s)
    check("a failing part is left out, not fatal", s.startswith("Good morning."))
    empty = Briefings(BriefingConfig(), clock=lambda: at(0, 8))
    check("nothing to say: nothing", empty.compose("morning") == "" and empty.stats["empty"] == 1)
    fu.observe("the interview was cancelled")
    check("a cancelled event is not briefed", "interview" not in b.compose("morning"))
    one = tw.rig()
    one.observe("I told Omar about it")
    check("one question", Briefings(BriefingConfig(), wonder=one, clock=lambda: at(0, 8)).compose("morning")
          == "Good morning. I have a question for you when you have a minute.")


def b4_activity():
    print("\n=== B4. Activity keeps time per day ===")
    t = [at(0, 9)]
    am = ActivityMemory(clock=lambda: t[0])

    def window(subject, project="A.S.H", process="Code.exe"):
        am.observe(SensorEvent(source="active_window", modality=Modality.TEXT, text="",
                               meta={"subject": subject, "project": project, "process": process}), t[0])

    window("brain.py")
    t[0] += 900
    window("notes.md")
    k = "subject:brain py"
    check("a stretch adds to its day", abs(am.topics[k].by_day.get(d(0), 0) - 900) < 1, str(am.topics[k].by_day))
    window("brain.py")
    t[0] += 600
    window("notes.md")
    check("and a second one the same day adds up", abs(am.topics[k].by_day.get(d(0), 0) - 1500) < 1,
          str(am.topics[k].by_day))
    for i in range(20):
        t[0] = at(i + 1, 9)
        window("brain.py")
        t[0] += 400
        window("notes.md")
    check("two weeks are kept", len(am.topics[k].by_day) == 14 and d(20) in am.topics[k].by_day)


def rt_rig(**bc):
    clock = [at(0, 19)]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    dd = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()),
        gate=AttentionGate(max_offers_per_hour=20, quiet_hours=None), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             followups=True, followups_path=os.path.join(dd, "f.json"),
                             briefings=True, briefings_cfg=bc, briefings_path=os.path.join(dd, "b.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock = rt, ash, out, idle, clock

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True):
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    def free():
        for _ in range(15):
            step(10, 0.0, ticks=1)
        step(15, 25.0)

    h.step, h.say, h.free = step, say, free
    h.said = lambda: [o[0] for o in out]
    return h


def b5_runtime():
    print("\n=== B5. The runtime ===")
    h = rt_rig()
    h.say("I have an interview tomorrow at 10am")
    h.step(3 * 60, 300.0)
    check("no briefing without a free moment in the evening",
          not any(s.startswith("Here's your day") for s in h.said()), str(h.said()))
    h.free()
    briefs = [s for s in h.said() if s.startswith("Here's your day")]
    check("the evening briefing at a free moment", len(briefs) == 1 and "Tomorrow: your interview at 10:00." in briefs[0],
          str(h.said()))
    check("without the model", h.ash.calls == ["I have an interview tomorrow at 10am"])
    h.free()
    check("only once", len([s for s in h.said() if s.startswith("Here's your day")]) == 1)
    h.clock[0] = at(1, 8)
    h.step(1, 600.0)
    check("in the morning, not while away", not any(s.startswith("Good morning") for s in h.said()))
    h.step(1, 1.0)
    check("nor while typing", not any(s.startswith("Good morning") for s in h.said()))
    h.step(20, 20.0)
    morning = [s for s in h.said() if s.startswith("Good morning")]
    check("but once back and pausing", len(morning) == 1 and "Today: your interview at 10:00." in morning[0],
          str(h.said()))
    h.step(30, 30.0)
    check("only once", len([s for s in h.said() if s.startswith("Good morning")]) == 1)
    n = len(h.ash.calls)
    h.say("brief me")
    check("'brief me' any time, without the model", len(h.ash.calls) == n and h.said()[-1].startswith("Good morning"),
          str(h.said()[-1]))
    check("status carries it", h.rt.status()["briefings"]["given"])

    h = rt_rig()
    h.rt.gate.mute(True)
    h.say("I have an interview tomorrow at 10am")
    h.free()
    check("muted: no briefing", not any(s.startswith("Here's your day") for s in h.said()))

    h = rt_rig(enabled=False)
    h.say("I have an interview tomorrow at 10am")
    h.free()
    check("off: no briefing", not any(s.startswith("Here's your day") for s in h.said()))
    h.say("brief me")
    check("(and 'brief me' still is the user's to say)", h.said()[-1].startswith("Nothing much")
          or h.said()[-1].startswith("Here's your day"))

    h = rt_rig()
    h.clock[0] = at(3, 19)          # later than every rig above: the conversation state is shared
    h.free()
    check("nothing to say in the evening: nothing said, and not tried again",
          not any(s.startswith("Here's your day") for s in h.said())
          and h.rt.briefings.given.get("evening") == d(3), f"{h.said()} {h.rt.briefings.given}")
    check("and not counted as given", h.rt.briefings.stats["given"] == 0 and h.rt.briefings.stats["empty"] == 1)


if __name__ == "__main__":
    for fn in (b1_asking, b2_when, b3_what, b4_activity, b5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
