"""
tst_cheer.py -- remembering the good moments, and celebrating.

  M1  Reading a birthday: "March 3", "3rd of March", "03/03", ISO; nonsense
  M2  Good moments: follow-ups that went well, newest first; brought back to
      someone worried, the same kind first; "tell me something good"; none
      on record, the brain answers
  M3  Celebrations: the birthday once a year (briefing or free moment, not
      both), streaks at 5, 7, 10..., a long stretch from pieces joined up,
      never twice, kept across a restart; not to someone having a hard time
  M4  The runtime: a worried line answered with a good moment, no model; a
      celebration said at a free moment

Run:  PYTHONIOENCODING=utf-8 python tst_cheer.py
"""

import os
import sys
import tempfile
import threading
import time
from datetime import date, timedelta
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.cheer import Celebrate, Cheer, parse_day  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402

check = tw.check
NOW = time.mktime((2026, 3, 3, 10, 0, 0, 0, 0, -1))      # a Tuesday, March 3
DAY = 86400.0
FREE = dict(idle_s=60.0, free_moment=True)


def m1_dates():
    print("\n=== M1. Reading a birthday ===")
    for text, want in (("March 3", (3, 3)), ("march 3rd", (3, 3)), ("3rd of March", (3, 3)), ("3 Mar", (3, 3)),
                       ("03/03", (3, 3)), ("12/04", (4, 12)), ("1990-03-03", (3, 3)), ("on the 21st of december", (12, 21)),
                       ("Sept 9", (9, 9))):
        check(f"{text!r}", parse_day(text) == want, str(parse_day(text)))
    for text in ("", None, "sometime in spring", "13/13", "March 40", "99/99"):
        check(f"not a day: {text!r}", parse_day(text) is None)


def fu(*rows):
    items = {str(i): NS(what=w, noun=n, outcome=o, status=s, closed_at=t) for i, (w, n, o, s, t) in enumerate(rows)}
    return NS(_lock=threading.RLock(), items=items)


def m2_good():
    print("\n=== M2. Good moments ===")
    f = fu(("your interview", "interview", "good", "answered", NOW - 60 * DAY),
           ("your exam", "exam", "good", "told", NOW - 3 * DAY),
           ("your trip to Paris", "trip", "bad", "answered", NOW - 2 * DAY),
           ("the dentist", "dentist", "good", "pending", 0),
           ("your old exam", "exam", "good", "answered", NOW - 400 * DAY))
    c = Cheer(followups=f, clock=lambda: NOW)
    rows = c.good_moments(NOW)
    check("went well, closed, within a year, newest first", [r["what"] for r in rows] == ["your exam", "your interview"])
    check("worried about the same kind of thing", c.hear("I'm so nervous about the interview tomorrow", NOW)
          == "You've done this before -- remember your interview in January? That went well.",
          str(c.hear("I'm so nervous about the interview tomorrow", NOW)))
    check("otherwise the latest", c.hear("I'm really anxious about my presentation", NOW)
          == "You've done this before -- remember your exam on Saturday? That went well.")
    check("worried, about nothing said", c.hear("feeling stressed", NOW).startswith("You've done this before"))
    check("tell me something good", c.hear("tell me something good", NOW)
          == "Some good ones: your exam on Saturday; your interview in January -- all went well.",
          str(c.hear("tell me something good", NOW)))
    for t in ("what went well lately?", "cheer me up", "what's gone well"):
        check(f"{t!r}", (c.hear(t, NOW) or "").startswith("Some good ones"))
    for t in ("the exam went well", "I am not worried", "what is an interview", "", None):
        check(f"left alone: {str(t)[:30]!r}", c.hear(t, NOW) is None or t == "I am not worried")
    check("'I am not worried' is not a worry", c.hear("I am not worried", NOW) is None)
    e = Cheer(followups=fu(), clock=lambda: NOW)
    check("none on record: the brain answers", e.hear("I'm nervous about the exam", NOW) is None
          and e.hear("tell me something good", NOW) is None)
    check("no follow-ups at all: none", Cheer().good_moments(NOW) == [])


def act(**projects):
    return NS(_lock=threading.RLock(), topics={k: NS(name=k, kind="project", by_day=v) for k, v in projects.items()})


def days_back(n, secs=3600):
    return {(date.fromtimestamp(NOW) - timedelta(days=i)).isoformat(): secs for i in range(n)}


def m3_celebrate():
    print("\n=== M3. Celebrations ===")
    p = os.path.join(tempfile.mkdtemp(), "cheer.json")
    kb = NS(fact=lambda k: "March 3rd" if k == "birthday" else None)
    c = Cheer(path=p, knowledge=kb, clock=lambda: NOW)
    check("the birthday, in the morning briefing", c.briefing_part("morning", NOW) == "Happy birthday!")
    c.tick(NOW)
    check("and not again at a free moment", c.next_line() is None)
    check("nor in a second briefing", c.briefing_part("morning", NOW + 60) is None)
    c2 = Cheer(path=p, knowledge=kb, clock=lambda: NOW)
    c2.tick(NOW)
    check("kept across a restart", c2.next_line() is None)
    c3 = Cheer(knowledge=kb, clock=lambda: NOW)
    c3.tick(NOW)
    check("no briefing: at a free moment instead", c3.next_line()[1] == "Happy birthday! I hope it's a good one.")
    check("not on another day", not Cheer(knowledge=kb).birthday_today(NOW + DAY))
    check("no birthday known: nothing", not Cheer(knowledge=NS(fact=lambda k: None)).birthday_today(NOW))

    c = Cheer(activity=act(ASH=days_back(7), thesis=days_back(4), site=days_back(6, secs=300)), clock=lambda: NOW)
    c.tick(NOW)
    c.tick(NOW + 30)
    lines = [l for _, l in c.pending]
    check("a 7-day streak", "That's 7 days in a row on ASH. Nice streak." in lines, str(lines))
    check("4 is not yet one; a few minutes a day is not a day; queued once", len(lines) == 1, str(lines))
    key = c.pending[0][0]
    c.said(key, NOW)
    c.tick(NOW + 60)
    check("said once", c.next_line() is None)
    gap = days_back(3)
    gap.update({(date.fromtimestamp(NOW) - timedelta(days=i)).isoformat(): 3600 for i in range(4, 9)})
    c = Cheer(activity=act(ASH=gap), clock=lambda: NOW)
    c.tick(NOW)
    check("a gap breaks the streak", c.next_line() is None)

    c = Cheer(clock=lambda: NOW)
    t0 = NOW - 3 * 3600
    for i in range(5):                                   # 30-minute pieces, back to back
        c.stretch("brain.py", t0 + i * 1800 + (10 if i else 0), t0 + (i + 1) * 1800)
    check("still going: nothing yet", c.next_line() is None)
    c.tick(t0 + 5 * 1800 + 60)
    check("a minute after the last piece: maybe a short pause, still nothing", c.next_line() is None)
    c.tick(t0 + 5 * 1800 + 400)
    check("finished: two and a half hours", c.next_line()[1] == "That was a solid 2.5 hours on brain.py. Nice work.",
          str(c.next_line()))
    c = Cheer(clock=lambda: NOW)
    c.stretch("brain.py", t0, t0 + 3600)
    c.stretch("brain.py", t0 + 3600 + 600, t0 + 3 * 3600)
    c.stretch("notes.md", t0 + 3 * 3600, t0 + 4 * 3600)
    check("a ten-minute break splits it: neither half is long", c.next_line() is None)
    c = Cheer(clock=lambda: NOW)
    c.stretch("a", t0, t0 + 3 * 3600)
    c.stretch("b", t0 + 3 * 3600, t0 + 3 * 3600 + 60)
    check("another thing started: the long one is done", c.next_line()[1].endswith("on a. Nice work."))

    c = Cheer(knowledge=kb, clock=lambda: NOW)
    c.tick(NOW)
    a = Celebrate(c)
    cand = a.propose(Environment(**FREE), NOW)
    check("offered at a free moment", cand is not None and cand.text.startswith("Happy birthday"))
    check("not to someone having a hard time", Celebrate(c).propose(Environment(mood=-0.6, **FREE), NOW) is None)
    a.spoken(NOW)
    check("said: done", c.next_line() is None and c.stats["celebrated"] == 1)


def m4_runtime():
    print("\n=== M4. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [NOW]
    out = []
    d = tempfile.mkdtemp()
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=99),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0, cheer=True,
                             activity=True, activity_path=os.path.join(d, "a.json"),
                             cheer_path=os.path.join(d, "cheer.json"), followups=True,
                             followups_path=os.path.join(d, "f.json"), moments=True,
                             moments_cfg={"min_gap_s": 0, "max_per_hour": 99,
                                          "disabled": ["system_assessment", "joke", "mood_check"]}),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    from src.daemon.followups import FollowUp
    rt.followups.items["x"] = FollowUp(id="x", key="interview", what="your interview", noun="interview", start=NOW - 9 * DAY,
                                       clock=False, due=NOW - 9 * DAY, created=NOW - 10 * DAY, status="answered",
                                       closed_at=NOW - 8 * DAY, outcome="good")

    def say(text):
        clock[0] += 3
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("I'm nervous about the interview on friday")
    check("a good moment brought back, no model", out[-1] == "You've done this before -- remember your interview "
                                                             "last week? That went well." and ash.calls == [], str(out[-1:]))
    rt.cheer.stretch("ASH", clock[0] - 3 * 3600, clock[0] - 600)
    rt.moments.set_mood(0.5, clock[0])

    def step(dt, idle_s, ticks=1):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()
    for _ in range(15):
        step(10, 0.0)
    step(15, 25.0, ticks=2)
    step(1, 25.0)
    check("the long stretch celebrated at a free moment", any("solid" in o and "ASH" in o for o in out), str(out[-3:]))
    check("status", rt.status()["cheer"]["celebrated"] == 1)
    check("activity memory tells it of each stretch", rt.activity.on_stretch == rt.cheer.stretch)
    from src.daemon.activity import ActivityMemory
    am = ActivityMemory(clock=lambda: NOW)
    got = []
    am.on_stretch = lambda name, a, b: got.append((name, a, b))
    am.observe(NS(source="active_window", text="", meta={"process": "code.exe", "subject": "brain.py",
                                                          "project": "ASH"}), NOW)
    am.observe(NS(source="active_window", text="", meta={"process": "code.exe", "subject": "notes.md",
                                                          "project": "Other"}), NOW + 900)
    check("a stretch recorded is passed on, by project", got == [("ASH", NOW, NOW + 900)], str(got))


if __name__ == "__main__":
    for fn in (m1_dates, m2_good, m3_celebrate, m4_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
