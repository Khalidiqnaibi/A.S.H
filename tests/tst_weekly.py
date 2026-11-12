"""
tst_weekly.py -- a weekly review.

  W1  Time on projects: the last seven days only, projects over files when
      there are projects, small bits left out, most first
  W2  What got done: follow-ups closed, questions answered, reminders that
      went off -- this week only
  W3  Who came up: counted per day, names only, two weeks kept, across a
      restart; once is not "most"
  W4  Said: the sentence, a quiet week, the phrasings; the Sunday evening
      briefing only
  W5  The runtime: answered without the model; a name said twice counts

Run:  PYTHONIOENCODING=utf-8 python tests/tst_weekly.py
"""

import os
import sys
import tempfile
import time
from datetime import date, timedelta
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import threading  # noqa: E402

import tst_wondering as tw  # noqa: E402
from src.daemon.weekly import Weekly  # noqa: E402

check = tw.check
# A Sunday evening.
NOW = time.mktime((2026, 3, 15, 19, 0, 0, 0, 0, -1))
DAY = 86400.0


def d(n):
    return (date.fromtimestamp(NOW) - timedelta(days=n)).isoformat()


class Act:
    def __init__(self, topics):
        self._lock = threading.RLock()
        self.topics = {t.name: t for t in topics}


def topic(name, kind, by_day):
    return NS(name=name, kind=kind, by_day=by_day)


def w1_projects():
    print("\n=== W1. Time on projects ===")
    a = Act([topic("A.S.H", "project", {d(0): 4 * 3600, d(3): 5 * 3600, d(9): 20 * 3600}),
             topic("thesis", "project", {d(1): 2 * 3600}),
             topic("taxes", "project", {d(2): 300}),
             topic("brain.py", "subject", {d(0): 9 * 3600})])
    w = Weekly(activity=a, clock=lambda: NOW)
    check("the last seven days, projects, most first", w.projects(NOW) == [("A.S.H", 9 * 3600), ("thesis", 2 * 3600)],
          str(w.projects(NOW)))
    a2 = Act([topic("brain.py", "subject", {d(0): 3600}), topic("notes.md", "subject", {d(1): 1800})])
    check("no projects: files", Weekly(activity=a2).projects(NOW) == [("brain.py", 3600), ("notes.md", 1800)])
    check("no activity memory: none", Weekly().projects(NOW) == [])


def w2_done():
    print("\n=== W2. What got done ===")
    lock = threading.RLock()
    fu = NS(_lock=lock, items={"a": NS(status="answered", closed_at=NOW - DAY), "b": NS(status="told", closed_at=NOW - 2 * DAY),
                              "c": NS(status="answered", closed_at=NOW - 9 * DAY), "d": NS(status="pending", closed_at=0)})
    wo = NS(_lock=lock, wonders={"x": NS(status="answered", closed_at=NOW - 3 * DAY),
                                 "y": NS(status="dismissed", closed_at=NOW - DAY)})
    re_ = NS(_lock=lock, items={"r": NS(status="fired", fired_at=NOW - DAY), "s": NS(status="set", fired_at=0),
                                "t": NS(status="fired", fired_at=NOW - 8 * DAY)})
    w = Weekly(followups=fu, wonder=wo, reminders=re_)
    check("this week only", w.counts(NOW) == {"followups": 2, "answered": 1, "reminders": 1}, str(w.counts(NOW)))
    bad = NS(_lock=lock, items=None)
    check("a broken part counts nothing, the rest still count",
          Weekly(followups=bad, wonder=wo).counts(NOW) == {"followups": 0, "answered": 1, "reminders": 0})


def w3_people():
    print("\n=== W3. Who came up ===")
    p = os.path.join(tempfile.mkdtemp(), "weekly.json")
    w = Weekly(path=p)
    for _ in range(3):
        w.named("Omar", NOW - DAY)
    w.named("Sara", NOW)
    w.named("Sara", NOW - 2 * DAY)
    w.named("Karim", NOW)
    w.named("Lina", NOW - 10 * DAY)
    w.named("Lina", NOW - 10 * DAY)
    w.named("Lina", NOW - 10 * DAY)
    check("most first; once is not most; last week not counted", w.top_people(NOW) == ["Omar", "Sara"],
          str(w.top_people(NOW)))
    one = Weekly()
    one.named("Karim", NOW)
    check("named once is not 'came up most'", one.top_people(NOW) == [])
    w.save()
    check("kept across a restart", Weekly(path=p).top_people(NOW) == ["Omar", "Sara"])
    w.named("Zed", NOW + 15 * DAY)
    check("two weeks kept", len(w.people) == 1, str(list(w.people)))
    w.named("  ", NOW + 15 * DAY)
    check("no name, nothing", sum(sum(m.values()) for m in w.people.values()) == 1)


def w4_said():
    print("\n=== W4. Said ===")
    a = Act([topic("A.S.H", "project", {d(0): 9 * 3600}), topic("thesis", "project", {d(1): 2 * 3600}),
             topic("site", "project", {d(1): 2700})])
    lock = threading.RLock()
    fu = NS(_lock=lock, items={"a": NS(status="answered", closed_at=NOW - DAY), "b": NS(status="answered", closed_at=NOW),
                              "c": NS(status="told", closed_at=NOW)})
    wo = NS(_lock=lock, wonders={str(i): NS(status="answered", closed_at=NOW) for i in range(4)})
    re_ = NS(_lock=lock, items={"r": NS(status="fired", fired_at=NOW)})
    w = Weekly(activity=a, followups=fu, wonder=wo, reminders=re_, clock=lambda: NOW)
    for n in ("Omar", "Omar", "Omar", "Sara", "Sara"):
        w.named(n, NOW)
    s = w.compose(NOW)
    check("the sentence", s == "This week: 9 hours on A.S.H, 2 hours on thesis and an hour on site. 3 follow-ups closed, "
                               "4 of my questions answered, 1 reminder. Omar came up most, then Sara.", s)
    check("a quiet week", Weekly().compose(NOW) == "A quiet week -- I don't have much on it yet.")
    check("minutes", Weekly(activity=Act([topic("x", "project", {d(0): 1200})])).compose(NOW)
          == "This week: 20 minutes on x.")
    for t in ("how was my week?", "weekly review", "my week in review", "what did I do this week",
              "what did I work on this week?", "recap my week", "sum up the week", "ash, how has my week been"):
        check(f"{t!r}", w.hear(t, NOW) == s, str(w.hear(t, NOW)))
    for t in ("what's the weather this week", "how was your week", "I had a long week", "", None):
        check(f"left alone: {str(t)[:30]!r}", w.hear(t, NOW) is None)
    check("in the Sunday evening briefing", w.briefing_part("evening", NOW) == s)
    check("not the morning", w.briefing_part("morning", NOW) is None)
    check("not on Monday", w.briefing_part("evening", NOW + DAY) is None)
    check("another day if set", Weekly(activity=a, day="monday", clock=lambda: NOW).briefing_part("evening", NOW + DAY))
    check("off when no day", Weekly(activity=a, day="").briefing_part("evening", NOW) is None)
    check("a quiet week is not briefed", Weekly().briefing_part("evening", NOW) is None)


def w5_runtime():
    print("\n=== W5. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [NOW]
    out = []
    dd = tempfile.mkdtemp()
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, weekly=True,
                             weekly_path=os.path.join(dd, "weekly.json"), wonder=True,
                             wonder_cfg={"min_gap_s": 10 ** 9}, wonder_path=os.path.join(dd, "w.json"),
                             knowledge_path=os.path.join(dd, "k.json")),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]

    def say(text):
        clock[0] += 3
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("I had lunch with Omar today")
    say("tomorrow I see Omar again at the office")
    say("how was my week?")
    check("answered without the model", "Omar came up most." in out[-1] and
          ash.calls == ["I had lunch with Omar today", "tomorrow I see Omar again at the office"], str(out[-1:]))
    check("status", rt.status()["weekly"]["day"] == "sunday")
    import json
    check("saved on a tick", json.load(open(os.path.join(dd, "weekly.json")))["people"][d(0)]["Omar"] == 2)


if __name__ == "__main__":
    for fn in (w1_projects, w2_done, w3_people, w4_said, w5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
