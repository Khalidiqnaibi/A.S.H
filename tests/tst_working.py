"""
tst_working.py -- "what am I working on?" from what ASH keeps about the user's windows.

It went to the model, which answered "Khalid, who you mentioned is Immortal" -- it never read the activity record (stretches on a file, a page or
a project) or how long the window in front has been there. Now it does, by rule.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_working.py
"""

import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import working  # noqa: E402

check = tw.check
NOW = time.mktime((2026, 11, 23, 15, 0, 0, 0, 0, -1))
TODAY = time.strftime("%Y-%m-%d", time.localtime(NOW))
YEST = time.strftime("%Y-%m-%d", time.localtime(NOW - 86400))
LAST_WEEK = time.strftime("%Y-%m-%d", time.localtime(NOW - 10 * 86400))


def topic(name, kind, **by_day):
    days = {{"today": TODAY, "yest": YEST, "old": LAST_WEEK}[k]: v for k, v in by_day.items()}
    return NS(name=name, kind=kind, by_day=days)


def act(topics, cur=None):
    return NS(_lock=threading.RLock(), topics={t.name: t for t in topics}, _cur=cur, observe=lambda *a, **k: None, tick=lambda *a, **k: None,
              save=lambda *a, **k: None, status=lambda: {})


def w1_parse():
    print("\n=== W1. What is asked ===")
    for t, k in [("what am I working on", "now"), ("what am I working on right now?", "now"), ("what project am I on", "now"),
                 ("what have I been working on today", "today"), ("what was I doing", "today"), ("what did I work on earlier", "today"),
                 ("what have I been working on this week", "week"), ("what did I do this week", "week")]:
        check(f"{t!r} -> {k}", working.parse(t) == k, str(working.parse(t)))
    for t in ("what's new", "what am I wearing", "what time is it", "what are you working on", "what is a transformer", "who am I"):
        check(f"{t!r} is not this", working.parse(t) is None)


def w2_answers():
    print("\n=== W2. The answers ===")
    cur = NS(subject="main.py", project="microcode", start=NOW - 25 * 60)
    a = act([topic("microcode", "project", today=4200, yest=3000), topic("A.S.H", "project", today=2400, old=9999), topic("main.py", "subject", today=1800),
             topic("Chart.png", "subject", today=60)], cur=cur)
    rt = NS(activity=a, dwell=None)
    s = working.answer(rt, "now", NOW)
    check("now: the stretch in progress, with how long, and today's top", s.startswith("Right now: main.py in microcode, for 25 minutes.") and "Today so far: microcode (1 hour 10 minutes), A.S.H (40 minutes)." in s, s)
    check("projects are named, not their files", "main.py (" not in s.split("Today so far")[1])
    s = working.answer(rt, "today", NOW)
    check("today: the top things, then what is on now", s.startswith("Today: microcode (1 hour 10 minutes), A.S.H (40 minutes).") and s.endswith("And right now, main.py in microcode."), s)
    s = working.answer(rt, "week", NOW)
    check("this week: includes yesterday, not last week's", "microcode (2 hours)" in s and "A.S.H (40 minutes)" in s and "9999" not in s, s)
    only_files = NS(activity=act([topic("main.py", "subject", today=1800), topic("notes.md", "subject", today=900)]), dwell=None)
    s = working.answer(only_files, "today", NOW)
    check("with no projects, the files", "main.py (30 minutes)" in s and "notes.md (15 minutes)" in s, s)
    nothing = NS(activity=act([topic("brief", "subject", today=30)]), dwell=None)
    check("a glance is not work: nothing to say", working.answer(nothing, "today", NOW, fallback=False) == "")
    s = working.answer(nothing, "now", NOW)
    check("and it says what it needs, not a guess", "a few minutes" in s and "remember that I'm working on" in s, s)
    check("no activity memory at all is the same", "a few minutes" in working.answer(NS(activity=None, dwell=None), "now", NOW))


def w1b_adverbs():
    print("\n=== W1b. The same question with its adverbs ===")
    for t in ("What project am I actively doing?", "what am I currently working on", "what am I working on at the moment", "what projects am I really working on these days",
              "what thing am I busy with right now", "what am I actually working on"):
        check(f"{t!r} is a 'now' question", working.parse(t) == "now", str(working.parse(t)))
    for t in ("what am I wearing", "what am I doing wrong", "what project is due", "what am I going to do tomorrow"):
        check(f"{t!r} is not", working.parse(t) is None, str(working.parse(t)))


def w2b_time_on():
    print("\n=== W2b. How much did I work on X ===")
    for t, want in [("how much did I work on A.S.H this week", ("A.S.H", 7, "this week")), ("how long was I on youtube today", ("youtube", 1, "today")),
                    ("how much time did I spend on NoSQL", ("NoSQL", 1, "today")), ("how long did I work on the thesis yesterday", ("the thesis", -1, "yesterday")),
                    ("how much have I worked on microcode lately?", ("microcode", 7, "over the last week"))]:
        check(f"{t!r} -> {want}", working.parse_time(t) == want, str(working.parse_time(t)))
    for t in ("how long have I been on this", "how long was I on it", "how much did I work", "how long was I away", "how much did I spend on lunch and dinner and a very long list of other things",
              "how much is that", "what am I working on"):
        check(f"{t!r} is not this", working.parse_time(t) is None, str(working.parse_time(t)))
    a = act([topic("A.S.H", "project", today=5400, yest=3600, old=9000), topic("Devil's Advocate (Part 3) - YouTube", "subject", today=1200), topic("microcode", "project", today=40)])
    rt = NS(activity=a)
    check("this week: today's and yesterday's, not last week's", working.answer_time(rt, "A.S.H", 7, "this week", NOW) == "About 2 hours 30 minutes on A.S.H this week.",
          working.answer_time(rt, "A.S.H", 7, "this week", NOW))
    check("yesterday alone", working.answer_time(rt, "A.S.H", -1, "yesterday", NOW) == "About 1 hour on A.S.H yesterday.")
    check("a page by part of its title", working.answer_time(rt, "youtube", 1, "today", NOW) == "About 20 minutes on youtube today.")
    check("under a minute is not claimed", working.answer_time(rt, "microcode", 1, "today", NOW).startswith("I haven't tracked anything on microcode today"))
    check("never heard of: said so", working.answer_time(rt, "gardening", 7, "this week", NOW).startswith("I haven't tracked anything on gardening"))
    check("no activity memory at all: said so, no crash", working.answer_time(NS(activity=None), "x", 1, "today", NOW).startswith("I haven't tracked"))


def w3_runtime():
    print("\n=== W3. In the runtime ===")
    import tst_modellast as tm
    rt = tm.rig()
    rt.activity = act([topic("microcode", "project", today=4200)], cur=NS(subject="main.py", project="microcode", start=rt.clock() - 1500))
    used, said = rt.say("what am I working on")
    check("answered by rule, not the model", not used and said and said[-1].startswith("Right now: main.py in microcode"), f"{used} {said}")
    rt2 = tm.rig()
    used, said = rt2.say("what have I been doing today")
    check("with nothing in the activity record, the per-window answer (dwell) still takes it", not used and said, f"{used} {said}")
    rt3 = tm.rig()
    day = time.strftime("%Y-%m-%d", time.localtime(rt3.clock()))
    rt3.activity = NS(_lock=threading.RLock(), topics={"A.S.H": NS(name="A.S.H", kind="project", by_day={day: 7200})}, _cur=None,
                      observe=lambda *a, **k: None, tick=lambda *a, **k: None, save=lambda *a, **k: None, status=lambda: {})
    used, said = rt3.say("how much did I work on A.S.H this week")
    check("'how much did I work on X' is answered by rule, from the per-day record", not used and said and said[-1] == "About 2 hours on A.S.H this week.", f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    w1_parse()
    w1b_adverbs()
    w2_answers()
    w2b_time_on()
    w3_runtime()
