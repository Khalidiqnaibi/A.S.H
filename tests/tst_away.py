"""
tst_away.py -- "what did I miss?" and the welcome back, from what really happened.

"whats new while i was away" went to the model, which said "away about an hour and a half" (made up). The reminders that fired, the questions
saved up, a shutdown the crash watch found and the window left open were all kept -- in different places. away.py joins them, by rule, and says
"nothing came up" when nothing did.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_away.py
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
from src.daemon import away  # noqa: E402

check = tw.check
NOW = time.mktime((2026, 11, 22, 18, 0, 0, 0, 0, -1))
SINCE = NOW - 4080                                           # away 1 h 8 min


def fake(**kw):
    rt = NS(reminders=None, crashwatch=None, wonder=None, dwell=None)
    for k, v in kw.items():
        setattr(rt, k, v)
    return rt


def reminders(*rows):
    return NS(_lock=threading.RLock(), items={str(i): NS(status="fired", fired_at=SINCE + 600 + i, **r) for i, r in enumerate(rows)})


def a1_asking():
    print("\n=== A1. What counts as asking ===")
    for t in ("what's new", "what did I miss", "whats new while i was away", "catch me up", "what happened while I was away", "Ash, anything new since I left?",
              "give me a quick update", "what have I missed", "what did i miss while i was gone"):
        check(f"{t!r} asks for the digest", away.asks_digest(t))
    for t in ("what's the weather", "what time is it", "what did you say", "what's new in python 3.13", "tell me the news", "what happened to the file I made"):
        check(f"{t!r} does not", not away.asks_digest(t))


def a2_digest():
    print("\n=== A2. What happened ===")
    nothing = away.digest(fake(), SINCE, NOW)
    check("nothing happened: said so, with how long (not invented)", nothing.startswith("Nothing came up while you were away (1 hour 8 minutes)"), nothing)
    check("...and the welcome says nothing extra", away.digest(fake(), SINCE, NOW, welcome=True) == "")
    rt = fake(reminders=reminders(dict(kind="reminder", text="call mom", label="", due=SINCE + 500), dict(kind="timer", text="", label="5-minute", due=SINCE + 300)),
              crashwatch=NS(unreported=lambda now: [{"t": SINCE + 900, "code": 0}]), wonder=NS(pending=lambda: [{}, {}]),
              dwell=NS(_lock=threading.RLock(), recent=[(SINCE - 900, SINCE - 10, "Code.exe", "main.py")]))
    s = away.digest(rt, SINCE, NOW)
    check("a reminder, a timer, a shutdown and saved questions are all in it", all(x in s for x in ("a reminder went off", "call mom", "5-minute timer finished", "shut down unexpectedly", "2 questions")), s)
    check("and what was left open", "Before that, you left main.py (VS Code) open." in s, s)
    check("it starts as an answer to the question", s.startswith("While you were away: "))
    w = away.digest(rt, SINCE, NOW, welcome=True)
    check("on 'Welcome back' the same facts, without the heading or the window", w.startswith("A reminder went off") and "While you were away" not in w and "left" not in w, w)
    old = fake(reminders=reminders(dict(kind="reminder", text="earlier thing", label="", due=SINCE - 5000)))
    old.reminders.items["0"].fired_at = SINCE - 4000
    check("a reminder that fired BEFORE they left is not news", "earlier thing" not in away.digest(old, SINCE, NOW))
    one = fake(wonder=NS(pending=lambda: [{}]))
    check("one question: singular", "I saved 1 question to ask" in away.digest(one, SINCE, NOW))


def a3_runtime():
    print("\n=== A3. In the runtime ===")
    import tst_modellast as tm
    rt = tm.rig()
    used, said = rt.say("what did I miss")
    check("asked, it is answered by rule, never the model", not used and said and said[-1].startswith("Nothing came up while you were away"), f"{used} {said}")
    used, said = rt.say("whats new while i was away")
    check("(the exact words from 22 Nov)", not used and said, f"{used} {said}")
    used, said = rt.say("what's new in python")
    check("but a question about something else still goes where it did", used or (said and "Nothing came up" not in said[-1]), str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    a1_asking()
    a2_digest()
    a3_runtime()
