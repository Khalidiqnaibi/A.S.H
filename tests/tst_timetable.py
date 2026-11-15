"""
tst_timetable.py -- a weekly timetable you say aloud; "when is my class?" answered from it, never invented.

  W1  Saying it: names, days, times, with am/pm left out; what is not a timetable line
  W2  Asking: next class, a named class, today/tomorrow/a weekday, how long until, the whole timetable
  W3  Changing it: replace, remove, clear; saved and loaded; a bad file
  W4  The clock: ten minutes before (once), in_class
  W5  The runtime: no model, a guest, the room, quiet while in class

Run:  PYTHONIOENCODING=utf-8 python tests/tst_timetable.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.timetable import NO_TIMETABLE, Timetable, _span  # noqa: E402

check = tw.check
SUN = time.mktime((2026, 3, 8, 8, 0, 0, 0, 0, -1))        # Sunday 08:00


def at(day_offset, h, m=0):
    return time.mktime((2026, 3, 8 + day_offset, h, m, 0, 0, 0, -1))


def rig(path=None):
    t = [SUN]
    tt = Timetable(path=path, clock=lambda: t[0])
    tt.t = t
    return tt


def w1_saying():
    print("\n=== W1. Saying it ===")
    tt = rig()
    s = tt.hear("my NoSQL class is on Sunday and Tuesday from 10 to 11:30")
    check("the example, as said", s == "Got it -- NoSQL: Tuesday and Sunday, 10:00 to 11:30.", s)
    check("kept as days 1 and 6", tt.classes[0]["days"] == [1, 6] and tt.classes[0]["start"] == 600 and tt.classes[0]["end"] == 690)
    for text, want in [("I have databases every monday at 2pm for 2 hours", "Databases: Monday, 14:00 to 16:00"),
                       ("add a class called Algorithms on weekdays 9 to 10", "Algorithms: weekdays, 09:00 to 10:00"),
                       ("my Networks lecture is on Wednesdays at 3", "Networks: Wednesday, 15:00 to 16:30"),
                       ("remember that my Maths class is Thursday 12 to 1", "Maths: Thursday, 12:00 to 13:00"),
                       ("my AI lab runs mon wed fri from 8:30 to 10", "AI: Monday, Wednesday and Friday, 08:30 to 10:00")]:
        s = rig().hear(text)
        check(f"{text!r}", s is not None and want in s, str(s))
    check("1 to 3 is the afternoon, 9 to 11 the morning", _span("from 1 to 3") == (13 * 60, 15 * 60) and _span("9 to 11") == (540, 660))
    check("with am/pm said, it is as said", _span("9am to 11am") == (540, 660) and _span("8pm to 10pm") == (1200, 1320))
    for text in ("my class is boring", "my NoSQL class is hard", "I have a headache on Monday", "my class is on Sunday",
                 "my NoSQL class is on Sunday", "what is a class", "I have an exam on monday at 9"):
        check(f"not a timetable line: {text!r}", rig().hear(text) is None, str(rig().hear(text)))


def w2_asking():
    print("\n=== W2. Asking ===")
    tt = rig()
    check("with nothing in it: says how to tell it, never invents a time", tt.hear("when is my next class") == NO_TIMETABLE)
    tt.hear("my NoSQL class is on Sunday and Tuesday from 10 to 11:30")
    tt.hear("I have databases every monday at 2pm for 2 hours")
    s = tt.hear("when is my next class?")
    check("Sunday 08:00: the next is NoSQL today at 10:00, in 2 hours", s == "Your next class is NoSQL today at 10:00 (in 2 hours).", s)
    check("when is my NoSQL class", tt.hear("when is my NoSQL class") == "Your next NoSQL class is today at 10:00 (in 2 hours).",
          tt.hear("when is my NoSQL class"))
    tt.t[0] = at(0, 12)
    check("after it: the next is Monday's", tt.hear("when is my next class") == "Your next class is Databases tomorrow at 14:00.",
          tt.hear("when is my next class"))
    check("a class not on it", tt.hear("when is my Chemistry class") == "I don't have a Chemistry class on your timetable.",
          tt.hear("when is my Chemistry class"))
    s = tt.hear("how long until my class")
    check("how long until", s.startswith("Databases starts in 26 hours") and "tomorrow" in s, s)
    tt.t[0] = SUN
    check("do I have class today", tt.hear("do I have class today") == "Today: NoSQL at 10:00.", tt.hear("do I have class today"))
    check("tomorrow", tt.hear("do I have a class tomorrow?") == "Tomorrow: Databases at 14:00.", tt.hear("do I have a class tomorrow?"))
    check("a weekday", tt.hear("do I have class on tuesday") == "Tuesday: NoSQL at 10:00.", tt.hear("do I have class on tuesday"))
    check("an empty day", tt.hear("do I have class on wednesday") == "No classes Wednesday.", tt.hear("do I have class on wednesday"))
    s = tt.hear("what's my timetable")
    check("the whole timetable", s.startswith("Your timetable: NoSQL: Tuesday and Sunday, 10:00 to 11:30; Databases: Monday"), s)
    tt.t[0] = at(0, 10, 30)
    s = tt.hear("when is my next class")
    check("in a class: says so", s.startswith("You're in NoSQL now, until 11:30."), s)


def w3_changing():
    print("\n=== W3. Changing it ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "tt.json")
    tt = rig(p)
    tt.hear("my NoSQL class is on Sunday from 10 to 11:30")
    s = tt.hear("my NoSQL class is on Monday from 9 to 10")
    check("the same name again replaces it", len(tt.classes) == 1 and "replaced" in s and tt.classes[0]["days"] == [0], s)
    check("saved to disk", json.load(open(p, encoding="utf-8"))["classes"][0]["name"] == "NoSQL")
    check("and loaded again", Timetable(path=p).classes[0]["days"] == [0])
    tt.hear("my AI class is on Friday at 11")
    check("remove one", tt.hear("remove my AI class") == "Okay, I took AI off your timetable." and len(tt.classes) == 1)
    check("remove one that is not there says nothing (left to others)", tt.hear("remove my Zebra class") is None)
    check("clear it", tt.hear("clear my timetable").startswith("Cleared your timetable (1 class") and not tt.classes)
    check("clear an empty one", tt.hear("clear my timetable") == "Your timetable is already empty.")
    with open(p, "w") as f:
        f.write("{not json")
    check("a broken file starts empty, not crashed", Timetable(path=p).classes == [])
    with open(p, "w") as f:
        json.dump({"classes": [{"name": "X", "days": [9], "start": 10, "end": 5}, {"name": "Ok", "days": [1], "start": 600, "end": 660}]}, f)
    check("bad entries in a file are dropped, good ones kept", [c["name"] for c in Timetable(path=p).classes] == ["Ok"])


def w4_clock():
    print("\n=== W4. The clock ===")
    tt = rig()
    tt.hear("my NoSQL class is on Sunday from 10 to 11:30")
    tt.t[0] = at(0, 9, 40)
    check("twenty minutes before: nothing", tt.tick() == [])
    tt.t[0] = at(0, 9, 52)
    s = tt.tick()
    check("eight minutes before: said", s == ["NoSQL starts in 8 minutes, at 10:00."], str(s))
    check("...once", tt.tick() == [])
    tt.t[0] = at(7, 9, 55)
    check("and again the next week", len(tt.tick()) == 1)
    tt.t[0] = at(0, 10, 0)
    check("in_class at the start", tt.in_class() is not None and tt.in_class()["name"] == "NoSQL")
    tt.t[0] = at(0, 11, 30)
    check("not at the end", tt.in_class() is None)
    tt.t[0] = at(1, 10, 30)
    check("not on another day", tt.in_class() is None)


def w5_runtime():
    print("\n=== W5. The runtime ===")
    import datetime as _dt
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    rt = te.rig(timetable=True, timetable_path=os.path.join(tempfile.mkdtemp(), "tt.json"))
    check("switched on by config, and in the status", rt.timetable is not None and rt.status()["timetable"] is not None)
    n = len(rt.ash.calls)
    rt.say("my NoSQL class is on Sunday and Tuesday from 10 to 11:30", source="console")
    check("said to it: kept, no model", rt.out[-1].startswith("Got it -- NoSQL") and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("when is my next class", source="console")
    check("asked: answered from it, no model", rt.out[-1].startswith("Your next class is NoSQL") and len(rt.ash.calls) == n, rt.out[-1])
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="my Chemistry class is on Monday at 9", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest cannot change it", len(rt.timetable.classes) == 1 and "only do that for" in rt.out[-1], rt.out[-1])
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="my Biology class is on Friday at 9", addressed=False,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("speech from the room cannot either", len(rt.timetable.classes) == 1)
    cur = rt.timetable.classes[0]
    d = _dt.datetime.fromtimestamp(rt.clock())
    cur["days"] = [d.weekday()]
    cur["start"] = max(0, d.hour * 60 + d.minute - 5)
    cur["end"] = min(24 * 60, d.hour * 60 + d.minute + 30)
    n = len(rt.out)
    rt.response_sink("You keep switching windows.", {"addressed": False, "source": "flow", "pathway": "offer"})
    check("in class, nothing is volunteered", len(rt.out) == n and rt.stats.get("held_busy", 0) >= 1, str(rt.out[n:]))
    rt.response_sink("Reminder: stand up.", {"addressed": False, "source": "reminder", "pathway": "reminder"})
    check("...but a reminder still is", rt.out[-1] == "Reminder: stand up.", str(rt.out[-1]))


if __name__ == "__main__":
    w1_saying()
    w2_asking()
    w3_changing()
    w4_clock()
    w5_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
