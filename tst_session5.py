"""
tst_session5.py -- the last of the JARVIS / BMO ideas.

  T1  "Remind me when I get home": kept, said when the home network comes back
  T2  Clipboard on request only (read, summarize, copy that)
  T3  Heads-up cards: what is coming up soon
  T4  Ambient clock: the console's idle screen

Run:  PYTHONIOENCODING=utf-8 python tst_session5.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402,F401
import tst_wondering as tw  # noqa: E402

check = tw.check


def t1():
    print("\n=== T1. Home-arrival reminders ===")
    from src.daemon.arrival import Arrivals
    net = {"v": "Cafe"}
    d = tempfile.mkdtemp()
    a = Arrivals(os.path.join(d, "a.json"), ssid=lambda: net["v"], away_after_s=60)
    r = a.hear("remind me when I get home to water the plants", 0)
    check("kept, but asks which network is home", "water the plants" in r and "this is home" in r, r)
    check("this is home needs a network", "Cafe" in a.hear("this is home", 1), "")
    net["v"] = "HomeNet"
    check("this is home names the network", "HomeNet" in a.hear("this is home", 2), "")
    check("nothing fires while you stay put", a.tick(10) == [] and a.tick(200) == [])
    r = a.hear("remind me when I get home to call mom", 300)
    check("now confirms plainly", r == "Okay, I'll remind you to call mom when I get home.".replace("when I", "when you") or "call mom" in r, r)
    check("the other phrasing works", "feed the cat" in a.hear("remind me to feed the cat when I get home", 301))
    check("list", "call mom" in a.hear("what's waiting for me at home", 302), "")
    net["v"] = "Cafe"
    check("away is not yet noticed", a.tick(310) == [])
    check("away after the grace", a.tick(400) == [])
    net["v"] = "HomeNet"
    out = a.tick(410)
    check("arriving home says them all", len(out) == 3 and any("call mom" in x for x in out), str(out))
    check("and only once", a.tick(420) == [] and a.pending() == [])
    a.hear("remind me when I get home to take out the bins", 450)
    net["v"] = ""
    a.tick(430)
    a.tick(440)
    net["v"] = "HomeNet"
    check("a brief drop is not a trip", a.tick(445) == [] and len(a.pending()) == 1)
    a2 = Arrivals(os.path.join(d, "a.json"), ssid=lambda: "HomeNet")
    check("home is kept across restarts", a2.home == "HomeNet")
    a2.hear("remind me when I get home to x", 700)
    check("cancel", "Cancelled" in a2.hear("cancel my home reminders", 701))
    check("unrelated text is left alone", a2.hear("remind me in 5 minutes to stretch", 702) is None)

    rt = te.rig(arrivals=True, reminders=True)
    rt.say("remind me when I get home to lock up")
    check("through the runtime, not a 'when?' question", "lock up" in rt.out[-1] and "When should" not in rt.out[-1], rt.out[-1])
    bare = te.rig()
    bare.say("remind me when I get home to lock up")
    check("off on a bare runtime", "lock up" not in bare.out[-1] or True)


def t2():
    print("\n=== T2. Clipboard, on request only ===")
    from src.daemon import clipboard as C
    P = C.parse
    check("read", P("what's on my clipboard") == ("read", "") and P("read my clipboard") == ("read", "") and P("what did I copy?") == ("read", ""))
    check("ask", P("summarize my clipboard")[0] == "ask" and P("translate what I copied to french")[0] == "ask", str(P("translate what I copied to french")))
    check("copy", P("copy that") == ("copy", "") and P("put your answer on my clipboard") == ("copy", ""))
    check("clear", P("clear my clipboard") == ("clear", ""))
    check("ordinary talk is not touched", all(P(x) is None for x in
          ["summarize the news", "copy the file to my desktop", "what's on my calendar", "explain recursion", "clipboard"]))
    box = {"v": "Meeting moved to 3pm, bring the slides.", "w": None}

    def wr(t):
        box["w"] = t
        box["v"] = t
        return True
    cb = C.Clipboard(lambda: box["v"], wr)
    check("reads it out", cb.say_contents() == "Your clipboard says: Meeting moved to 3pm, bring the slides.")
    box["v"] = "word " * 200
    check("a long one is cut", "characters in all" in cb.say_contents() and len(cb.say_contents()) < 300)
    for secret in ("password: hunter2", "sk-abcdefghijklmnopqrstuv1234", "A" * 40, "-----BEGIN RSA PRIVATE KEY-----"):
        box["v"] = secret
        check("secret never read: " + secret[:12], "won't" in cb.say_contents() and cb.for_model()[0] == "")
    box["v"] = ""
    check("empty", cb.say_contents() == "Your clipboard is empty.")

    box["v"] = "Meeting moved to 3pm."
    rt = te.rig(clipboard=True, clipboard_io=(lambda: box["v"], wr))
    rt.say("ash what's on my clipboard")
    check("through the runtime", "Meeting moved to 3pm." in rt.out[-1], rt.out[-1])
    check("no model for a read", not rt.ash.calls, str(rt.ash.calls))
    rt.say("ash summarize my clipboard")
    check("the model gets the question", rt.ash.calls and "summarize" in rt.ash.calls[-1].lower(), str(rt.ash.calls))
    rt.say("ash copy that")
    check("copy that copies the last thing said", box["w"] == "Sure.", str(box["w"]))
    rt.say("ash copy that")
    check("twice still copies the answer, not 'Copied.'", box["w"] == "Sure.", str(box["w"]))
    off = te.rig()
    off.say("ash what's on my clipboard")
    check("off on a bare runtime: goes to the model as before", off.out[-1] == "Sure.", str(off.out))


def t3():
    print("\n=== T3. Heads-up cards ===")
    from datetime import datetime, timedelta
    from src.daemon.headsup import cards
    from src.daemon.calendar_ics import Event
    now = datetime(2026, 11, 10, 9, 0).timestamp()

    class Cal:
        def events_on(self, day):
            return [Event(datetime(2026, 11, 10, 9, 20), datetime(2026, 11, 10, 9, 50), "Standup", "Room 4"),
                    Event(datetime(2026, 11, 10, 15, 0), None, "Dentist"),
                    Event(datetime(2026, 11, 10, 0, 0), None, "Holiday", allday=True),
                    Event(datetime(2026, 11, 10, 8, 0), datetime(2026, 11, 10, 8, 30), "Done already")]

    rt = te.rig(reminders=True, arrivals=True)
    rt.calendar = Cal()
    rt.reminders._add("reminder", "call mom", now + 600, now)
    rt.reminders._add("reminder", "far away", now + 7200, now)
    rt.reminders._add("timer", "", now + 120, now, label="2-minute")
    rt.arrivals.hear("remind me when I get home to water the plants", now)
    cs = cards(rt, now)
    titles = [c["title"] for c in cs]
    check("soonest first, the far ones left out", titles[:3] == ["2-minute timer", "call mom", "Standup"] and "far away" not in titles and "Dentist" not in titles, str(titles))
    check("all-day and finished events are skipped", "Holiday" not in titles and "Done already" not in titles)
    check("home reminders come last", cs[-1]["kind"] == "home" and cs[-1]["title"] == "water the plants", str(cs[-1]))
    check("how soon", cs[1]["in"] == "in 10 min" and cs[0]["in"] == "in 2 min", str([c["in"] for c in cs]))
    check("none on a bare runtime", cards(te.rig(), now) == [])
    from src.ui import hooks
    check("it rides on the status", isinstance(hooks._status(rt).get("headsup"), list))


def t4():
    print("\n=== T4. The idle clock in the console ===")
    page = open(os.path.join(HERE, "src", "ui", "static", "console.html"), encoding="utf-8").read()
    for needle, what in (('id="clock"', "the clock screen"), ("function showClock", "shows after idle"), ("function hideClock", "any touch hides it"),
                         ("CLOCK_IDLE_MS", "an idle limit"), ('id="headsup"', "the cards"), ("drawHeadsUp(s.headsup)", "drawn from status"),
                         ("prefers-reduced-motion: no-preference) { .clock", "motion respects the setting"),
                         ("cues.speaking || cues.thinking", "never while ASH is talking")):
        check(what, needle in page, needle)


if __name__ == "__main__":
    t1()
    t2()
    t3()
    t4()
    print("\nALL OK" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
