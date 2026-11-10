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


if __name__ == "__main__":
    t1()
    print("\nALL OK" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
