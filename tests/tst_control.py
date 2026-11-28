"""
tst_control.py -- the keyboard, the mouse and the screen by voice, and the limits on it.

25 Nov: "just click space", "go full screen", "click the channel" got talk. Run:  PYTHONIOENCODING=utf-8 python tests/tst_control.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import control  # noqa: E402

check = tw.check


class Ex:
    def __init__(self):
        self.log = []
        self.found = (640, 360)

    def keys(self, combo):
        self.log.append(("keys", tuple(combo)))
        return True

    def type_text(self, t):
        self.log.append(("type", t))
        return True

    def click(self, kind="click", at=None):
        self.log.append(("click", kind, at))
        return True

    def move(self, arg):
        self.log.append(("move", arg))
        return True

    def scroll(self, n):
        self.log.append(("scroll", n))
        return True

    def find_text(self, phrase):
        self.log.append(("find", phrase))
        return self.found

    def devices(self):
        return [("Monitor", "Generic PnP Monitor"), ("AudioEndpoint", "Speakers (Realtek)"), ("Bluetooth", "Galaxy Buds")]


def mk(**kw):
    t = [1000.0]
    ex = Ex()
    c = control.Control(control.ControlConfig(**kw), ex, clock=lambda: t[0])
    c.t = t
    return c, ex


def c1_parse():
    print("\n=== C1. What is asked ===")
    want = {"click space": ("keys", ("space",)), "press enter": ("keys", ("enter",)), "hit control S": ("keys", ("ctrl", "s")),
            "press control shift t": ("keys", ("ctrl", "shift", "t")), "press the escape key": ("keys", ("esc",)), "press up arrow twice": ("keys", ("up",)),
            "go full screen": ("keys", ("f11",)), "exit full screen": ("keys", ("esc",)), "copy": ("keys", ("ctrl", "c")), "paste that": ("keys", ("ctrl", "v")),
            "new tab": ("keys", ("ctrl", "t")), "close this tab": ("keys", ("ctrl", "w")), "switch to tab 3": ("keys", ("ctrl", "3")),
            "go back a page": ("keys", ("alt", "left")), "show the desktop": ("keys", ("win", "d")), "take a screenshot": ("keys", ("win", "shift", "s")),
            "go to the bottom": ("keys", ("ctrl", "end")), "scroll down": ("scroll", ()), "double click": ("mouse", ()), "right click": ("mouse", ()),
            "move the mouse left a lot": ("move", ()), "type hello there": ("type", ()), "click subscribe": ("find_click", ()),
            "what's connected to this computer": ("devices", ()), "type a": ("type", ())}
    for t, (kind, keys) in want.items():
        a = control.parse(t)
        check(f"{t!r} -> {kind} {keys}", a is not None and a.kind == kind and (not keys or a.keys == keys), str(a))
    check("a video page takes 'f' for full screen", control.parse("full screen", "Lecture - YouTube - Opera").keys == ("f",))
    check("elsewhere F11", control.parse("full screen", "Report - Word").keys == ("f11",))
    check("'press up arrow twice' presses twice", control.parse("press up arrow twice").repeat == 2)
    for t in ["open chrome", "play", "pause", "what time is it", "who is Linda", "remember to buy milk", "tell me about space", "click on my notes"]:
        a = control.parse(t)
        check(f"not for control: {t!r}", a is None, str(a))


def c2_do():
    print("\n=== C2. It is done, and said ===")
    c, ex = mk()
    check("space", c.hear("click space") == "Pressed space." and ex.log[-1] == ("keys", ("space",)))
    check("press twice", c.hear("press down arrow twice") and ex.log[-2:] == [("keys", ("down",)), ("keys", ("down",))])
    check("scroll down is negative", c.hear("scroll down") == "Scrolling down." and ex.log[-1][0] == "scroll" and ex.log[-1][1] < 0)
    check("scroll up is positive", c.hear("scroll up a lot") and ex.log[-1][1] > 0)
    check("click on a word: found, then clicked there", c.hear("click subscribe") == "Clicked subscribe." and ex.log[-1] == ("click", "click", (640, 360)))
    ex.found = None
    check("not on the screen: said, nothing clicked", c.hear("click unsubscribe") == "I can't see “unsubscribe” on the screen." and ex.log[-1][0] == "find")
    check("typing", c.hear("type hello there") == "Typed it." and ex.log[-1] == ("type", "hello there"))
    said = c.hear("what devices are connected")
    check("devices listed", said.startswith("Connected: screens: Generic PnP Monitor; audio:"), said)


def c3_limits():
    print("\n=== C3. The limits ===")
    c, ex = mk()
    n = len(ex.log)
    say = c.hear("press delete")
    check("Delete asks first and does nothing yet", say.startswith("Press delete?") and len(ex.log) == n and c.waiting, say)
    check("a yes does it", c.hear("yes") == "Pressed delete." and ex.log[-1] == ("keys", ("delete",)) and not c.waiting)
    c.hear("close this window")
    check("a no cancels", c.hear("no") == "Okay, not doing it." and ex.log[-1] == ("keys", ("delete",)))
    c.hear("close this window")
    c.t[0] += 120
    check("an old question is not answered by a late yes", c.hear("yes") is None and ex.log[-1] == ("keys", ("delete",)))
    check("lock / run / ctrl alt delete are never pressed", all(c.hear(x).startswith("I don't press that") for x in ("press control alt delete", "press windows L", "press windows R")))
    check("...and nothing was sent for them", ex.log[-1] == ("keys", ("delete",)))
    check("blocklisted window: refused", "privacy blocklist" in c.hear("press space", blocked=True) and ex.log[-1] == ("keys", ("delete",)))
    check("privacy paused: refused", "paused for privacy" in c.hear("press space", privacy_paused=True))
    check("long typing asks first", c.hear("type " + "x" * 80).startswith("Type 80 characters"))
    c.hear("no")
    r, rex = mk(max_per_minute=3)
    for _ in range(3):
        r.hear("press space")
    check("a rate limit: the fourth in a minute is refused", "a lot of key presses" in r.hear("press space") and len(rex.log) == 3)
    off, oex = mk(enabled=False)
    check("switched off: nothing is a control line", off.hear("press space") is None and not oex.log)


def c4_runtime():
    print("\n=== C4. On the runtime ===")
    import tst_effects as te
    h = te.rig(media=True)
    rt = getattr(h, "rt", h)
    check("it is on with the media keys", rt.control is not None)
    ex = Ex()
    rt.control.ex = ex
    n = len(rt.ash.calls)
    h.say("press space", source="console")
    check("said to ASH: pressed, no model", h.out[-1] == "Pressed space." and ex.log[-1] == ("keys", ("space",)) and len(rt.ash.calls) == n, h.out[-1])
    h.say("press delete", source="console")
    h.say("yes", source="console")
    check("Delete asked, then done on a yes", ex.log[-1] == ("keys", ("delete",)), str(ex.log[-2:]))
    check("it shows in the status", rt.status()["control"]["done"] >= 2)


def c5_compound():
    print("\n=== C5. Two commands in one sentence run as a pipe ===")
    import tst_desktop as td
    from src.daemon import pipeline as pl
    check("'switch to Opera and make it full screen' is a pipe of two actions", [st.name for st in pl.parse("Switch to Opera and make it full screen.")["parallel"][0]] == ["act", "act"])
    check("a single command is not a pipe", pl.parse("switch to opera") is None and pl.parse("make it full screen") is None)
    check("'open Discord and type in it' is not two commands (the 'in it' is not text to type)", pl.parse("open Discord and type in it") is None)
    h = td.rt_rig()
    rt = h.rt
    h.ex.procs.add("opera.exe")
    rt.control = control.Control(control.ControlConfig(), Ex(), clock=lambda: rt.clock())
    ex = rt.control.ex
    rt._pipeline_ctx_pause = None
    h.say("Switch to Opera and make it full screen")
    check("both were done, in order, and said in one line", h.ex.focused[-1:] == ["opera.exe"] and ex.log[-1:] == [("keys", ("f11",))] and "Switching to Opera" in h.said()[-1] and "Full screen" in h.said()[-1], f"{h.ex.focused} {ex.log} {h.said()[-2:]}")
    h.say("press delete and scroll down")
    check("a command that asks first stops the pipe with its question (nothing after it runs)", "Press delete?" in h.said()[-1] and not any(x[0] == "scroll" for x in ex.log[-2:]), f"{h.said()[-1]} {ex.log[-2:]}")


if __name__ == "__main__":
    c1_parse()
    c2_do()
    c3_limits()
    c4_runtime()
    c5_compound()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
