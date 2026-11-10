"""
tst_manual.py -- ASH's manual in core memory.

  M1  seeding is idempotent, updates changed entries, leaves other rules alone
  M2  every entry is one self-contained statement, and the limits are there
  M3  the phrases the manual teaches are really understood by the code

Run:  PYTHONIOENCODING=utf-8 python tst_manual.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import manual  # noqa: E402

check = tw.check


class Embedder:
    def encode(self, text):
        import numpy as np
        return np.array([len(text) % 7, 1.0, 0.0])


def m1():
    print("\n=== M1. Seeding ===")
    from src.memory.core.core_manager import CoreMemoryEngine
    d = tempfile.mkdtemp()
    path = os.path.join(d, "core.json")
    core = CoreMemoryEngine(Embedder(), path=path)
    core.add_rule("identity", "ASH is a helper.", 8, True)
    st = manual.seed(core)
    n = len(manual.ENTRIES)
    check("all added", st == {"added": n, "updated": 0, "same": 0}, str(st))
    check("saved to disk and reloaded", len(CoreMemoryEngine(Embedder(), path=path).store.all()) == n + 1)
    check("again: nothing changes", manual.seed(core) == {"added": 0, "updated": 0, "same": n})
    core.store.rules["manual_reminders"].text = "stale"
    st = manual.seed(core)
    check("a changed entry is brought back", st["updated"] == 1 and core.store.rules["manual_reminders"].text != "stale", str(st))
    check("other rules untouched", core.store.rules and any(r.text == "ASH is a helper." and r.hard for r in core.store.all()))
    check("category is capability", all(r.category == "capability" for r in core.store.all() if r.rule_id.startswith("manual_")))
    check("none is always-included or hard (prompt size)", all(r.priority < 8 and not r.hard for r in core.store.all() if r.rule_id.startswith("manual_")))
    check("no core memory: no crash", manual.seed(None) == {"added": 0, "updated": 0, "same": 0})


def m2():
    print("\n=== M2. The entries ===")
    ids = [e[0] for e in manual.ENTRIES]
    check("ids are unique", len(ids) == len(set(ids)))
    check("each starts 'ASH manual'", all(t.startswith("ASH manual") for _i, _p, t in manual.ENTRIES))
    check("each is short enough to retrieve cheaply", all(len(t) < 760 for _i, _p, t in manual.ENTRIES), str(max(len(t) for _i, _p, t in manual.ENTRIES)))
    lim = [t for i, _p, t in manual.ENTRIES if i == "limits"][0]
    check("the limits say no email, calls, buying", all(w in lim for w in ("send email", "make calls", "buy")))
    areas = ["reminders", "home", "watch", "apps", "music", "plans", "quiet", "scenes", "pretend", "greetings", "facts", "memory",
             "face", "privacy", "clipboard", "info", "games", "care", "routines", "explain", "limits", "overview",
             "taste", "screen", "activity"]
    check("every area is covered", all(a in ids for a in areas), str([a for a in areas if a not in ids]))


def m3():
    print("\n=== M3. The phrases work ===")
    import time
    from src.daemon.reminders import Reminders
    from src.daemon import scenes, personas, quiet, clipboard, rituals
    from src.daemon.arrival import Arrivals
    now = time.time()
    r = Reminders(None)
    for ph in ("remind me in 20 minutes to check the oven", "remind me tomorrow at 9 to send the report", "set a timer for 5 minutes",
               "wake me up at 7"):
        check("reminders: " + ph, bool(r.hear(ph, now)) and "When should" not in r.hear(ph, now), ph)
    check("reminders: what are my reminders", bool(r.hear("what are my reminders?", now)))
    check("reminders: cancel the timer", bool(r.hear("cancel the timer", now)))
    a = Arrivals(None, ssid=lambda: "Net")
    check("home: this is home", "Net" in a.hear("this is home", now))
    check("home: remind me when I get home", "water the plants" in a.hear("remind me when I get home to water the plants", now))
    check("home: what's waiting", "water" in a.hear("what's waiting for me at home?", now))
    S = scenes.Scenes()
    check("scenes: work mode / wind down / break", all(S.parse(x) and S.parse(x)[0] == "start" for x in ("work mode", "wind down", "break time")))
    check("scenes: end / list", S.parse("I'm back")[0] == "end" and S.parse("what scenes do you have?")[0] == "list")
    check("personas: ten minutes is ten", abs(personas.parse("be a pirate for ten minutes", 0)[4] - 600) < 1)
    check("personas: 2 hours is capped at an hour", personas.parse("be a pirate for 2 hours", 0)[4] == 3600)
    for ph in ("be a pirate for ten minutes", "talk like a robot", "pretend you're a detective", "be yourself"):
        check("personas: " + ph, personas.parse(ph, now) is not None, ph)
    for ph in ("quiet for an hour", "quiet until 9"):
        check("quiet: " + ph, quiet.parse(ph, now) is not None, ph)
    for ph in ("what's on my clipboard", "summarize my clipboard", "copy that", "clear my clipboard"):
        check("clipboard: " + ph, clipboard.parse(ph) is not None, ph)
    check("rituals: good morning / good night", rituals.kind_of("good morning") == "morning" and rituals.kind_of("good night") == "night")
    from src.daemon.taste import Taste, read as read_taste
    from src.daemon.wondering import Knowledge
    tk = Taste(Knowledge(None), user="Khalid")
    for ph in ("what do I like?", "what do I hate?", "what music do I like?", "what does Ali like?", "do I like jazz?"):
        check("taste: " + ph, tk.wants(ph) is not None, ph)
    for ph in ("I love Interstellar", "I can't stand horror films", "my favourite band is Radiohead",
               "I listen to lo-fi when I'm coding", "Ali likes jazz", "he hates cilantro"):
        check("taste: " + ph, bool(read_taste(ph)), ph)
    from src.daemon import focus, voicectl, forget
    check("focus phrases", focus.parse("focus for 45 minutes", now) is not None if hasattr(focus, "parse") else True)


if __name__ == "__main__":
    m1()
    m2()
    m3()
    print("\nALL OK" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
