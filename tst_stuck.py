"""
tst_stuck.py -- help when stuck, without the model.

  H1  What is searched for: the error's own words, without paths, line
      numbers and addresses that change every run
  H2  Finding it: in the lookup folders only, text files only, skipped
      folders skipped, newest first, at most three
  H3  Said back: where it was found, or that it wasn't, and the look-up
  H4  The runtime: the same error five times, a pause, the offer; "yes"
      searches the files and starts a look-up, no model; "not now" does
      neither; with no folders and no research the model answers as before

Run:  PYTHONIOENCODING=utf-8 python tst_stuck.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import stuck  # noqa: E402

check = tw.check
T0 = tw.T0


def h1_needle():
    print("\n=== H1. What is searched for ===")
    check("a short error as it is", stuck.needle("KeyError: 'start work'") == "KeyError: 'start work'")
    check("the last line of a traceback", stuck.needle("Traceback (most recent call last):\n  File \"x.py\", "
                                                       "line 3, in <module>\nZeroDivisionError: division by zero")
          == "ZeroDivisionError: division by zero")
    check("paths dropped", stuck.needle(r"FileNotFoundError: No such file: C:\Users\me\data\a.csv")
          == "FileNotFoundError: No such file", str(stuck.needle(r"FileNotFoundError: No such file: C:\Users\me\a.csv")))
    check("addresses dropped", "0x" not in stuck.needle("Segfault at 0x7ffe3a2b in worker thread"))
    check("line numbers dropped", "line" not in stuck.needle("SyntaxError at line 42: invalid syntax"))
    check("cut to 60", len(stuck.needle("ValueError: " + "x" * 200)) <= 60)
    for e in ("", None, "Error", "  :  "):
        check(f"nothing worth searching: {e!r}", stuck.needle(e) is None)


def tree():
    root = tempfile.mkdtemp()
    def w(rel, text, age=0):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(text)
        t = time.time() - age
        os.utime(p, (t, t))
        return p
    w("notes/fixes.md", "# fixes\nKeyError: 'start work' -- the tasks dict was keyed by the raw name\n", age=100)
    w("old/log.txt", "boot\nkeyerror: 'START WORK' seen again\n", age=10)
    w("proj/a.py", "x = 1\n", age=5)
    w("node_modules/pkg/index.js", "KeyError: 'start work'\n")
    w(".hidden/x.md", "KeyError: 'start work'\n")
    w("pic.png", "KeyError: 'start work'\n")
    for i in range(4):
        w(f"many/f{i}.txt", "KeyError: 'start work'\n", age=1000 + i)
    return root


def h2_find():
    print("\n=== H2. Finding it ===")
    root = tree()
    hits = stuck.find_in_files("KeyError: 'start work'", [root])
    names = [os.path.basename(p) for p, _, _ in hits]
    check("newest first, at most three", names == ["log.txt", "fixes.md", "f0.txt"], str(names))
    check("the line, and its number", hits[1][1] == 2 and "keyed by the raw name" in hits[1][2])
    check("case does not matter", hits[0][1] == 2)
    check("skipped folders and hidden ones not searched", all("node_modules" not in p and ".hidden" not in p
                                                             for p, _, _ in stuck.find_in_files(
                                                                 "KeyError: 'start work'", [root], limit=20)))
    check("only text files", all(not p.endswith(".png") for p, _, _ in
                                 stuck.find_in_files("KeyError: 'start work'", [root], limit=20)))
    check("a missing folder is fine", stuck.find_in_files("KeyError: 'start work'", [root + "_nope"]) == [])
    check("nothing worth searching: nothing searched", stuck.find_in_files("Error", [root]) == [])
    t = [0.0]
    def slow():
        t[0] += 1.0
        return t[0]
    check("a time budget", len(stuck.find_in_files("KeyError: 'start work'", [root], budget_s=2.5, limit=20,
                                                 clock=slow)) <= 2)
    check("a file budget", len(stuck.find_in_files("KeyError: 'start work'", [root], max_files=2, limit=20)) <= 2)


def h3_describe():
    print("\n=== H3. Said back ===")
    hits = [("/x/notes/fixes.md", 2, "KeyError ..."), ("/y/log.txt", 9, "...")]
    s = stuck.describe("KeyError: 'id'", hits, True)
    check("where, and the look-up", s == "I found \u201cKeyError: 'id'\u201d in your files before: fixes.md line 2; "
                                        "log.txt line 9. I'm also looking it up -- I'll tell you what I find.", s)
    s = stuck.describe("KeyError: 'id'", [], False)
    check("not found, no look-up", s == "I didn't find \u201cKeyError: 'id'\u201d anywhere in your files.", s)
    home = os.path.expanduser("~")
    check("paths shown short", stuck.tidy_hits([(os.path.join(home, "a.md"), 3, "x")])[0].startswith("~"))


def h4_runtime():
    print("\n=== H4. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    root = tree()

    def build(roots=True, research=True):
        clock = [T0]
        out = []
        ash = tw.Ash()
        ash.contexts = []
        _think = ash.think
        ash.think = lambda q, image=None, context="": (ash.contexts.append(context), _think(q, image, context))[1]
        per = PeripheryRegistry()
        idle = tw.FakeIdle()
        per.sensors["idle"] = idle
        d = tempfile.mkdtemp()
        rt = AmbientRuntime(
            ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=99),
            privacy=PrivacyPolicy(),
            config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, research=research,
                                 research_path=os.path.join(d, "rs.json"), wonder=True, lookup=True,
                                 wonder_cfg={"min_gap_s": 10 ** 9,
                                             "lookup": {"roots": [root] if roots else [], "web": False}},
                                 wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
            response_sink=lambda t, m: out.append(t))
        rt.clock = lambda: clock[0]
        if rt.research is not None:
            rt.research.background = False
            rt.research.lookup = None
            rt.research._run = lambda j: None            # the look-up itself is research.py's business
        return rt, clock, out, ash, idle

    def stuck_offer(rt, clock, idle):
        for _ in range(5):                          # five times: well past the "stuck" bar
            clock[0] += 20
            idle.value = 0.0
            rt.submit(SensorEvent(source="system", modality=Modality.TEXT,
                                  text="Traceback (most recent call last): KeyError: 'start work'", meta={"error": True}, salience=0.6))
            rt._tick()
        for _ in range(4):
            clock[0] += 15
            idle.value = 40.0
            rt._tick()

    rt, clock, out, ash, idle = build()
    stuck_offer(rt, clock, idle)
    offered = [o for o in out if "keeps coming back" in o]
    check("(the offer was made)", offered, str(out))

    def say(text):
        clock[0] += 3
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    if offered:
        say("yes")
        check("files searched and found", "I found \u201cKeyError: 'start work'\u201d" in out[-1]
              and "log.txt line 2" in out[-1], str(out[-1:]))
        check("a look-up started", any(j.topic == "KeyError: 'start work'" for j in rt.research.jobs.values())
              and "looking it up" in out[-1])
        check("no model asked", ash.calls == [], str(ash.calls))
        check("counted", rt.stats.get("stuck_helped") == 1)

    rt, clock, out, ash, idle = build()
    stuck_offer(rt, clock, idle)
    if any("keeps coming back" in o for o in out):
        say("not now")
        check("'not now': neither", not rt.research.jobs and "I found" not in out[-1], str(out[-1:]))

    rt, clock, out, ash, idle = build()
    stuck_offer(rt, clock, idle)
    if any("keeps coming back" in o for o in out):
        say("what does a KeyError on a dict usually mean when the key comes from user input")
        check("talking about it instead of a yes: the model answers", ash.calls and not rt.research.jobs,
              str(out[-1:]))

    rt, clock, out, ash, idle = build(roots=False, research=False)
    stuck_offer(rt, clock, idle)
    if any("keeps coming back" in o for o in out):
        say("yes")
        check("no folders and no research: the model answers as before", ash.calls != [], str(out[-1:]))
        check("...and is told what the offer was about", "You offered help because" in (ash.contexts[-1] if
              ash.contexts else ""), str(ash.contexts[-1:])[:200])


if __name__ == "__main__":
    for fn in (h1_needle, h2_find, h3_describe, h4_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
