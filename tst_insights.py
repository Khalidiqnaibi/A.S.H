"""
tst_insights.py -- the rest of the browsing profile: context, suggestions, what is learned, what is shown.

  T1  A browser tab says what the user is doing ("while coding" works in a browser)
  T2  The social learner learns per activity: a remark can land at leisure and not while coding
  T3  "How are your nudges landing?" -- what it has learned, said back, and resettable
  T4  Suggestions and habits: what changed this week, when and how they browse -- numbers, not advice
  T5  "What I'm learning" in the console: one place to see everything it holds
  T6  No LLM in any of it

Run:  PYTHONIOENCODING=utf-8 python tst_insights.py
"""

import ast
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_browsing as tb  # noqa: E402
import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
HERE = os.path.dirname(os.path.abspath(__file__))
hm, midnight = tb.hm, tb.midnight


def t1_occasion():
    print("\n=== T1. A tab says what the user is doing ===")
    from src.daemon.occasion import Occasion, page_activity
    t = [1000.0]
    o = Occasion(clock=lambda: t[0])
    check("categories that are an activity", [page_activity(c) for c in ("code", "learning", "work", "docs", "reading", "games")]
          == ["coding", "studying", "working", "working", "reading", "gaming"])
    check("...and those that are not", all(page_activity(c) == "" for c in ("video", "social", "news", "shopping", "music", "unknown", "")))
    o.window("chrome.exe", 1000)
    check("a browser window alone says nothing", o.now() == "")
    o.page("code", 1005)
    check("a code page in a browser is coding", o.now() == "coding" and o.context() == "coding")
    t[0] = 1100
    o.window("chrome.exe", 1100)                    # the title changed: the same browser, still the same page
    check("a window event for the same browser does not lose it", o.now() == "coding")
    o.page("video", 1200)
    check("a leisure page ends it and is 'leisure' to the learner", o.now() == "" and o.context() == "leisure")
    check("...and the stretch was recorded as coding", o.during(1005, 1200, 1200) == "coding")
    o.page("learning", 1300)
    o.window("Code.exe", 1310)
    o.page("", 1320, stopped=True)                  # the extension's late "stopped" must not end the editor
    check("a late 'stopped' from the browser does not end what an editor started", o.now() == "coding", o.now())
    o.window("chrome.exe", 1400, blocked=True)
    check("a blocked window is nothing", o.now() == "")
    o.page("video", 1500)
    o.window("Spotify.exe", 1510)
    check("leaving the browser ends 'leisure'", o.context() == "", o.context())

    # the taste link end to end: a listen during a coding tab is "while coding"
    from src.daemon.taste import Taste, TasteConfig
    from tst_taste import kb_with
    t2 = [5000.0]
    o2 = Occasion(clock=lambda: t2[0])
    o2.window("chrome.exe", 5000)
    o2.page("code", 5000)
    t2[0] = 5400
    tk = Taste(kb_with(), user="Khalid", cfg=TasteConfig(), clock=lambda: t2[0])
    occ = o2.during(5000, 5400, 5400)
    check("a song heard through a coding tab is linked to coding", occ == "coding", occ)
    for i in range(3):
        tk.watch({"finished": True, "finished_kind": "music", "finished_s": 200, "finished_title": "Lofi", "finished_artist": "Lofi Girl",
                  "finished_how": "changed"}, 5400 + i, occasion=occ)
    check("...three times makes it 'while coding'", any("while coding" in tk.sentence("Khalid", True, e)
                                                            for e in tk.kb.tastes["khalid"]["items"].values()),
          str(tk.kb.tastes))


def t2_learner():
    print("\n=== T2. The learner learns per activity ===")
    from src.daemon.social import SocialLearner, context_key
    check("the context gains the activity", context_key(0.6, 1_800_000_000.0, "coding").endswith("/coding"))
    check("...only a known one", context_key(0.6, 1_800_000_000.0, "banana").count("/") == 1
          and context_key(0.6, 1_800_000_000.0).count("/") == 1)
    now = 1_800_000_000.0
    lr = SocialLearner(clock=lambda: now)
    coding, fun = "idle/evening/coding", "idle/evening/leisure"
    for _ in range(6):
        lr.update("joke", fun, 1.0, 1.0, now)
    for _ in range(5):
        lr.update("joke", coding, 0.0, 2.0, now)
    check("a joke lands at leisure and not while coding", lr.p("joke", fun, now) > 0.6 > 0.4 > lr.p("joke", coding, now),
          f"{lr.p('joke', fun, now):.2f} {lr.p('joke', coding, now):.2f}")
    check("...so the gate's bar moves the right way", lr.bar_shift("joke", coding, 0.0, now) > 0 > lr.bar_shift("joke", fun, 0.0, now))
    check("an activity it has little on leans on the kind overall",
          abs(lr.p("joke", "idle/evening/writing", now) - lr.p_kind("joke", now)) < 0.15)
    lr2 = SocialLearner(clock=lambda: now)
    lr2.update("joke", "idle/evening", 1.0, 1.0, now)
    check("no activity: the cells are what they always were", set(lr2.cells) == {"*", "joke", "joke|idle/evening"}, str(set(lr2.cells)))
    import tst_social as ts
    g = ts.gate() if hasattr(ts, "gate") else None
    if g is not None:
        d = g.decide(ts.offer_ev(strength=0.7), ts.Drives(), user_busy=0.6, now=now, activity="coding")
        check("the gate puts it in the decision's context", str(d.ctx).endswith("/coding"), str(d.ctx))
        d = g.decide(ts.offer_ev(strength=0.7), ts.Drives(), user_busy=0.6, now=now)
        check("...and without one, nothing changes", str(d.ctx).count("/") == 1, str(d.ctx))


def t3_report():
    print("\n=== T3. What it has learned, said back ===")
    from src.daemon import uxreport
    from src.daemon.social import SocialLearner
    q = {"how are your nudges landing?": "report", "what have you learned about when to talk to me?": "report",
         "when should you stay quiet?": "report", "forget how I react": "reset", "reset what you learned about my reactions": "reset"}
    check("the ways people ask", all(uxreport.parse(k) == v for k, v in q.items()), str({k: uxreport.parse(k) for k in q}))
    check("ordinary talk is not taken for it", not any(uxreport.parse(x) for x in ("how are you", "be quiet", "what is a nudge", "forget Omar")))
    now = 1_800_000_000.0
    lr = SocialLearner(clock=lambda: now)
    check("with little data it says so", "only seen 0 reactions" in uxreport.describe(lr, now), uxreport.describe(lr, now))
    for _ in range(8):
        lr.update("joke", "idle/evening/leisure", 1.0, 1.0, now)
    for _ in range(6):
        lr.update("care", "typing/morning/coding", 0.0, 2.0, now)
    r = uxreport.describe(lr, now)
    check("overall rate, with the number of reactions", "lands well, from 14 reactions" in r, r)
    check("best and weakest kind", "Best: jokes" in r and "Weakest: health nudges" in r, r)
    check("by what you were doing", "most welcome while you're relaxing online" in r and "least while coding" in r, r)
    check("it says how to reset", "forget how I react" in r)
    lr.snooze("joke", now)
    check("what it was told to hold off on is said", "holding off on jokes" in uxreport.describe(lr, now), uxreport.describe(lr, now))
    out = uxreport.reset(lr)
    check("reset starts from neutral", "14 reactions" in out and lr.updates == 0 and not lr.cells and not lr.snoozes, out)
    check("no learner: said, not a crash", "switched off" in uxreport.describe(None))

    rt = te.rig()
    rt.brain.user = "Khalid"
    from src.senses.base import Modality, SensorEvent
    for _ in range(7):
        rt.learner.update("joke", "idle/evening/leisure", 1.0, 1.0, rt.t[0])
    rt.say("how are your nudges landing?")
    check("on the runtime, by rule", "lands well" in rt.out[-1], rt.out[-1])
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="how are your nudges landing?", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice is not told", "lands well" not in rt.out[-1], rt.out[-1])
    rt.say("forget how I react")
    check("reset by voice", rt.learner.updates == 0 and "starting from neutral" in rt.out[-1], rt.out[-1])


def two_weeks():
    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor
    now = hm(0, 15)
    led = Ledger(clock=lambda: now)
    for k in range(8, 14):                                       # last week: ordinary
        led.record(hm(k, 10), hm(k, 11), "github.com", "code", "table")
        led.record(hm(k, 11), hm(k, 11, 30), "docs.google.com", "docs", "table")
        led.record(hm(k, 20), hm(k, 20, 30), "youtube.com", "video", "table")
        led.record(hm(k, 13), hm(k, 13, 10), "bbc.com", "news", "table")
        led.churn(10, hm(k, 12))
    for k in range(1, 7):                                        # this week: much more video, more tab hopping
        led.record(hm(k, 10), hm(k, 11), "github.com", "code", "table")
        led.record(hm(k, 11), hm(k, 11, 30), "docs.google.com", "docs", "table")
        led.record(hm(k, 19), hm(k, 21, 30), "youtube.com", "video", "table")
        led.record(hm(k, 13), hm(k, 13, 10), "bbc.com", "news", "table")
        led.churn(100, hm(k, 12))
    for k in (2, 4):                                             # and some nights
        led.record(hm(k, 1), hm(k, 3, 30), "example.org", "other", "table")
    return BrowserTabSensor(enabled=True, ledger=led, clock=lambda: now), led, now


def t4_suggest():
    print("\n=== T4. Suggestions and habits ===")
    from src.daemon.browsing import Browsing, parse
    s, led, now = two_weeks()
    b = Browsing(lambda: s, clock=lambda: now)
    r = b.hear("any suggestions about my browsing?", now)
    check("what changed, with the numbers", r.startswith("Here's what stands out:") and "video is up: 15 hours this week against 3 hours last week" in r, r)
    check("...tabs opened and closed within seconds, against last week", "opening and closing tabs within seconds more" in r, r)
    check("...nights", "between midnight and 5" in r, r)
    check("...and when focus falls", "clusters between 10:00 and 12:00" in r, r)
    check("it observes; it does not scold", not any(w in r.lower() for w in ("you should", "you must", "stop ", "waste", "addict", "unhealthy")), r)

    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor
    now2 = hm(0, 15)
    led2 = Ledger(clock=lambda: now2)
    for k in list(range(1, 7)) + list(range(8, 14)):
        led2.record(hm(k, 20), hm(k, 21), "youtube.com", "video", "table")
    s2 = BrowserTabSensor(enabled=True, ledger=led2, clock=lambda: now2)
    check("a steady week: nothing stands out", Browsing(lambda: s2, clock=lambda: now2).hear("how am I doing with my time?", now2)
          == "Nothing stands out -- this week looks much like the last.")
    s3 = BrowserTabSensor(enabled=True, ledger=Ledger(clock=lambda: now2), clock=lambda: now2)
    check("too little: it says so", "enough browsing this week yet" in Browsing(lambda: s3, clock=lambda: now2).hear("any suggestions about my time", now2))

    r = b.hear("what are my browsing habits?", now)
    check("habits: how much, when, how long a stretch", "Over 12 days you browse about" in r and "typical stretch" in r and "mostly in the" in r, r)
    check("...and how many tabs an hour are opened and closed within seconds", "tabs an hour within seconds" in r, r)
    check("habits with too little data say what is needed", "enough yet" in Browsing(lambda: s3, clock=lambda: now2).hear("what are my browsing habits", now2))
    check("with tabs off, the general phrasings are left to the model", Browsing(lambda: None).hear("how can I use my time better?") is None
          and Browsing(lambda: None).hear("how do I browse") is None)
    check("...but a plain browsing question still says it is off", "off" in Browsing(lambda: None).hear("what have I been browsing this week?"))
    check("'how do I use the browser' is a habit question only when it asks about me", parse("how do I use the browser to open a pdf") is None)

    rt = te.rig()
    rt.brain.user = "Khalid"
    seen = []
    orig = rt.ash.think
    rt.ash.think = lambda *a, **k: (seen.append(a), orig(*a, **k))[1]
    rt.say("how can I use my time better?")
    check("on the runtime with tabs off: the model answers as before", seen and "not keeping track" not in rt.out[-1], f"{seen} {rt.out[-1]}")


def t5_panel():
    print("\n=== T5. What I'm learning ===")
    from src.daemon import learning
    from src.ui import hooks
    rt = te.rig()
    check("nothing learned: nothing to show", learning.snapshot(rt, 1.0) == {})
    s, led, now = two_weeks()
    rt.periphery.sensors["browser_tabs"] = s
    rt.t[0] = now
    from src.daemon.taste import Taste, TasteConfig
    rt.brain.user = "Khalid"
    from tst_taste import kb_with
    rt.taste = Taste(kb_with(), user="Khalid", cfg=TasteConfig(), clock=lambda: now)
    rt.taste.observe("I love Interstellar. I can't stand horror films.")
    for _ in range(6):
        rt.learner.update("joke", "idle/evening/leisure", 1.0, 1.0, now)
    learning._CACHE.clear()
    snap = learning.snapshot(rt, now)
    lines = snap.get("lines", [])
    check("tastes are listed", any(l.startswith("Tastes:") and "Interstellar" in l for l in lines), str(lines))
    check("browsing says what it holds and in what form", any("Browser tabs: watching" in l and "site and category only" in l for l in lines), str(lines))
    check("interests and focus", any(l.startswith("Interests:") for l in lines) and any(l.startswith("Focus:") for l in lines), str(lines))
    check("reactions", any(l.startswith("Reactions:") for l in lines), str(lines))
    check("and how to delete it", "forget my browsing" in snap["forget"])
    s.pause()
    learning._CACHE.clear()
    check("a paused sensor says paused", any("Browser tabs: paused" in l for l in learning.snapshot(rt, now)["lines"]))
    check("it is cached for a moment, not recomputed on every poll", learning.snapshot(rt, now + 5) is learning.snapshot(rt, now + 6))
    st = hooks._status(rt)
    check("the console's status carries it", "learning" in st and st["learning"].get("lines"), str(st.get("learning"))[:120])
    html = open(os.path.join(HERE, "src", "ui", "static", "console.html"), encoding="utf-8").read()
    fn = html[html.index("function drawLearning"):html.index("const MOOD_FACE")]
    check("the page has the panel, drawn with textContent (never innerHTML)", 'id="learning"' in html and "drawLearning(s.learning)" in html
          and "innerHTML" not in fn and "textContent" in fn)


def t6_no_llm():
    print("\n=== T6. No LLM ===")
    banned = {"requests", "openai", "ollama", "langchain", "langchain_core", "transformers", "torch", "urllib3", "httpx",
              "aiohttp", "anthropic", "tools"}
    for rel in ("src/daemon/uxreport.py", "src/daemon/learning.py", "src/daemon/browsing.py", "src/daemon/occasion.py"):
        tree = ast.parse(open(os.path.join(HERE, rel), encoding="utf-8").read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                mods.add(n.module.split(".")[0])
        check(f"{rel} imports no model or HTTP client", not (mods & banned), str(mods & banned))

    class Boom:
        touched = []

        def __getattr__(self, n):
            Boom.touched.append(n)
            raise AssertionError(f"the model was touched: {n}")
    rt = te.rig()
    rt.brain.user = "Khalid"
    s, led, now = two_weeks()
    rt.periphery.sensors["browser_tabs"] = s
    rt.t[0] = now
    rt.ash.llm = Boom()
    for q in ("any suggestions about my browsing?", "what are my browsing habits?", "how are your nudges landing?",
              "when do I focus best?", "what am I into lately?"):
        rt.say(q)
    check("five questions answered with the model rigged to explode", Boom.touched == [] and rt.out, str(Boom.touched))


if __name__ == "__main__":
    t1_occasion()
    t2_learner()
    t3_report()
    t4_suggest()
    t5_panel()
    t6_no_llm()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
