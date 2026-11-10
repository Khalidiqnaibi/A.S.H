"""
tst_browsing.py -- what the ledger says about the user, said back by rule.

  T1  Reading the question: what is, and is not, about browsing
  T2  The profile: totals, interests (recent, repeated, fading), focus hours, "not enough yet"
  T3  The commands: tag what is in front, pause, resume, forget (all, a site, a category, tags)
  T4  On the runtime: answered by rule, never by the model, never to a guest, honest when it is off

Run:  PYTHONIOENCODING=utf-8 python tst_browsing.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
DAY = 86400.0


def midnight(k=0, base=None):
    t = time.localtime(base or time.time())
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday - k, 0, 0, 0, 0, 0, -1))


def hm(k, h, m=0):
    return midnight(k) + h * 3600 + m * 60


def world(fill=True):
    """A sensor with a ledger holding six days of a plausible life."""
    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor
    now = hm(0, 15)                                    # 15:00 today
    led = Ledger(clock=lambda: now)
    if fill:
        for k in range(1, 7):
            led.record(hm(k, 10), hm(k, 11), "github.com", "code", "table")
            led.record(hm(k, 11), hm(k, 11, 30), "docs.google.com", "docs", "table")
            led.record(hm(k, 13), hm(k, 13, 10), "bbc.com", "news", "table")
            led.record(hm(k, 20), hm(k, 20, 40), "youtube.com", "video", "table")
        led.record(hm(0, 9), hm(0, 9, 30), "github.com", "code", "table")
    s = BrowserTabSensor(enabled=True, ledger=led, clock=lambda: now)
    return s, led, now


def ask(b, text, now):
    return b.hear(text, now)


def t1_parse():
    print("\n=== T1. Is this about browsing? ===")
    from src.daemon.browsing import parse
    yes = {"what have I been browsing this week?": "summary", "where did my time go today": "summary",
           "how much time did I spend on youtube this week?": "time_on", "how long did I spend on github.com": "time_on",
           "how much time have I spent on social sites": "time_on", "what sites do I use the most?": "top",
           "my most visited sites": "top", "what am I into lately?": "interests", "when do I focus best?": "focus",
           "when am I most productive": "focus", "what do you know about my browsing?": "status",
           "stop watching my tabs": "pause", "pause tab tracking": "pause", "resume watching my tabs": "resume",
           "forget my browsing": "forget_all", "delete my browsing history": "forget_all",
           "forget everything about github.com": "forget_site", "forget my social browsing": "forget_cat",
           "tag this site as work": "tag", "this site is for learning": "tag", "that page is a work page": None,
           "forget my tags": "forget_tags", "hey ash, what have I been browsing today": "summary"}
    got = {q: (parse(q) or (None,))[0] for q in yes}
    bad = {q: g for q, g in got.items() if g != yes[q]}
    check("the ways people ask", not bad, str(bad))
    no = ["what's the weather", "how long did I spend on homework", "forget Interstellar", "forget what I said about Omar",
          "that's work", "this is work", "what do I like?", "undo that", "remind me to forget my keys", "tell me about my day",
          "I'm browsing for shoes", "stop the music", "what is a browser"]
    check("ordinary talk is not taken for it", all(parse(q) is None for q in no), str([q for q in no if parse(q)]))
    k, d = parse("how much time did I spend on youtube this week?")
    check("details are read: site and period", d == {"site": "youtube.com", "per": "this week"}, str(d))
    check("synonyms: 'coding' is code", parse("tag this site as coding")[1]["cat"] == "code")


def t2_profile():
    print("\n=== T2. The profile ===")
    from src.daemon.browsing import Browsing, dur
    s, led, now = world()
    b = Browsing(lambda: s, clock=lambda: now)
    check("durations read like a person", [dur(30), dur(60), dur(3600), dur(5400), dur(7260)] ==
          ["under a minute", "1 minute", "1 hour", "1 hour 30 minutes", "2 hours 1 minute"])
    r = ask(b, "what have I been browsing this week?", now)
    check("a week by category, biggest first", r.startswith("This week you spent") and r.index("code") < r.index("video")
          and "docs" in r and "news" in r, r)
    r = ask(b, "how much time did I spend on youtube this week?", now)
    check("one site", r == "This week you spent 4 hours on youtube.com.", r)
    r = ask(b, "how much time did I spend on video sites this week", now)
    check("one category", "4 hours on video sites" in r, r)
    r = ask(b, "how much time did I spend on github today", now)
    check("today only counts today (30 minutes)", r == "Today you spent 30 minutes on github.com.", r)
    r = ask(b, "how much time did I spend on reddit.com this week", now)
    check("a site never seen is said so", r == "I haven't seen you on reddit.com this week.", r)
    r = ask(b, "what sites do I use the most?", now)
    check("top sites with their time", r.startswith("This week, mostly github.com (") and "youtube.com" in r, r)
    r = ask(b, "what am I into lately?", now)
    check("interests are sites that came back on several days", "github.com (code)" in r and "youtube.com (video)" in r, r)

    # fading: an old heavy habit ranks below a lighter recent one
    s2, led2, now2 = world(fill=False)
    b2 = Browsing(lambda: s2, clock=lambda: now2)
    for k in range(25, 30):
        led2.record(hm(k, 10), hm(k, 11), "oldhabit.com", "games", "table")          # 1 h a day, a month ago
    for k in range(1, 5):
        led2.record(hm(k, 10), hm(k, 10, 30), "newhabit.com", "learning", "table")   # 30 min a day, this week
    led2.record(hm(1, 15), hm(1, 18), "binge.com", "video", "table")                # 3 hours, but only one day
    rows = b2.interests(now2)
    check("interests fade: recent beats old", [r[0] for r in rows][:2] == ["newhabit.com", "oldhabit.com"], str([(r[0], round(r[2])) for r in rows]))
    check("...and one big day is not an interest", all(r[0] != "binge.com" for r in rows))
    r = ask(b2, "what am I into lately?", now2)
    check("...said", "newhabit.com" in r and "binge" not in r, r)

    r = ask(b, "when do I focus best?", now)
    check("focus hours: the two hours holding most of it, with the evidence", "between 10:00 and 12:00" in r and "over 7 days" in r, r)
    check("...and the longest unbroken stretch (10:00-11:30 is one stretch)", "1 hour 30 minutes" in r, r)
    fresh, _l, n3 = world(fill=False)
    fresh.ledger.record(n3 - 7200, n3 - 3600, "github.com", "code", "table")
    r = ask(Browsing(lambda: fresh, clock=lambda: n3), "when do I focus best?", n3)
    check("too little data: it says so and what it needs", r.startswith("I don't have enough yet") and "3 days" in r and "1 day" in r, r)
    r = ask(Browsing(lambda: world(False)[0], clock=lambda: n3), "what have I been browsing today", n3)
    check("an empty ledger is not a made-up answer", "haven't seen much" in r, r)
    r = ask(Browsing(lambda: world(False)[0], clock=lambda: n3), "what am I into lately?", n3)
    check("...nor interests from nothing", r.startswith("Nothing stands out yet"), r)
    r = ask(b, "what do you know about my browsing?", now)
    check("it says what is kept, what is not, and how to delete it", "not pages" in r and "forget my browsing" in r and "days" in r, r)


def t3_commands():
    print("\n=== T3. Commands ===")
    from src.daemon.browsing import Browsing
    s, led, now = world()
    b = Browsing(lambda: s, clock=lambda: now)
    s.submit({"focused": True, "tabs": [dict(id=1, url="https://acme-internal.dev/board", title="Quarterly numbers", active=True)]})
    s.read()
    r = ask(b, "tag this site as work", now)
    check("tag what is in front", r == "Okay, acme-internal.dev is work from now on." and s.current()["category"] == "work", r)
    s.pause()
    check("...nothing in front: honest", ask(b, "tag this page as learning", now) == "I don't see a page in front right now to tag.")
    r = ask(b, "stop watching my tabs", now)
    check("pause", "stopped watching" in r and s.paused)
    r = ask(b, "what have I been browsing this week?", now)
    check("...and while paused it does not answer from stale data", "not watching" in r, r)
    r = ask(b, "resume watching my tabs", now)
    check("resume", "watching your tabs again" in r and not s.paused)
    n = led.counts()["episodes"]
    r = ask(b, "forget everything about youtube.com", now)
    check("forget a site", "Forgotten" in r and not led.totals(0, now + 1, site="youtube.com") and led.counts()["episodes"] < n, r)
    r = ask(b, "forget my news browsing", now)
    check("forget a category", "Forgotten" in r and not led.totals(0, now + 1, category="news"), r)
    r = ask(b, "forget my tags", now)
    check("forget the tags", "forgotten the sites you tagged" in r and s.cls.status()["tags"] == 0, r)
    r = ask(b, "forget my browsing", now)
    check("forget everything", "deleted everything" in r and led.counts() == {"episodes": 0, "daily": 0, "sites": 0}, r)
    r = ask(b, "forget my browsing", now)
    check("...and says so when there is nothing", r == "I didn't have anything about your browsing.", r)
    check("without the sensor, tagging is ordinary talk (not handled)", Browsing(lambda: None).hear("this site is work") is None)
    check("...and a question says it is off, not made up", "off" in Browsing(lambda: None).hear("what have I been browsing this week?"))


def t4_runtime():
    print("\n=== T4. On the runtime ===")
    from src.senses.base import Modality, SensorEvent

    class Boom:
        touched = []

        def __getattr__(self, n):
            Boom.touched.append(n)
            raise AssertionError(f"the model was touched: {n}")
    rt = te.rig()
    rt.brain.user = "Khalid"
    seen = []
    orig = rt.ash.think
    rt.ash.think = lambda *a, **k: (seen.append(a), orig(*a, **k))[1]
    rt.say("what have I been browsing this week?")
    check("with tabs off it says so, honestly, without the model", "off" in rt.out[-1] and not seen, f"{rt.out[-1]!r} {seen}")
    rt.say("stop watching my tabs")
    check("...same for the controls", "not watching your tabs" in rt.out[-1], rt.out[-1])

    s, led, now = world()
    s.clock = lambda: now
    rt.periphery.sensors["browser_tabs"] = s
    rt.t[0] = now
    rt.browsing.clock = lambda: now
    rt.ash.llm = Boom()
    seen.clear()
    rt.say("how much time did I spend on youtube this week?")
    check("answered from the ledger", "4 hours on youtube.com" in rt.out[-1], rt.out[-1])
    check("the model was not asked and not touched", not seen and Boom.touched == [], f"{seen} {Boom.touched}")
    rt.say("what sites do I use the most?")
    check("top sites by voice", "github.com" in rt.out[-1], rt.out[-1])
    st = rt.status()
    check("the status carries the sensor's state", st["browsing"] and "dropped" in st["browsing"], str(st.get("browsing")))

    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="what have I been browsing this week?",
                          addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice is refused", rt.out[-1].startswith("Sorry, I only do that for Khalid"), rt.out[-1])
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="forget my browsing", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("...and cannot delete it either", led.counts()["episodes"] > 0 and rt.out[-1].startswith("Sorry, I only do that"), rt.out[-1])

    rt.say("forget Interstellar")
    check("'forget X' for anything else still goes the old way", "browsing" not in rt.out[-1].lower(), rt.out[-1])
    rt.say("forget my browsing")
    check("forget everything, by voice", led.counts()["episodes"] == 0 and "deleted everything" in rt.out[-1], rt.out[-1])
    check("still no model", Boom.touched == [], str(Boom.touched))


if __name__ == "__main__":
    t1_parse()
    t2_profile()
    t3_commands()
    t4_runtime()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
