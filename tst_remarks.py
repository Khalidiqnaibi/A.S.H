"""
tst_remarks.py -- the remark log, the review, and a simulated day.

  K1  Writing: every line said, its kind, asked or not, the text shortened (or
      left out); reactions; a file per day; old days pruned
  K2  Reviewing: counts per kind, how each landed, the busiest hour, repeats,
      the small hours, and the tips each of those gives
  K3  The runtime: every line goes through the log, reactions included, and
      the console UI's own wrapper still sees everything
  K4  A simulated day with everything on: within the budgets, nothing in
      quiet hours or while muted, nothing said over and over, briefings once,
      and a review at the end with nothing alarming in it

Run:  PYTHONIOENCODING=utf-8 python tst_remarks.py
"""

import json
import os
import sys
import tempfile
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.remarks import RemarkLog, kind_of  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
T0 = time.mktime((2026, 3, 10, 9, 0, 0, 0, 0, -1))
HOUR, DAY = 3600.0, 86400.0


def log(**kw):
    t = [T0]
    d = tempfile.mkdtemp()
    r = RemarkLog(d, clock=lambda: t[0], **kw)
    r.t, r.d = t, d
    return r


def k1_writing():
    print("\n=== K1. Writing ===")
    r = log()
    r.said("It's 14:05.", {"addressed": True, "pathway": "fast"})
    r.said("Who is Omar to you?", {"addressed": False, "attention": {"kind": "wonder"}})
    r.reacted("Who is Omar to you?", "engaged", "wonder")
    rows = r.load()
    check("each line and reaction written", len(rows) == 3)
    check("an answer's kind", rows[0]["kind"] == "answer" and rows[0]["asked"])
    check("a question's kind, from the attention decision", rows[1]["kind"] == "wonder" and not rows[1]["asked"])
    check("the reaction, tied to the line", rows[2]["react"] == "engaged" and rows[2]["id"] == rows[1]["id"])
    check("the text kept, shortened", rows[1]["text"] == "Who is Omar to you?")
    r.said("x" * 500, {})
    check("(to 120 characters)", len(r.load()[-1]["text"]) == 120)
    r = log(keep_text=False)
    r.said("Who is Omar to you?", {"attention": {"kind": "wonder"}})
    check("keep_text off: no words at all", "text" not in r.load()[0])
    r = log()
    r.said("a", {})
    r.t[0] += DAY
    r.said("b", {})
    check("a file per day", len(os.listdir(r.d)) == 2)
    r.t[0] += 20 * DAY
    r.said("c", {})
    check("days older than two weeks are pruned", len(os.listdir(r.d)) == 1)
    check("no directory: nothing written, nothing fails", RemarkLog(None).said("x", {}) is None)
    check("kinds", kind_of({"addressed": True, "pathway": "watch"}) == "watch" and kind_of({"pathway": "reminder"}) ==
          "reminder" and kind_of({}) == "remark")


def k2_review():
    print("\n=== K2. Reviewing ===")
    r = log()
    for i in range(8):
        r.t[0] = T0 + i * 600
        r.said(f"joke {i % 2}", {"attention": {"kind": "moment"}})
        r.reacted(f"joke {i % 2}", "rejected" if i < 6 else "engaged", "moment")
    for i in range(5):
        r.said(f"answer {i}", {"addressed": True})
    rv = r.review()
    check("counts", rv["said"] == 13 and rv["asked"] == 5 and rv["unasked"] == 8)
    check("how a kind landed", rv["by_kind"]["moment"]["rejected"] == 6 and rv["by_kind"]["moment"]["engaged"] == 2)
    check("a kind waved off too often is a tip", any(t.startswith("moment: waved off 6 of 8") for t in rv["tips"]),
          str(rv["tips"]))
    check("a line said over and over is a tip", any("said 4 times" in t for t in rv["tips"]), str(rv["tips"]))
    check("answers are never 'repetitive'", not any("answer" in t for t in rv["tips"] if "said" in t))
    r = log()
    for i in range(8):
        r.said(f"remark {i}", {"attention": {"kind": "observation"}})
    check("a chatty hour is a tip", any("8 unprompted remarks in one hour" in t for t in r.review()["tips"]))
    r = log()
    r.t[0] = time.mktime((2026, 3, 10, 3, 0, 0, 0, 0, -1))
    r.said("psst", {})
    check("the small hours are a tip", any("midnight and 6am" in t for t in r.review(1, r.t[0] + HOUR)["tips"]))
    r = log()
    for i in range(6):
        r.said(f"q{i}", {"attention": {"kind": "wonder"}})
        r.reacted(f"q{i}", "silence", "wonder")
    check("a kind mostly unanswered is a tip", any("unanswered 6 of 6" in t for t in r.review()["tips"]))
    r = log()
    r.said("fine", {"addressed": True})
    rep = r.report()
    check("a quiet, healthy day: nothing to tune", rep.endswith("Nothing stands out to tune.") and
          "1 answers, 0 unasked" in rep, rep)
    check("only the days asked for", log().review(1, T0 - 3 * DAY)["said"] == 0)


class Ash(tw.Ash):
    def think(self, query, image=None, context=""):
        self.calls.append(query)
        return tw.Resp("Sure.")

    def volunteer(self, prompt, context=""):
        self.calls.append(prompt)
        return tw.Resp("Noted.")


def rt_rig(**extra):
    clock = [T0]
    out = []
    ash = Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    cfg = dict(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
               wonder=True, wonder_cfg={}, wonder_path=os.path.join(d, "w.json"),
               knowledge_path=os.path.join(d, "k.json"), moments=True, followups=True,
               followups_path=os.path.join(d, "f.json"), briefings=True, briefings_path=os.path.join(d, "b.json"),
               reminders=True, reminders_path=os.path.join(d, "r.json"), play=True,
               remarks_dir=os.path.join(d, "remarks"))
    cfg.update(extra)
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()),
                        gate=AttentionGate(quiet_hours=(23, 7)), privacy=PrivacyPolicy(),
                        config=AmbientConfig(**cfg), response_sink=lambda t, m: out.append((clock[0], t, m)))
    rt.clock = lambda: clock[0]
    for part in ("reminders", "followups"):
        obj = getattr(rt, part, None)
        if obj is not None:
            obj.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock = rt, ash, out, idle, clock

    def step(dt, idle_s, ticks=1):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True):
        clock[0] += 2
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    h.step, h.say = step, say
    return h


def k3_runtime():
    print("\n=== K3. The runtime ===")
    h = rt_rig()
    h.say("what's the time")
    h.say("set a timer for 1 minutes")
    h.step(61, 300.0)
    rows = h.rt.remarks.load(1, h.clock[0])
    kinds = [r["kind"] for r in rows if "react" not in r]
    check("every line goes through the log", len(kinds) == len(h.out) == 3, str(kinds))
    check("with its kind", "answer" in kinds and "reminder" in kinds, str(kinds))
    h.say("I told Omar about the project")
    for _ in range(15):
        h.step(10, 0.0)
    h.step(15, 25.0, ticks=2)
    h.step(5, 5.0)
    h.say("not now")
    rows = h.rt.remarks.load(1, h.clock[0])
    check("a reaction is logged against the line", any("react" in r for r in rows), str(rows[-3:]))
    check("review works on a live log", "ASH said" in h.rt.remarks.report(1, h.clock[0]))
    from src.ui import UIBus, hooks
    bus = UIBus()
    hooks.attach_ui(h.rt, bus, opener=None, status_every_s=10 ** 9)
    n = len(h.rt.remarks.load(1, h.clock[0]))
    h.say("what's the time")
    check("with the console UI attached, lines are still logged", len(h.rt.remarks.load(1, h.clock[0])) > n)


def k4_day():
    print("\n=== K4. A simulated day ===")
    h = rt_rig()
    h.clock[0] = time.mktime((2026, 3, 10, 7, 30, 0, 0, 0, -1))
    script = [
        (8, 0, "my exam is tomorrow at 9am"), (9, 15, "I told Omar about the deploy and the meeting"),
        (10, 30, "we should move the jobs to ZXQ"), (11, 0, "ZXQ is slow again"),
        (13, 0, "remind me to call mom in 30 minutes"), (15, 0, "Sara called about the project deadline"),
        (17, 0, "Sara and Omar are colleagues"), (20, 0, "what's the weather"),
    ]
    i = 0
    end = time.mktime((2026, 3, 11, 1, 0, 0, 0, 0, -1))
    while h.clock[0] < end:
        hr = time.localtime(h.clock[0]).tm_hour
        if i < len(script) and (hr, time.localtime(h.clock[0]).tm_min) >= script[i][:2]:
            h.say(script[i][2])
            i += 1
        working = 8 <= hr < 22
        # 50 minutes of work, then a pause; away overnight
        phase = (h.clock[0] // 60) % 60
        h.step(30, 0.0 if working and phase < 50 else (40.0 if working else 5000.0))
    said = [(t, s, m) for t, s, m in h.out]
    unasked = [(t, s, m) for t, s, m in said if not m.get("addressed")]
    per_hour = Counter(time.strftime("%H", time.localtime(t)) for t, s, m in unasked
                       if m.get("pathway") not in ("reminder", "watch"))
    check("(the day was busy enough to say things)", len(unasked) >= 5, str(len(unasked)))
    check("never more unprompted remarks in an hour than the budget allows",
          max(per_hour.values() or [0]) <= h.rt.gate.max_offers_per_hour + 2, str(per_hour))
    night = [s for t, s, m in unasked if (time.localtime(t).tm_hour >= 23 or time.localtime(t).tm_hour < 7)
             and m.get("pathway") not in ("reminder", "watch")]
    check("nothing unprompted in quiet hours", night == [], str(night))
    reps = Counter(s for t, s, m in unasked)
    check("nothing said over and over", max(reps.values() or [0]) <= 2, str(reps.most_common(3)))
    briefs = [s for t, s, m in said if s.startswith(("Good morning.", "Here's your day."))]
    check("each briefing at most once", sum(s.startswith("Here's your day.") for s in briefs) <= 1
          and sum(s.startswith("Good morning.") for s in briefs) <= 1, str(briefs))
    check("the reminder went off", any(s == "Reminder: call mom." for t, s, m in said))
    rv = h.rt.remarks.review(2, h.clock[0])
    check("the review agrees with what was said", rv["said"] == len(said), f"{rv['said']} vs {len(said)}")
    check("and finds nothing alarming", not any("between midnight" in t or "too chatty" in t for t in rv["tips"]),
          str(rv["tips"]))
    check("it does flag what a real day would want tuned (questions met with silence)",
          any(t.startswith("wonder: unanswered") for t in rv["tips"]), str(rv["tips"]))
    print("     (the day's review)\n" + "\n".join("     " + ln for ln in h.rt.remarks.report(2, h.clock[0]).split("\n")))


if __name__ == "__main__":
    for fn in (k1_writing, k2_review, k3_runtime, k4_day):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
