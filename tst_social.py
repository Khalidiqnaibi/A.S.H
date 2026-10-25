"""
tst_social.py -- the social loop: when to speak, and learning from how it lands.

Everything here runs on a fake clock, so hours of behaviour take milliseconds
and nothing depends on when the test is run.

  S1  reading replies ("yes please", "not now", a long new request)
  S2  situations: busy / time-of-day buckets
  S3  SocialLearner: prior, direction, bounds, specificity, hierarchy, urgency
      immunity, forgetting, exploration, snooze, persistence
  S4  ReactionTracker: accepted / engaged / rejected / dismissed / silence
  S5  FlowDetector: free moments and being stuck, from raw events
  S6  The gate: offers, the learned bar, and the hard stops that stay hard
  S7  The runtime end to end: offers delivered, replies learned from, "not now"
      honoured, a simulated user who dislikes offers stops getting them and one
      who likes them keeps getting them
  S8  Switches: learning off, offers off, persistence, status

Run:  python tst_social.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

from src.daemon import (AmbientConfig, AmbientRuntime, AttentionGate, FlowDetector,  # noqa: E402
                        Journal, PrivacyPolicy, ReactionTracker, SocialLearner)
from src.daemon.social import classify_reply, context_key, error_signature  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


T0 = time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))      # a Tuesday, 2pm local
HOUR = 3600.0
DAY = 86400.0
CTX = context_key(0.6, T0)                                # "active/afternoon"


class Drives:
    curiosity = 0.5
    fatigue = 0.0
    urgency = 0.0
    caution = 0.0


def offer_ev(kind="free_moment", strength=0.7):
    return SensorEvent(source="flow", modality=Modality.TEXT, text="x",
                       meta={"offer": kind, "offer_strength": strength})


def sys_ev(urgency=0.95, source="system"):
    return SensorEvent(source=source, modality=Modality.TELEMETRY, text="battery at 3%",
                       salience=0.9, urgency=urgency)


def err_ev(n, process=None, dwell=None):
    meta = {}
    if process:
        meta = {"process": process, "dwell_s": dwell or 0}
    return SensorEvent(source="screen", modality=Modality.TEXT, meta=meta,
                       text=f"Traceback: ValueError: bad input at C:\\proj\\run.py line {n}")


def learner(**kw):
    clock = [T0]
    kw.setdefault("clock", lambda: clock[0])
    lr = SocialLearner(**kw)
    lr._t = clock
    return lr


# ======================================================================
def s1_replies():
    print("\n=== S1. Reading replies ===")
    cases = {
        "yes please": "accepted", "sure": "accepted", "thanks": "accepted", "ok": "accepted",
        "go ahead": "accepted", "that would help": "accepted", "show me": "accepted",
        "not now": "deferred", "later": "deferred", "no thanks": "rejected", "be quiet": "rejected",
        "leave me alone": "rejected", "stop interrupting": "rejected", "no": "rejected",
        "I'm busy": "deferred", "shut up": "rejected", "don't bother me": "rejected",
        "what time is it": "engaged", "tell me about the weather today": "engaged",
        "yes the parser fails because the input is empty and I have tried everything": "engaged",
        "no I don't think that is what the traceback says at all, look at line twelve": "engaged",
        "": "engaged",
    }
    for text, want in cases.items():
        got = classify_reply(text)
        check(f"{text!r:.60} -> {want}", got == want, f"got {got}")


def s2_context():
    print("\n=== S2. Situations ===")
    check("typing / active / idle / away buckets",
          [context_key(b, T0).split("/")[0] for b in (1.0, 0.6, 0.25, 0.0)]
          == ["typing", "active", "idle", "away"])
    parts = {h: context_key(0.0, time.mktime((2026, 3, 10, h, 0, 0, 0, 0, -1))).split("/")[1]
             for h in (3, 8, 14, 19, 23)}
    check("time of day", parts == {3: "night", 8: "morning", 14: "afternoon",
                                   19: "evening", 23: "night"}, str(parts))


def s3_learner():
    print("\n=== S3. SocialLearner ===")
    lr = learner()
    check("no evidence: estimate is the prior", abs(lr.p("a", CTX) - 0.5) < 1e-9)
    check("no evidence: the bar does not move", abs(lr.bar_shift("a", CTX)) < 1e-9)

    for _ in range(6):
        lr.update("a", CTX, 0.0)
    check("unwelcome remarks raise the bar", lr.bar_shift("a", CTX) > 0.1, f"{lr.bar_shift('a', CTX):.3f}")
    check("...but never beyond the limit", lr.bar_shift("a", CTX) <= lr.max_shift + 1e-9)
    for _ in range(100):
        lr.update("a", CTX, 0.0)
    check("a hundred more stay within the limit", lr.bar_shift("a", CTX) <= lr.max_shift + 1e-9)

    lr = learner()
    for _ in range(6):
        lr.update("b", CTX, 1.0)
    check("welcome remarks lower the bar", lr.bar_shift("b", CTX) < -0.1, f"{lr.bar_shift('b', CTX):.3f}")
    check("...within the limit", lr.bar_shift("b", CTX) >= -lr.max_shift - 1e-9)

    lr = learner()
    one = learner()
    one.update("a", CTX, 0.0)
    for _ in range(8):
        lr.update("a", CTX, 0.0)
    check("one bad reaction moves it less than eight", 0 < one.bar_shift("a", CTX) < lr.bar_shift("a", CTX))

    # Specificity: dismissed while typing, welcomed while idle.
    lr = learner()
    busy, calm = context_key(1.0, T0), context_key(0.25, T0)
    for _ in range(8):
        lr.update("chat", busy, 0.0)
        lr.update("chat", calm, 1.0)
    check("the same remark is held back where it was unwelcome, offered where it was welcome",
          lr.bar_shift("chat", busy) > 0.05 > -0.05 > lr.bar_shift("chat", calm),
          f"{lr.bar_shift('chat', busy):.2f} / {lr.bar_shift('chat', calm):.2f}")
    check("a situation it has no record of sits between the two",
          lr.bar_shift("chat", busy) > lr.bar_shift("chat", "away/night") > lr.bar_shift("chat", calm))

    # Hierarchy: what it learns about a kind carries to a new situation, and
    # about the user in general to a new kind.
    lr = learner()
    for _ in range(8):
        lr.update("nag", CTX, 0.0)
    check("a new situation inherits the kind's record", lr.bar_shift("nag", "idle/night") > 0.05)
    check("a new kind inherits a little of the general record",
          lr.bar_shift("other", CTX) > 0.0 and lr.bar_shift("other", CTX) < lr.bar_shift("nag", CTX))

    # Urgency immunity.
    check("urgency 1.0 removes the shift entirely: nothing learned can silence an emergency",
          lr.bar_shift("nag", CTX, urgency=1.0) == 0.0)
    check("urgency scales it down", 0 < lr.bar_shift("nag", CTX, urgency=0.5) < lr.bar_shift("nag", CTX))

    # Exploration.
    lr = learner()
    e0 = lr.exploration("a", CTX)
    for _ in range(9):
        lr.update("a", CTX, 0.5)
    check("exploration is largest with no evidence and shrinks with it",
          e0 > lr.exploration("a", CTX) > 0 and abs(e0 - lr.explore) < 1e-9)
    check("no exploration bonus on an emergency", lr.exploration("a", CTX, urgency=1.0) == 0.0)

    # Forgetting: changing their mind is followed, and it fades with time.
    lr = learner()
    for _ in range(6):
        lr.update("a", CTX, 0.0)
    low = lr.p("a", CTX)
    for _ in range(25):
        lr.update("a", CTX, 1.0)
    check("a user who changes their mind is followed", lr.p("a", CTX) > 0.5 > low,
          f"{low:.2f} -> {lr.p('a', CTX):.2f}")
    lr = learner()
    lr.update("a", CTX, 0.0)
    lr.update("a", CTX, 0.0)
    shy = lr.bar_shift("a", CTX)
    lr._t[0] += 2 * DAY
    later = lr.bar_shift("a", CTX)
    lr._t[0] += 20 * DAY
    much_later = lr.bar_shift("a", CTX)
    check("what was learned fades back toward the prior with time (no permanent mute)",
          shy > later > much_later >= 0 and much_later < 0.01, f"{shy:.3f} {later:.3f} {much_later:.4f}")

    # Snooze.
    lr = learner(snooze_s=600)
    lr.snooze("a", T0)
    check("snoozed right after", lr.snoozed("a", T0 + 1))
    check("only that kind", not lr.snoozed("b", T0 + 1))
    check("snooze ends", not lr.snoozed("a", T0 + 601))

    # Reward is clamped; junk does not poison it.
    lr = learner()
    lr.update("a", CTX, 5.0)
    lr.update("a", CTX, -3.0)
    check("rewards are clamped to 0..1", 0.0 <= lr.p("a", CTX) <= 1.0)

    # Persistence.
    path = os.path.join(tempfile.mkdtemp(), "sub", "social.json")
    lr = learner(path=path)
    for _ in range(5):
        lr.update("a", CTX, 0.0)
    lr.snooze("a", T0, 600)
    again = SocialLearner(path=path, clock=lambda: T0)
    check("record survives a restart", abs(again.p("a", CTX) - lr.p("a", CTX)) < 1e-9)
    check("snooze survives a restart", again.snoozed("a", T0 + 10))
    check("update count survives", again.updates == 5)
    with open(path, "w") as fh:
        fh.write("{not json")
    broken = SocialLearner(path=path, clock=lambda: T0)
    check("a corrupt record is survivable: starts fresh", abs(broken.p("a", CTX) - 0.5) < 1e-9)
    check("no path: nothing written, nothing raised",
          learner().save() is None)
    st = lr.status()
    check("status reports kinds and snoozes", "a" in st["kinds"] and st["kinds"]["a"]["p"] < 0.5
          and "a" in st["snoozed"], str(st))


def tracker(**kw):
    lr = learner()
    return ReactionTracker(lr, **kw), lr


def s4_tracker():
    print("\n=== S4. ReactionTracker ===")
    tr, lr = tracker(window_s=100)

    def say(kind="free_moment", at=T0, acc="ctx"):
        return tr.register(kind, CTX, "hello", at, accept_context=acc)

    def reply(text, at, **kw):
        return SensorEvent(source="mic", text=text, addressed=True, **kw)

    say()
    r = tr.on_user_event(reply("yes please", T0 + 5), T0 + 5)
    check("'yes please' is accepted and counted", r and r.outcome == "accepted" and tr.counts["accepted"] == 1)
    check("accepting raised the estimate", lr.p("free_moment", CTX) > 0.5)
    check("the offer is no longer pending", not tr.pending)
    check("the reaction carries the offer's context", r.offer.accept_context == "ctx")

    say(at=T0 + 10)
    r = tr.on_user_event(reply("what is the weather", T0 + 20), T0 + 20)
    check("talking about something else is engagement", r and r.outcome == "engaged")

    say(at=T0 + 30)
    r = tr.on_user_event(reply("not now", T0 + 35), T0 + 35)
    check("'not now' is a deferral", r and r.outcome == "deferred")
    check("...and snoozes that kind of remark", lr.snoozed("free_moment", T0 + 36))
    check("...but not others", not lr.snoozed("stuck_help", T0 + 36))
    check("...for the default time", not lr.snoozed("free_moment", T0 + 35 + lr.snooze_s + 1))
    say(kind="chat", at=T0 + 37)
    r = tr.on_user_event(reply("be quiet", T0 + 38), T0 + 38)
    check("'be quiet' is a rejection", r and r.outcome == "rejected")
    check("...snoozed for much longer", lr.snoozed("chat", T0 + 38 + lr.snooze_s * 3)
          and not lr.snoozed("chat", T0 + 38 + lr.snooze_s * 4 + 1))

    say(kind="stuck_help", at=T0 + 40)
    r = tr.on_user_event(SensorEvent(source="notification", text="", meta={"dismissed": True}), T0 + 45)
    check("a dismissed notification is a cost", r and r.outcome == "dismissed")

    say(at=T0 + 50)
    r = tr.on_user_event(SensorEvent(source="screen", text="window changed"), T0 + 55)
    check("an unrelated sensor event is not a reaction", r is None and len(tr.pending) == 1)
    r = tr.on_user_event(SensorEvent(source="flow", text="x", addressed=True), T0 + 56)
    check("ASH's own events are not a reaction", r is None and len(tr.pending) == 1)

    out = tr.expire(T0 + 50 + 101)
    check("no reaction within the window counts as silence", len(out) == 1 and out[0].outcome == "silence")
    check("...and it snoozes nothing new", tr.counts["silence"] == 1
          and lr.snoozes.get("free_moment", 0) <= T0 + 35 + lr.snooze_s)

    check("nothing pending: a reply is just a reply", tr.on_user_event(reply("no", T0 + 500), T0 + 500) is None)

    # Silence weighs less than a dismissal.
    a, la = tracker(window_s=10)
    b, lb = tracker(window_s=10)
    for _ in range(4):
        a.register("k", CTX, "x", T0)
        a.expire(T0 + 11)
        b.register("k", CTX, "x", T0)
        b.on_user_event(SensorEvent(source="n", text="", meta={"dismissed": True}), T0 + 1)
    check("silence is a weaker negative than a dismissal", la.p("k", CTX) > lb.p("k", CTX),
          f"{la.p('k', CTX):.2f} vs {lb.p('k', CTX):.2f}")
    check("silence is still negative", la.p("k", CTX) < 0.5)

    # The latest of several pending offers is the one that is answered.
    tr, lr = tracker(window_s=100)
    tr.register("old", CTX, "x", T0)
    tr.register("new", CTX, "y", T0 + 5)
    r = tr.on_user_event(reply("sure", T0 + 6), T0 + 6)
    check("a reply answers the most recent remark", r.offer.kind == "new" and len(tr.pending) == 1)

    # Bounded memory.
    tr, _ = tracker()
    for i in range(50):
        tr.register("k", CTX, "x", T0 + i)
    check("pending offers are bounded", len(tr.pending) <= 8)
    check("status", tr.status()["pending"] == len(tr.pending))


def feed(fd, seconds, idle, start, step=5):
    """Poll the detector every `step` seconds for `seconds`; return events."""
    out = []
    t = start
    while t <= start + seconds:
        out += fd.poll(idle, now=t)
        t += step
    return out


def s5_flow():
    print("\n=== S5. FlowDetector ===")
    # --- free moments
    fd = FlowDetector()
    check("working produces nothing", feed(fd, 150, 0, T0) == [])
    ev = fd.poll(25, now=T0 + 155)
    check("a pause after a real stretch of work is a free moment",
          len(ev) == 1 and ev[0].meta["offer"] == "free_moment" and ev[0].source == "flow")
    check("it is not addressed, urgent or persisted",
          not ev[0].addressed and ev[0].urgency == 0.0 and not ev[0].persistable)
    check("it says how long they worked", "minutes of work" in ev[0].text)
    check("it carries a strength and a context for a follow-up",
          0.5 < ev[0].meta["offer_strength"] <= 0.8 and ev[0].meta["accept_context"])
    check("only once per stretch", fd.poll(30, now=T0 + 160) == [] and fd.poll(60, now=T0 + 200) == [])

    fd = FlowDetector()
    feed(fd, 60, 0, T0)
    check("a short burst is not worth a chat", fd.poll(25, now=T0 + 90) == [])

    fd = FlowDetector()
    feed(fd, 150, 0, T0)
    check("not right at the keystroke", fd.poll(10, now=T0 + 151) == [])
    check("not once they are plainly away", fd.poll(400, now=T0 + 500) == [])

    fd = FlowDetector()
    feed(fd, 150, 0, T0)
    fd.poll(25, now=T0 + 155)
    feed(fd, 150, 0, T0 + 300)
    check("cooldown: no second chat soon after the first", fd.poll(25, now=T0 + 455) == [])
    feed(fd, 150, 0, T0 + 1500)
    check("...but later there is", len(fd.poll(25, now=T0 + 1655)) == 1)

    fd = FlowDetector()
    feed(fd, 100, 0, T0)
    feed(fd, 100, 0, T0 + 130)      # a 30s thought in the middle: one burst
    check("a thinking pause inside a burst does not split it", len(fd.poll(25, now=T0 + 235)) == 1)

    # --- stuck
    def stuck_after(errors, **kw):
        fd = FlowDetector(**kw)
        for i in range(errors):
            fd.observe(err_ev(10 + i), T0 + i * 20)
        return fd, fd.poll(8, now=T0 + errors * 20)

    for n, want in ((1, False), (2, False), (3, False), (4, False), (5, True), (8, True)):
        _, ev = stuck_after(n)
        check(f"the same error {n}x -> {'offers help' if want else 'stays quiet'}", bool(ev) == want)

    fd, ev = stuck_after(5)
    m = ev[0].meta
    check("the offer is a question, with the topic", m["offer"] == "stuck_help" and "?" in m["say"]
          and "keeps coming back" in m["say"] and "ValueError" in m["say"], m["say"])
    check("it says why", "same error 5 times" in m["reasons"])
    check("strength is a usable score", 0.6 <= m["offer_strength"] <= 1.0)
    check("it is not urgent", ev[0].urgency == 0.0)

    fd = FlowDetector()
    for i in range(5):
        fd.observe(err_ev(i), T0 + i)
    check("never over someone mid-keystroke", fd.poll(1, now=T0 + 10) == [])
    check("...but at the pause", len(fd.poll(8, now=T0 + 11)) == 1)
    check("cooldown: once, not every tick", fd.poll(8, now=T0 + 12) == [] and fd.poll(5, now=T0 + 600) == [])
    for i in range(5):
        fd.observe(err_ev(i), T0 + 2000 + i)
    check("after the cooldown, if it persists, it asks again", len(fd.poll(8, now=T0 + 2010)) == 1)

    fd = FlowDetector()
    for i in range(4):
        fd.observe(SensorEvent(source="screen", text=f"Traceback: error number {i}: {'x' * (i + 1)}-failure"),
                   T0 + i)
    for i, t in enumerate(("disk failed to mount", "cannot open socket", "fatal: not a repo", "build failed")):
        fd.observe(SensorEvent(source="term", text=t), T0 + 10 + i)
    check("different errors, once each, are not 'the same error'", fd.poll(8, now=T0 + 20) == [])

    fd = FlowDetector()
    for i in range(5):
        fd.observe(err_ev(i), T0 + i)
    check("errors from an hour ago are not evidence", fd.poll(8, now=T0 + 4000) == [])
    check("the same signature across paths and numbers",
          error_signature("Error at C:\\a\\b.py line 12") == error_signature("Error at /home/x/b.py line 99")
          or error_signature("Error at C:\\a\\b.py line 12") == error_signature("Error at C:\\z\\q.py line 7"))

    # Combining evidence.
    fd = FlowDetector()
    feed(fd, 30, 0, T0)
    fd.observe(err_ev(1, process="code", dwell=1500), T0 + 31)
    fd.observe(err_ev(2), T0 + 32)
    fd.observe(err_ev(3), T0 + 33)
    ev = fd.poll(8, now=T0 + 34)
    check("three of the same error plus 25 minutes in one window is enough",
          len(ev) == 1 and "minutes in code" in ev[0].meta["reasons"], str(ev and ev[0].meta["reasons"]))

    fd = FlowDetector()
    feed(fd, 30, 0, T0)
    fd.observe(err_ev(1, process="code", dwell=1500), T0 + 31)
    check("a long time in one window alone is not", fd.poll(8, now=T0 + 34) == [])

    def asks(fd, qs, t0):
        for i, q in enumerate(qs):
            fd.observe(SensorEvent(source="mic", text=q, addressed=True), t0 + i * 30)

    same = ["why does my parser fail on empty input", "parser fails on empty input why",
            "how do I stop the parser failing on empty input"]
    fd = FlowDetector()
    asks(fd, same, T0)
    check("asking the same thing three ways is not, alone, enough", fd.poll(8, now=T0 + 100) == [])
    for i in range(3):
        fd.observe(err_ev(i), T0 + 101 + i)
    ev = fd.poll(8, now=T0 + 110)
    check("...with the same error three times it is",
          len(ev) == 1 and "same question 3 times" in ev[0].meta["reasons"], str(ev and ev[0].meta["reasons"]))

    fd = FlowDetector()
    asks(fd, ["what is the capital of france", "how tall is mount everest", "tell me a joke about cats"], T0)
    for i in range(3):
        fd.observe(err_ev(i), T0 + 101 + i)
    check("different questions do not count as one repeated", fd.poll(8, now=T0 + 110) == [])

    fd = FlowDetector()
    asks(fd, same[:2], T0)
    for i in range(3):
        fd.observe(err_ev(i), T0 + 101 + i)
    check("a question twice plus an error three times is enough", len(fd.poll(8, now=T0 + 110)) == 1)

    fd = FlowDetector()
    fd.observe(SensorEvent(source="flow", text="Traceback error failed", meta={"offer": "x"}), T0)
    for i in range(6):
        fd.observe(SensorEvent(source="flow", text="Traceback error failed"), T0 + i)
    check("its own events are never evidence", fd.poll(8, now=T0 + 10) == [] and not fd.errors)

    fd = FlowDetector()
    feed(fd, 150, 0, T0)
    for i in range(5):
        fd.observe(err_ev(i), T0 + 151 + i)
    ev = fd.poll(25, now=T0 + 160)
    check("stuck takes precedence over a free moment",
          len(ev) == 1 and ev[0].meta["offer"] == "stuck_help")
    check("status counts what it raised", fd.status()["emitted"]["stuck_help"] == 1)


def gate(learner=None, **kw):
    return AttentionGate(learner=learner, **kw)


def s6_gate():
    print("\n=== S6. The gate ===")
    d = Drives()

    g = gate()
    r = g.decide(offer_ev(strength=0.7), d, user_busy=0.6, now=T0)
    check("without a learner an offer strong enough speaks", r.disposition.value == "respond")
    check("the decision names the kind and situation", r.kind == "free_moment" and r.ctx == CTX
          and r.as_dict()["kind"] == "free_moment")
    r = gate().decide(offer_ev(strength=0.5), d, user_busy=0.6, now=T0)
    check("a weak one does not", r.disposition.value == "observe")

    lr = learner()
    g = gate(lr)
    r = g.decide(offer_ev(strength=0.55), d, user_busy=0.6, now=T0)
    check("with no record, exploration lets a borderline offer through (so it can learn)",
          r.disposition.value == "respond", f"{r.score:.2f} vs {r.threshold:.2f}")

    # Dismissed: it stops.
    lr = learner()
    g = gate(lr)
    for _ in range(3):
        lr.update("free_moment", CTX, 0.0)
    r = g.decide(offer_ev(strength=0.7), d, user_busy=0.6, now=T0)
    check("after being dismissed it holds back an offer that would otherwise pass",
          r.disposition.value == "observe" and any("learned +" in x for x in r.reasons), str(r.reasons))
    r = g.decide(offer_ev("stuck_help", strength=0.9), d, user_busy=0.6, now=T0 + 7200)
    check("...but a strong offer of another kind still goes through", r.disposition.value == "respond",
          f"{r.score:.2f} vs {r.threshold:.2f}")

    # Welcomed: it speaks more readily.
    lr = learner()
    g = gate(lr)
    weak = offer_ev(strength=0.4)
    before = g.decide(weak, d, user_busy=0.6, now=T0).disposition.value
    for _ in range(12):
        lr.update("free_moment", CTX, 1.0)
    after = gate(lr).decide(weak, d, user_busy=0.6, now=T0 + 10).disposition.value
    check("a kind that is welcomed gets a lower bar", before == "observe" and after == "respond",
          f"{before} -> {after}")

    # Hard stops stay hard.
    lr = learner()
    for _ in range(12):
        lr.update("free_moment", CTX, 1.0)
    g = gate(lr)
    r = g.decide(offer_ev(strength=0.95), d, user_busy=1.0, now=T0)
    check("never an offer over someone typing, however welcome", r.disposition.value == "observe"
          and "typing" in r.reasons[-1], str(r.reasons))

    g = gate(lr, max_offers_per_hour=3)
    got = [g.decide(offer_ev(strength=0.9), d, user_busy=0.6, now=T0 + 600 * i).disposition.value
           for i in range(5)]
    check("an hourly cap on offers, whatever was learned", got[:3] == ["respond"] * 3 and got[3] == "observe",
          str(got))
    r = g.decide(offer_ev(strength=0.9), d, user_busy=0.6, now=T0 + 600 * 3 + 3700)
    check("...which rolls over", r.disposition.value == "respond")
    check("the cap shows in the status", g.status()["offer_budget"] == 3)

    g = gate(lr, quiet_hours=(0, 24))
    r = g.decide(offer_ev(strength=0.95), d, user_busy=0.6, now=T0)
    check("quiet hours still apply to offers", r.disposition.value == "observe" and "quiet" in r.reasons[-1])

    # Snooze.
    lr = learner(snooze_s=900)
    g = gate(lr)
    lr.snooze("free_moment", T0)
    r = g.decide(offer_ev(strength=0.95), d, user_busy=0.6, now=T0 + 10)
    check("'not now': held off", r.disposition.value == "observe" and "told to stop" in r.reasons[0])
    r = g.decide(offer_ev(strength=0.95), d, user_busy=0.6, now=T0 + 1000)
    check("...until the snooze ends", r.disposition.value == "respond")

    # Emergencies and the user are untouchable.
    lr = learner()
    for _ in range(40):
        lr.update("system", CTX, 0.0)
    lr.snooze("system", T0, 99999)
    g = gate(lr)
    r = g.decide(sys_ev(0.95), d, user_busy=0.6, now=T0)
    plain = gate().decide(sys_ev(0.95), d, user_busy=0.6, now=T0)
    check("a real emergency gets through whatever was learned or snoozed",
          r.disposition.value == "respond" and abs(r.threshold - plain.threshold) < 0.02,
          f"{r.threshold:.3f} vs {plain.threshold:.3f}")
    r = g.decide(SensorEvent(source="mic", text="ash what time is it", addressed=True), d, user_busy=1.0, now=T0)
    check("being spoken to is never gated", r.disposition.value == "respond" and r.addressed)

    # Ordinary remarks learn too, by source.
    lr = learner()
    for _ in range(8):
        lr.update("window", CTX, 0.0)
    ev = SensorEvent(source="window", text="you switched windows", salience=0.65, urgency=0.1)
    plain = gate().decide(ev, d, user_busy=0.6, now=T0)
    learned = gate(lr).decide(ev, d, user_busy=0.6, now=T0)
    check("a source whose remarks were unwelcome needs more to speak",
          learned.threshold > plain.threshold + 0.05, f"{plain.threshold:.2f} -> {learned.threshold:.2f}")


# ----------------------------------------------------------------------
class Pathway:
    value = "fast"


class Resp:
    def __init__(self, text):
        self.text = text
        self.pathway = Pathway()
        self.trace = None


class Homeo:
    drives = Drives()


class Brain:
    homeostasis = Homeo()
    follower = None

    def observe(self, **kw):
        class T:
            percepts = {"novelty": 0.5}
        return T()

    def status(self):
        return {}

    def sleep(self, **kw):
        return {}


class Ash:
    def __init__(self):
        self.brain = Brain()
        self.episodic_memory = None
        self.calls = []

    def think(self, query, image=None, context=""):
        self.calls.append((query, context))
        return Resp("Hey, good stretch of work. How's it going?" if "FREE MOMENT" in query
                    else "Sure.")


class FakeIdle:
    name = "idle"
    value = 0.0

    def due(self, now):
        return False

    def seconds_idle(self):
        return self.value


def rig(**cfg):
    clock = [T0]
    delivered = []
    ash = Ash()
    per = PeripheryRegistry()
    idle = FakeIdle()
    per.sensors["idle"] = idle
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()),
        gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, **cfg),
        response_sink=lambda t, m: delivered.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock = rt, ash, delivered, idle, clock

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def work(seconds, every=10):
        for _ in range(int(seconds / every)):
            step(every, 0.0, ticks=1)

    def say(text, addressed=True, **kw):
        rt.submit(SensorEvent(source="mic" if addressed else kw.pop("source", "screen"),
                              text=text, addressed=addressed, **kw))
        rt._tick()

    h.step, h.work, h.say = step, work, say
    h.offers = lambda: [o for o in delivered if not o[1]["addressed"]]
    return h


def free_moment(h, work_s=150):
    """A real stretch of work, then a pause. Returns the offers it produced."""
    n = len(h.offers())
    h.work(work_s)
    h.step(15, 25.0)
    return h.offers()[n:]


def s7_runtime():
    print("\n=== S7. The runtime, end to end ===")
    h = rig()
    got = free_moment(h)
    check("a pause after real work: one chat line is delivered", len(got) == 1, str(h.out))
    check("it came from the model, framed as a free moment, not an answer",
          "FREE MOMENT" in h.ash.calls[-1][0] and got[0][1]["attention"]["kind"] == "free_moment")
    check("and is being watched for a reaction", len(h.rt.reactions.pending) == 1)
    check("nothing more while they stay idle", len(h.offers()) == 1 and (h.step(30, 40) or True)
          and len(h.offers()) == 1)

    n_calls = len(h.ash.calls)
    h.say("yes please, how are you")
    check("their reply is read as welcome", h.rt.reactions.counts["accepted"] == 1
          or h.rt.reactions.counts["engaged"] == 1, str(h.rt.reactions.counts))
    check("and the learner has a better opinion of free moments",
          h.rt.learner.p("free_moment", CTX) > 0.5, f"{h.rt.learner.p('free_moment', CTX):.2f}")
    check("the reply was answered by the model, told what it was replying to",
          len(h.ash.calls) == n_calls + 1 and "checked in" in h.ash.calls[-1][1], str(h.ash.calls[-1]))

    # "not now"
    h = rig()
    free_moment(h)
    n_calls = len(h.ash.calls)
    h.say("not now")
    check("'not now' is acknowledged without running the brain",
          h.out[-1][0] == "Okay, I'll check back later." and len(h.ash.calls) == n_calls, str(h.out[-1]))
    check("it is counted as a deferral and that kind is snoozed",
          h.rt.reactions.counts["deferred"] == 1 and h.rt.learner.snoozed("free_moment", h.clock[0] + 1))
    h.step(100, 600)
    got = free_moment(h)
    check("a second free moment inside the snooze stays quiet", got == [])
    h.step(2000, 600)
    check("the snooze has ended", not h.rt.learner.snoozed("free_moment"))
    got = free_moment(h, work_s=900)
    check("after it, a long stretch of work (a stronger case) gets a chat again", len(got) == 1, f"{len(got)}")

    h = rig()
    free_moment(h)
    n_calls = len(h.ash.calls)
    h.say("be quiet")
    check("'be quiet' is acknowledged too, and held for much longer",
          h.out[-1][0] == "Okay, I'll stay out of your way." and len(h.ash.calls) == n_calls
          and h.rt.learner.snoozed("free_moment", h.clock[0] + 1.5 * HOUR))
    h.step(3 * HOUR, 600)
    check("...but not forever", not h.rt.learner.snoozed("free_moment"))

    # silence
    h = rig(reaction_window_s=60)
    free_moment(h)
    h.step(90, 40)
    check("no reply within the window is recorded as silence",
          h.rt.reactions.counts["silence"] == 1 and not h.rt.reactions.pending)
    check("...and nudges the estimate down a little",
          0.3 < h.rt.learner.p("free_moment", context_key(0.6, h.clock[0] - 100)) < 0.5)

    # dismissal
    h = rig()
    free_moment(h)
    h.say("", addressed=False, source="notification", meta={"dismissed": True})
    check("a dismissed notification is a cost", h.rt.reactions.counts["dismissed"] == 1)

    # a long message after an offer is a new request, not a verdict
    h = rig()
    free_moment(h)
    h.say("no I was actually hoping you could explain how the planner chooses between tools in detail")
    check("a long reply is engagement, not a rejection",
          h.rt.reactions.counts["engaged"] == 1 and h.rt.reactions.counts["rejected"] == 0
          and not h.rt.learner.snoozed("free_moment", h.clock[0]))

    # a reply with nothing pending
    h = rig()
    n_calls = len(h.ash.calls)
    h.say("no")
    check("a bare 'no' with nothing pending is just handled normally",
          len(h.ash.calls) == n_calls + 1 and h.rt.reactions.counts["rejected"] == 0)

    # stuck
    h = rig()
    h.step(10, 0.0)
    for i in range(5):
        h.say(f"Traceback: ValueError: bad input at line {20 + i}", addressed=False, source="term")
        h.step(5, 2.0, ticks=1)
    n_calls = len(h.ash.calls)
    h.step(5, 8.0)
    got = h.offers()
    check("repeated errors: it offers help, in so many words",
          any("keeps coming back" in o[0] and "?" in o[0] for o in got), str([o[0] for o in got]))
    check("...without asking the model (a plain, checkable sentence)", len(h.ash.calls) == n_calls
          and all(o[1]["pathway"] == "offer" for o in got if "keeps coming back" in o[0]))
    check("...as a stuck_help offer", got[-1][1]["attention"]["kind"] == "stuck_help")
    h.say("yes")
    check("'yes' to it is accepted and the model is told what the offer was about",
          h.rt.reactions.counts["accepted"] == 1 and "offered help because" in h.ash.calls[-1][1],
          str(h.ash.calls[-1]))

    h = rig()
    h.step(10, 0.0)
    for i in range(5):
        h.say(f"Traceback: ValueError: bad input at line {20 + i}", addressed=False, source="term")
    h.step(5, 1.0)
    check("it does not offer help while they are typing",
          not any("keeps coming back" in o[0] for o in h.offers()))
    h.step(5, 8.0)
    check("...it waits for the pause", any("keeps coming back" in o[0] for o in h.offers()))

    # a volunteered ordinary remark is learned from too
    h = rig()
    h.step(10, 40.0)
    h.say("battery at 3%", addressed=False, source="system", salience=0.9, urgency=0.8)
    spoke = [o for o in h.out if not o[1]["addressed"]]
    check("an urgent event is spoken about", len(spoke) == 1, str(h.out))
    h.say("thanks")
    check("and the thanks is learned from, under the event's source",
          h.rt.learner.p("system", context_key(0.25, h.clock[0])) > 0.5)

    # ----- the simulated users
    def run_persona(welcomes, trials=12):
        h = rig()
        offered = []
        for i in range(trials):
            h.step(2 * HOUR, 600.0)                       # hours pass, they are away
            got = free_moment(h)
            offered.append(bool(got))
            if got:
                if welcomes:
                    h.say("yes thanks")
                else:
                    h.say("not now" if i % 2 else "", addressed=bool(i % 2), source="notification",
                          meta={"dismissed": True})
            h.step(1, 600.0)
        return offered, h

    off, h = run_persona(welcomes=False)
    check("a user who always turns it down: it first tries, then stops",
          off[0] and sum(off) <= 4 and not any(off[6:]), str(off))
    off2, h2 = run_persona(welcomes=True)
    check("a user who welcomes it: it keeps offering", sum(off2) >= 11, str(off2))
    check("the two records differ in the right direction",
          h2.rt.learner.p("free_moment") > 0.7 > 0.3 > h.rt.learner.p("free_moment"),
          f"{h2.rt.learner.p('free_moment'):.2f} vs {h.rt.learner.p('free_moment'):.2f}")

    # ... and forgiveness: a day or two later it tries again.
    h.step(3 * DAY, 600.0)
    again = free_moment(h)
    check("a few days after being turned down it is willing to try once more", len(again) == 1)

    # Different people, different situations: unwelcome while busy, welcome when calm.
    h = rig()
    lr = h.rt.learner
    for _ in range(10):
        lr.update("free_moment", context_key(0.6, T0), 0.0)      # active
        lr.update("free_moment", context_key(0.25, T0), 1.0)     # idle
    check("per situation: held back when 'active', free when 'idle'",
          lr.bar_shift("free_moment", context_key(0.6, T0)) > 0.1
          and lr.bar_shift("free_moment", context_key(0.25, T0)) < -0.1)


def s8_switches():
    print("\n=== S8. Switches, persistence, status ===")
    h = rig(learn_social=False)
    check("learning off: no learner, nothing attached to the gate",
          h.rt.learner is None and h.rt.reactions is None and h.rt.gate.learner is None)
    got = free_moment(h)
    check("...offers still work, without learning", len(got) == 1)
    h.say("not now")
    check("...a reply does not crash it, and is answered", h.out[-1][0] is not None and h.rt.stats["errors"] == 0)

    h = rig(flow_offers=False)
    check("offers off: no detector", h.rt.flow is None)
    check("...so a pause after work says nothing", free_moment(h) == [])
    check("...and nothing crashes", h.rt.stats["errors"] == 0)

    h = rig()
    h.rt.periphery.sensors.pop("idle")
    h.rt._tick()
    check("no idle sensor: it cannot tell when they pause, so it says nothing",
          h.rt._idle_seconds() is None and not h.offers() and h.rt.stats["errors"] == 0)

    path = os.path.join(tempfile.mkdtemp(), "social.json")
    h = rig(social_path=path)
    free_moment(h)
    h.say("yes please")
    p = h.rt.learner.p("free_moment", CTX)
    again = rig(social_path=path)
    check("what it learned is there after a restart", abs(again.rt.learner.p("free_moment", CTX) - p) < 0.05
          and again.rt.learner.updates >= 1, f"{p:.3f}")

    h = rig()
    free_moment(h)
    soc = {"learner": h.rt.learner.status(), "reactions": h.rt.reactions.status(), "flow": h.rt.flow.status()}
    check("status exposes what it is learning",
          soc["reactions"]["pending"] == 1 and soc["flow"]["emitted"]["free_moment"] == 1
          and soc["learner"]["updates"] == 0)
    check("offers are counted", h.rt.stats["offers"] == 1)

    # A crashing detector must not take the loop down.
    h = rig()

    def boom(*a, **k):
        raise RuntimeError("detector on fire")
    h.rt.flow.poll = boom
    h.step(5, 0.0)
    check("a detector that raises is counted and survived", h.rt.stats["errors"] >= 1)


def main():
    s1_replies()
    s2_context()
    s3_learner()
    s4_tracker()
    s5_flow()
    s6_gate()
    s7_runtime()
    s8_switches()
    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
