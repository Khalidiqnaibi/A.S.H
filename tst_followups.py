"""
tst_followups.py -- asking how something went, after it happened.

  U1  Reading a time: today, tonight, tomorrow, weekdays, "next", dates,
      "in two days", clock times, and what is NOT a time
  U2  Spotting an event: big and routine ones, trips, seeing someone, and the
      phrases that only look like events ("the test suite", "the date format")
  U3  Keeping it: dedupe, reschedule, cancel, told-before-asked, caps, expiry
  U4  When to ask: after it is over, never in the night, the important first
  U5  Reading the answer: good, bad, so-so, not yet, did not happen, noise
  U6  Persistence: restart keeps everything, a broken file is set aside
  U7  As a free-moment activity: proposed only when due, asked once, answered
  U8  The runtime end to end
  U9  Fuzz: nothing anyone says crashes it

Run:  PYTHONIOENCODING=utf-8 python tst_followups.py
"""

import json
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.followups import (FollowUpActivity, FollowUps, FollowUpsConfig,  # noqa: E402
                                  detect_events, resolve_when)
from src.daemon.freemoment import FreeMoments, FreeMomentsConfig, Joke  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402
from src.senses.base import PeripheryRegistry  # noqa: E402

check = tw.check
T0 = tw.T0                       # Tuesday 10 March 2026, 14:00 local
HOUR, DAY = 3600.0, 86400.0
FREE = dict(idle_s=60.0, free_moment=True)


def lt(ts):
    return time.localtime(ts)


def at(days, hour, minute=0):
    t = lt(T0)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday + days, hour, minute, 0, 0, 0, -1))


def when(text, now=T0):
    w = resolve_when(text, now)
    return None if w is None else w.start


def fus(**cfg):
    t = [T0]
    f = FollowUps(FollowUpsConfig.from_dict(cfg), path=None, clock=lambda: t[0], sentiment=tw.fake_sentiment)
    f.t = t
    return f


def opened(f):
    return sorted((x.what, x.status) for x in f.items.values() if x.status in ("pending", "asked"))


# ---------------------------------------------------------------------------
def u1_time():
    print("\n=== U1. Reading a time ===")
    check("(the fixture is a Tuesday afternoon)", lt(T0).tm_wday == 1 and lt(T0).tm_hour == 14)
    check("tomorrow -> tomorrow, no clock", when("my exam is tomorrow") == at(1, 12)
          and not resolve_when("tomorrow", T0).clock)
    check("tomorrow at 3pm", when("tomorrow at 3pm") == at(1, 15))
    check("tomorrow morning", when("tomorrow morning") == at(1, 9))
    check("the day after tomorrow", when("the day after tomorrow") == at(2, 12))
    check("today at 5 is 17:00 today", when("today at 5") == at(0, 17))
    check("at 4:30 with no day is today if still ahead", when("at 4:30") == at(0, 16, 30))
    check("at 1 pm with no day, already past, is tomorrow", when("at 1 pm") == at(1, 13))
    check("at 9am with no day, past, is tomorrow", when("at 9am") == at(1, 9))
    check("tonight", when("tonight") == at(0, 20))
    check("at noon tomorrow", when("tomorrow at noon") == at(1, 12))
    check("on Thursday is in two days", when("on Thursday") == at(2, 12))
    check("a bare weekday name works", when("I see them friday") == at(3, 12))
    check("short weekday", when("on thu") == at(2, 12))
    check("Monday is the coming one", when("on Monday") == at(6, 12))
    check("next Thursday (said on a Tuesday) is the week after", when("next Thursday") == at(9, 12))
    t_sat = at(4, 11)
    check("next Monday said on a Saturday is the coming Monday",
          lt(resolve_when("next Monday", t_sat).start).tm_mday == lt(at(6, 12)).tm_mday)
    check("Tuesday said on a Tuesday is today", when("on Tuesday") == at(0, 14) or when("on Tuesday") >= T0)
    check("last Thursday is not a future time", when("last Thursday") is None)
    check("12 March", when("on 12 March") == at(2, 12))
    check("March 12th", when("March 12th") == at(2, 12))
    check("a date already gone this year is next year",
          lt(when("on 1 March")).tm_year == lt(T0).tm_year + 1)
    check("on the 20th", when("on the 20th") == at(10, 12))
    check("in two days", when("in two days") == at(2, 12))
    check("in 3 hours is exact, with a clock", when("in 3 hours") == T0 + 3 * HOUR
          and resolve_when("in 3 hours", T0).clock)
    check("in a week", when("in a week") == at(7, 12))
    check("next week is the Monday after", lt(when("next week")).tm_wday == 0 and when("next week") > T0)
    check("this weekend is Saturday", lt(when("this weekend")).tm_wday == 5)
    check("nothing that is a time -> None", when("I have an exam") is None
          and when("the meeting was fine") is None and when("") is None)
    check("'today' said in the evening is still ahead",
          resolve_when("my exam is today", at(0, 19)).start >= at(0, 19))
    check("a time and a weekday combine", when("on Friday at 10am") == at(3, 10))


def u2_detect():
    print("\n=== U2. Spotting an event ===")

    def nouns(s, routine=True):
        return [(e.what, e.important) for e in detect_events(s, T0, routine)]

    check("my interview", nouns("I have an interview tomorrow") == [("your interview", True)])
    check("my job interview", nouns("my job interview is on Thursday") == [("your job interview", True)])
    check("a modifier is kept, 'big' is not", nouns("I have a big presentation tomorrow")
          == [("your presentation", True)], str(nouns("I have a big presentation tomorrow")))
    check("my math exam", nouns("my math exam is on Friday")[0][0] == "your math exam")
    check("the dentist", nouns("going to the dentist tomorrow") == [("the dentist", True)])
    check("a meeting is routine", nouns("I have a meeting at 3") == [("your meeting", False)])
    check("an important meeting is not", nouns("I have an important meeting at 3") == [("your meeting", True)])
    check("routine off drops meetings", nouns("I have a meeting at 3", routine=False) == [])
    check("a trip", nouns("I'm flying to Paris on Friday") == [("your trip to Paris", True)])
    check("seeing someone", nouns("I'm seeing Sara tomorrow") == [("seeing Sara", False)])
    check("meeting with someone", nouns("meeting with Omar at 4") == [("seeing Omar", False)]
          or ("seeing Omar", False) in nouns("meeting with Omar at 4"), str(nouns("meeting with Omar at 4")))
    check("not an event: the test suite", nouns("run the test suite tomorrow") == [])
    check("not an event: the date format", nouns("change the date format on Monday") == [])
    check("not an event: the date on Friday", nouns("what is the date on Friday") == [])
    check("but my date is", nouns("I have a date on Friday") == [("your date", True)])
    check("not an event: a call to the API at 3?", all(w != "your call" for w, _ in nouns("make a call to the API")))
    check("a weekday is not a place", nouns("I'm going to Monday's class") == [] or
          all("trip" not in w for w, _ in nouns("I'm going to Monday's class")))
    e = detect_events("the interview was cancelled", T0)[0]
    check("cancellation is read", e.cancelled)
    e = detect_events("my exam got moved to Friday", T0)[0]
    check("a move is read, with its new time", e.moved and e.when and e.when.start == at(3, 12))
    e = detect_events("the interview went great yesterday", T0)[0]
    check("past tense is read", e.past)
    check("two events in one sentence", len(detect_events("I have an exam and a meeting tomorrow", T0)) == 2)


def u3_keep():
    print("\n=== U3. Keeping it ===")
    f = fus()
    f.observe("I have a job interview on Thursday at 10am")
    check("noted", opened(f) == [("your job interview", "pending")])
    f.observe("my job interview is on Thursday at 10am, wish me luck")
    check("the same event said again is not a second one", len(f.items) == 1)
    f.observe("my job interview got moved to Friday at 11am")
    x = list(f.items.values())[0]
    check("rescheduled, not duplicated", len(f.items) == 1 and x.start == at(3, 11) and x.reschedules == 1)
    f.observe("the job interview was cancelled")
    check("cancelled", x.status == "cancelled" and opened(f) == [])

    f = fus()
    f.observe("I have an exam tomorrow")
    f.t[0] = at(1, 20)
    learned = f.observe("the exam went great, I think I aced it")
    x = list(f.items.values())[0]
    check("told before asked: closed, nothing to ask", x.status == "told" and f.due_now() is None)
    check("with the outcome learned", learned == ["The user's exam went well."], str(learned))

    f = fus()
    f.observe("my exam was hard")
    check("a past event with nothing open notes nothing", f.items == {})
    f.observe("the interview is in 90 days")
    check("too far ahead is a plan, not a follow-up", f.items == {})
    f.observe("I had a meeting yesterday")
    check("a past one is not noted", f.items == {})
    f.observe("I have an exam")
    check("no time, no follow-up", f.items == {})

    f = fus(max_open=3)
    for thing in ("a meeting on Monday", "a call on Wednesday", "a class on Thursday", "a lecture on Friday"):
        f.observe(f"I have {thing}")
    check("(four routine events were offered)", f.stats["noted"] == 4, str(f.stats))
    f.observe("my big exam is tomorrow")
    check("capped: open never exceeds max_open", len(opened(f)) <= 3, str(opened(f)))
    check("and the important one is kept over a routine one",
          any(w == "your exam" for w, _ in opened(f)))

    f = fus()
    f.observe("I have a presentation tomorrow at 9am")
    f.t[0] = at(5, 12)
    f.housekeeping()
    check("unasked for longer than the window: expired", list(f.items.values())[0].status == "expired")

    f = fus(enabled=False)
    f.observe("I have an interview tomorrow")
    check("off: nothing noted", f.items == {})


def u4_when():
    print("\n=== U4. When to ask ===")
    f = fus()
    f.observe("I have an interview tomorrow at 10am")
    x = list(f.items.values())[0]
    check("with a clock: due two hours after it starts", x.due == at(1, 12))
    f.t[0] = at(1, 11)
    check("not before", f.due_now() is None)
    f.t[0] = at(1, 12, 1)
    check("then due", f.due_now() is x)

    f = fus()
    f.observe("my exam is on Thursday")
    check("no clock: the evening of the day", list(f.items.values())[0].due == at(2, 18))
    f = fus()
    f.observe("I have a party tonight")
    check("a late event is asked the next morning, not in the night",
          list(f.items.values())[0].due == at(1, 9), time.ctime(list(f.items.values())[0].due))
    f = fus()
    f.observe("I have a meeting at 9pm")
    check("a 9pm meeting is due 11pm -> next morning 9", list(f.items.values())[0].due == at(1, 9))

    f = fus()
    f.observe("I have a meeting tomorrow at 9am")
    f.observe("my driving test is tomorrow at 9am")
    f.t[0] = at(1, 13)
    check("two due: the important one first", f.due_now().what == "your driving test")
    v_new = f.value(f.due_now(), at(1, 13))
    v_old = f.value(f.due_now(), at(3, 13))
    check("value fades over the window but not below a floor", v_old < v_new and v_old >= 0.8 * 0.6 - 1e-9)


def u5_answers():
    print("\n=== U5. Reading the answer ===")

    def ans(text, important=True):
        f = fus()
        f.observe("I have an " + ("interview" if important else "online meeting") + " tomorrow at 10am")
        f.t[0] = at(1, 13)
        x = f.due_now()
        f.mark_asked(x.id)
        return f.hear(x.id, text), x

    r, x = ans("it went really well!")
    check("good", r[0] == "That's great to hear!" and x.outcome == "good" and x.status == "answered", str(r))
    check("and learned", r[1] == "The user's interview went well.")
    r, x = ans("honestly it was awful")
    check("bad", "sorry" in r[0].lower() and x.outcome == "bad" and r[1].endswith("went badly."))
    r, x = ans("I nailed it")
    check("'nailed it' is good with no mood word", x.outcome == "good", str(r))
    r, x = ans("I didn't get the job")
    check("'didn't get' is bad", x.outcome == "bad", str(r))
    r, x = ans("so-so")
    check("so-so is mixed", x.outcome == "mixed" and r[0].startswith("Okay"))
    r, x = ans("still waiting to hear back")
    check("not yet", x.outcome == "not_yet" and r[1] == "")
    r, x = ans("it got cancelled actually")
    check("did not happen", x.outcome == "didnt_happen")
    r, x = ans("ok")
    check("noise is not an answer", r is None and x.status == "asked")
    r, x = ans("can you open my email")
    check("a request is not an answer", r is None)
    r, x = ans("went fine", important=False)
    check("a routine event gets a plainer reply", r[0] == "Good to hear.", str(r))


def u6_persist():
    print("\n=== U6. Persistence ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "fu.json")
    t = [T0]
    f = FollowUps(FollowUpsConfig(), path=p, clock=lambda: t[0])
    f.observe("I have an interview tomorrow at 10am. My exam is on Friday.")
    f.observe("the exam got cancelled")
    f2 = FollowUps(FollowUpsConfig(), path=p, clock=lambda: t[0])
    check("a restart keeps the open ones and the closed",
          sorted((x.what, x.status) for x in f2.items.values())
          == [("your exam", "cancelled"), ("your interview", "pending")])
    check("and the stats", f2.stats["noted"] == 2 and f2.stats["cancelled"] == 1)
    with open(p, "w") as fh:
        fh.write("{not json")
    f3 = FollowUps(FollowUpsConfig(), path=p, clock=lambda: t[0])
    check("a broken file is set aside, not fatal", f3.items == {} and os.path.exists(p + ".bad"))
    f3.observe("I have a demo tomorrow")
    saved = open(p, encoding="utf-8").read()
    check("the file keeps the event phrase, not the sentence", "demo" in saved and "I have" not in saved)


def u7_activity():
    print("\n=== U7. As a free-moment activity ===")
    t = [T0]
    f = FollowUps(FollowUpsConfig(), clock=lambda: t[0], sentiment=tw.fake_sentiment)
    m = FreeMoments(FreeMomentsConfig(min_gap_s=0, max_per_hour=9), clock=lambda: t[0])
    learned, moods = [], []
    m.register(FollowUpActivity(f, learn=learned.append, mood=moods.append))
    m.register(Joke())
    f.observe("I have an interview tomorrow at 10am")
    c = m.pick(Environment(**FREE), T0)
    check("not due: a joke wins", c is not None and c.activity == "joke")
    t[0] = at(1, 13)
    c = m.pick(Environment(**FREE), t[0])
    check("due: the follow-up beats a joke", c.activity == "follow_up" and c.text == "How did your interview go?")
    check("proposing does not mark it asked", list(f.items.values())[0].status == "pending")
    m.mark_spoken("follow_up", t[0])
    check("being said marks it asked", list(f.items.values())[0].status == "asked")
    r = m.hear("it was rough, I don't think I got it", t[0] + 20)
    check("the answer is acknowledged", r and "sorry" in r.lower(), str(r))
    check("learned", learned == ["The user's interview went badly."])
    check("and the mood is set gently low", moods == [-0.4])
    t[0] += 3 * HOUR
    c = m.pick(Environment(**FREE), t[0])
    check("never asked twice", c is None or c.activity != "follow_up")
    check("a different utterance after the window is not taken", m.hear("hello", t[0] + 999) is None)


def rt_rig(**fc):
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()),
        gate=AttentionGate(max_offers_per_hour=20), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             wonder=True, wonder_cfg={"min_gap_s": 0},
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json"),
                             moments=True, moments_cfg={"min_gap_s": 0, "mood_check": False, "jokes": False},
                             followups=True, followups_cfg=fc,
                             followups_path=os.path.join(d, "fu.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock, h.dir = rt, ash, out, idle, clock, d

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True):
        rt.submit(tw.SensorEvent(source=source, modality=tw.Modality.TEXT, text=text, addressed=addressed,
                                 salience=0.9, urgency=0.6))
        rt._tick()

    def free():
        for _ in range(15):
            step(10, 0.0, ticks=1)
        step(15, 25.0)

    h.step, h.say, h.free = step, say, free
    h.said = lambda: [o[0] for o in out]
    return h


def u8_runtime():
    print("\n=== U8. The runtime ===")
    h = rt_rig()
    h.say("I have a job interview tomorrow at 10am")
    check("the sentence is still answered normally", h.ash.calls == ["I have a job interview tomorrow at 10am"])
    check("and the event is noted", h.rt.followups.status()["open"] == 1)
    h.free()
    check("before it happened, nothing is asked about it",
          not any("interview" in s for s in h.said()[1:]), str(h.said()))
    h.clock[0] = at(1, 13)
    h.free()
    check("after it, at a free moment, ASH asks", h.said()[-1] == "How did your job interview go?", str(h.said()))
    check("without the model", len(h.ash.calls) == 1)
    n = len(h.ash.calls)
    h.step(5, 5.0)
    h.say("it went really well actually", source="microphone", addressed=False)
    check("the answer is spent on the follow-up, not sent to the model",
          len(h.ash.calls) == n and h.said()[-1] == "That's great to hear!", str(h.said()))
    check("the concept network was told",
          any(s.get("source") == "wonder" and "went well" in s["text"] for s in h.ash.brain.seen))
    h.step(3 * HOUR, 0.0)
    h.free()
    check("and it is not asked again", sum("interview" in s for s in h.said()) == 1, str(h.said()))
    check("status carries followups", h.rt.status()["followups"]["stats"]["answered"] == 1)

    h = rt_rig()
    h.say("my exam is tomorrow")
    h.clock[0] = at(1, 21)
    h.say("the exam went great by the way")
    h.clock[0] = at(2, 10)
    h.free()
    check("told first: never asked", not any("exam" in s for s in h.said()[2:]), str(h.said()))
    check("and what was told reached the concept network",
          any(s.get("source") == "wonder" and "exam went well" in s["text"] for s in h.ash.brain.seen))

    h = rt_rig(enabled=False)
    h.say("I have an interview tomorrow at 10am")
    h.clock[0] = at(1, 13)
    h.free()
    check("switched off: nothing noted, nothing asked", h.rt.followups.status()["open"] == 0
          and not any("interview" in s for s in h.said()[1:]))

    h = rt_rig()
    h.rt.moments.cfg.disabled = ("follow_up",)
    h.say("I have an interview tomorrow at 10am")
    h.clock[0] = at(1, 13)
    h.free()
    check("disabled as an activity: not asked", not any("interview" in s for s in h.said()[1:]))

    h = rt_rig()
    h.say("I have an interview tomorrow at 10am", source="screen", addressed=False)
    check("text on the screen is not the user saying it", h.rt.followups.status()["open"] == 0)


def u9_fuzz():
    print("\n=== U9. Fuzz ===")
    rng = random.Random(7)
    words = ("I have my the a an interview exam meeting tomorrow today tonight next on at in "
             "Monday Friday 3pm 10:30 12 March the 20th two days hours was cancelled moved went great "
             "awful Sara Paris flying to seeing dentist ? ! , . big date test suite last 99 -5 am pm").split()
    f = fus()
    bad = 0
    for i in range(3000):
        s = " ".join(rng.choice(words) for _ in range(rng.randint(0, 14)))
        try:
            w = resolve_when(s, T0 + rng.uniform(-DAY, 30 * DAY))
            if w is not None and not (abs(w.start - T0) < 400 * DAY):
                bad += 1
            f.t[0] = T0 + i * 600
            f.observe(s)
            x = f.due_now()
            if x:
                f.mark_asked(x.id)
                f.hear(x.id, rng.choice(words))
        except Exception as e:
            bad += 1
            print("   crash on", repr(s), e)
    check("3000 random utterances: no crash, every time sane", bad == 0, str(bad))
    check("open never exceeds the cap", sum(1 for x in f.items.values() if x.status in ("pending", "asked"))
          <= f.cfg.max_open)
    for junk in (None, 5, b"x", "", " " * 10000, "\x00\x01", "a" * 100000):
        try:
            f.observe(junk)
            f.hear("nope", junk)
        except Exception as e:
            check(f"junk {type(junk).__name__} is harmless", False, str(e))
            break
    else:
        check("junk input is harmless", True)


if __name__ == "__main__":
    for fn in (u1_time, u2_detect, u3_keep, u4_when, u5_answers, u6_persist, u7_activity, u8_runtime, u9_fuzz):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
