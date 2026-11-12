"""
tst_freemoment.py -- what ASH does with a free moment, and how it chooses.

  F1  The registry: scoring, cool-downs, budget, every quiet rule, fits
  F2  The built-ins: joke, mood check (reads an answer), gap closure
  F3  Extending: a one-function activity, a failing activity, a custom fit
  F4  The runtime end to end: a free moment picks among several things,
      a mood answer changes what comes next, nothing is said when it should not be

Run:  PYTHONIOENCODING=utf-8 python tests/tst_freemoment.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import (AmbientConfig, AmbientRuntime, AttentionGate, Journal,  # noqa: E402
                        PrivacyPolicy)
from src.daemon.freemoment import (Activity, Candidate, FreeMoments, FreeMomentsConfig,  # noqa: E402
                                   GapClosure, Joke, MoodCheck, ProviderActivity, default_moments,
                                   mood_of, read_with)
from src.daemon.wondering import Environment  # noqa: E402
from src.senses.base import PeripheryRegistry  # noqa: E402

check = tw.check
T0, HOUR = tw.T0, tw.HOUR
FREE = dict(idle_s=60.0, free_moment=True)


class Fixed(Activity):
    def __init__(self, name, text, value, weight=1.0, cooldown_s=0.0):
        self.name, self.text, self.value = name, text, value
        self.weight, self.cooldown_s = weight, cooldown_s

    def propose(self, env, now):
        return Candidate(self.name, self.text, self.value)


def fm(**cfg):
    t = [T0]
    kw = dict(min_gap_s=0.0)
    kw.update(cfg)
    m = FreeMoments(FreeMomentsConfig.from_dict(kw), clock=lambda: t[0], sentiment=tw.fake_sentiment)
    m.t = t
    return m


def f1_registry():
    print("\n=== F1. The registry ===")
    m = fm()
    m.register(Fixed("a", "A", 0.3))
    m.register(Fixed("b", "B", 0.6))
    check("the highest value x weight wins", m.pick(Environment(**FREE)).text == "B")
    m = fm()
    m.register(Fixed("a", "A", 0.5, weight=1.0))
    m.register(Fixed("b", "B", 0.6, weight=0.5))
    check("weight is a standing preference", m.pick(Environment(**FREE)).text == "A")
    m = fm()
    m.register(Fixed("lo", "x", 0.05))
    check("below the minimum score, silence beats the remark", m.pick(Environment(**FREE)) is None)

    for label, env in (("not a free moment", dict(idle_s=60.0)), ("idle unknown", dict(free_moment=True)),
                       ("muted", dict(muted=True)), ("quiet hours", dict(quiet_hours=True)),
                       ("privacy paused", dict(paused=True)), ("in conversation", dict(conversation_open=True)),
                       ("a plan running", dict(plan_active=True)), ("speaking", dict(speaking=True))):
        m = fm()
        m.register(Fixed("a", "A", 0.9))
        e = dict(FREE)
        e.update(env)
        if label in ("not a free moment",):
            e = env
        if label == "idle unknown":
            e = env
        check(f"nothing when {label}", m.pick(Environment(**e)) is None)

    m = fm(min_gap_s=600.0, max_per_hour=2)
    m.register(Fixed("a", "A", 0.9))
    m.register(Fixed("b", "B", 0.8))
    c = m.pick(Environment(**FREE))
    m.mark_spoken(c.activity)
    m.t[0] += 150                      # past the answer window
    check("a minimum gap between free-moment remarks", m.pick(Environment(**FREE)) is None)
    m.t[0] += 500
    c = m.pick(Environment(**FREE))
    check("then the next one may go", c is not None)
    m.mark_spoken(c.activity)
    m.t[0] += 700
    check("the hourly budget holds", m.pick(Environment(**FREE)) is None)
    m.t[0] += HOUR
    check("and renews", m.pick(Environment(**FREE)) is not None)

    m = fm()
    m.register(Fixed("a", "A", 0.9, cooldown_s=3600.0))
    m.register(Fixed("b", "B", 0.4))
    m.mark_spoken("a")
    m.t[0] += 130
    check("an activity cool-down hands the moment to the next best", m.pick(Environment(**FREE)).text == "B")
    m.t[0] += 3600
    check("and it is back after", m.pick(Environment(**FREE)).text == "A")

    m = fm(answer_window_s=120.0)
    m.register(Fixed("a", "A", 0.9))
    m.mark_spoken("a")
    check("no second remark while an answer is awaited", m.pick(Environment(**FREE)) is None)

    m = fm(disabled=("a",))
    m.register(Fixed("a", "A", 0.9))
    m.register(Fixed("b", "B", 0.2))
    check("a disabled activity is never picked", m.pick(Environment(**FREE)).text == "B")
    m = fm(enabled=False)
    m.register(Fixed("a", "A", 0.9))
    check("the master switch silences everything", m.pick(Environment(**FREE)) is None)

    m = fm()
    m.register(Fixed("a", "A", 0.9))
    m.register(Fixed("b", "B", 0.5))
    m.add_fit(lambda c, e: 0.0 if c.activity == "a" else 1.0)
    check("a fit of 0 forbids", m.pick(Environment(**FREE)).text == "B")
    m.add_fit(lambda c, e: 2.0 if c.activity == "b" else 1.0)
    m.fits = None
    check("fits multiply", m.pick(Environment(**FREE)).score == 1.0)


def f2_builtins():
    print("\n=== F2. The built-ins ===")
    check("the model's positive is a good mood, scaled by how sure it is",
          abs(mood_of({"sentiment": "POSITIVE", "confidence": 0.99}) - 0.98) < 1e-9
          and abs(mood_of({"sentiment": "POSITIVE", "confidence": 0.75}) - 0.5) < 1e-9)
    check("its negative is a low one", mood_of({"sentiment": "NEGATIVE", "confidence": 0.95}) < -0.8)
    check("too unsure to count", mood_of({"sentiment": "NEGATIVE", "confidence": 0.55}) is None)
    check("a three-way model's neutral is 0", mood_of({"sentiment": "NEUTRAL", "confidence": 0.9}) == 0.0)
    check("an unknown label, or nothing, is nothing",
          mood_of({"sentiment": "unknown", "confidence": 0.0}) is None and mood_of(None) is None
          and mood_of({"sentiment": "POSITIVE", "confidence": "x"}) is None)
    check("the pipeline's own shape is read too", mood_of({"label": "NEGATIVE", "score": 0.9}) < 0)
    check("read through the model", (read_with(tw.fake_sentiment, "pretty good thanks") or 0) > 0.3
          and (read_with(tw.fake_sentiment, "tired and a bit stressed") or 0) < -0.3)
    check("no model: nothing is read", read_with(None, "pretty good thanks") is None)
    check("a failing model is not fatal", read_with(lambda t: 1 / 0, "good") is None)
    check("junk is nothing", read_with(tw.fake_sentiment, None) is None and read_with(tw.fake_sentiment, "") is None)
    m = fm()
    m.note_mood(0.8)
    m.note_mood(-0.8)
    check("a single reading moves the mood, it does not replace it", abs(m.mood()) < 1e-9)
    m.note_mood(None)
    check("(nothing read: nothing changes)", abs(m.mood()) < 1e-9)

    m = fm()
    j = m.register(Joke())
    check("a joke is offered when the mood is unknown", m.pick(Environment(**FREE)).activity == "joke")
    check("never to someone who is low",
          m.pick(Environment(mood=-0.5, **FREE)) is None)
    first = j.propose(Environment(**FREE), T0).text
    j.hear("ha", T0)
    check("the next joke is a different one", j.propose(Environment(**FREE), T0).text != first)
    check("a joke ignores the reply (it is handled as any utterance)", j.hear("ok", T0) is None)

    m = fm()
    m.register(MoodCheck(m))
    c = m.pick(Environment(**FREE))
    check("a mood check is asked when there is no recent reading", c is not None and c.activity == "mood_check")
    m.mark_spoken("mood_check")
    check("an unclear answer is left to be handled as an ordinary utterance",
          m.hear("what's the weather", T0 + 10) is None and m.mood() is None)
    m._awaiting = ("mood_check", T0)
    r = m.hear("honestly exhausted", T0 + 10)
    check("a low answer is read, answered kindly, and remembered", r and "Sorry" in r and m.mood(T0 + 10) < -0.3, str(r))
    m.t[0] = T0 + 100
    m._last_by.clear()
    check("with a recent reading it does not ask again", m.pick(Environment(**FREE)) is None)
    check("the reading is believed only for a while", m.mood(T0 + 5 * HOUR) is None)
    m2 = fm()
    m2.register(MoodCheck(m2))
    m2.register(Joke())
    m2.set_mood(-0.6)
    check("a low reading keeps the jokes away", m2.pick(Environment(**FREE)) is None)
    m2.set_mood(0.6)
    check("a good one lets them through", m2.pick(Environment(**FREE)).activity == "joke")

    check("the answer window closes", (lambda mm: (mm.register(MoodCheck(mm)), mm.mark_spoken("mood_check"),
          mm.hear("great", T0 + 500))[2])(fm()) is None)

    # gap closure: competes with the others by the question's urgency
    w = tw.rig()
    w.observe("I told Omar something", "console", T0)
    m = default_moments({"min_gap_s": 0}, wonderer=w, clock=lambda: T0 + 100)
    c = m.pick(Environment(**FREE))
    check("an open gap is offered in a free moment, ahead of a joke",
          c is not None and c.activity == "gap_closure" and c.data["wonder"].subject.lower() == "omar", str(c))
    check("(and the wonderer's own rules still apply: not when the user is typing)",
          m.pick(Environment(idle_s=1.0, free_moment=False)) is None)
    w2 = tw.rig(enabled=False)
    w2.observe("I told Omar something", "console", T0)
    m = default_moments({"min_gap_s": 0}, wonderer=w2, clock=lambda: T0 + 100)
    check("with wondering off, the others still get the moment", m.pick(Environment(**FREE)).activity != "gap_closure")


def f3_extending():
    print("\n=== F3. Extending ===")
    m = fm()
    m.register(ProviderActivity("calibration", lambda env, now: ("Was that forecast right?", 0.7)))
    check("a one-function activity is enough", m.pick(Environment(**FREE)).text == "Was that forecast right?")
    m.register(ProviderActivity("system_assessment", lambda env, now: None))
    check("a provider with nothing to say is skipped", m.pick(Environment(**FREE)).activity == "calibration")

    got = []
    m = fm()
    m.register(ProviderActivity("cal", lambda e, n: ("Right?", 0.7), on_answer=lambda t, n: got.append(t) or "Noted."))
    m.mark_spoken("cal")
    check("its answer reaches it, and its reply is said", m.hear("yes", T0 + 5) == "Noted." and got == ["yes"])
    check("and is spent once", m.hear("yes", T0 + 6) is None)

    def boom(env, now):
        raise RuntimeError("x")
    m = fm()
    m.register(ProviderActivity("bad", boom))
    m.register(Fixed("ok", "fine", 0.5))
    check("a failing activity does not take the moment down", m.pick(Environment(**FREE)).text == "fine")
    m.add_fit(lambda c, e: 1 / 0)
    check("nor does a failing fit", m.pick(Environment(**FREE)) is not None)

    m = fm()
    m.register(Fixed("a", "A", 0.9))
    m.register(Fixed("quiet", "Q", 0.9))
    m.add_fit(lambda c, e: 0.0 if (e.activity or "").startswith("meeting") else 1.0)
    check("a fit can read what the user was doing",
          m.pick(Environment(activity="meeting notes", **FREE)) is None and m.pick(Environment(**FREE)) is not None)
    s = m.status()
    check("status says what is registered and the budget", s["activities"] == ["a", "quiet"] and s["budget"] == 2, str(s))


def rt_rig(**mc):
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    mc.setdefault("min_gap_s", 0)
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=20),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             wonder=True, wonder_cfg={"min_gap_s": 0},
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json"),
                             moments=True, moments_cfg=mc),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock = rt, ash, out, idle, clock

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


def f4_runtime():
    print("\n=== F4. The runtime ===")
    h = rt_rig()
    h.free()
    check("a free moment with no gaps is filled by something else", len(h.out) >= 1, str(h.said()))
    first = h.said()[0]
    check("what it said is the mood question (nothing is known of the mood)", first.startswith("How are you"), first)
    check("the model was not used", h.ash.calls == [])
    check("it counts against the budget", h.rt.moments.status()["spoken"] == 1)

    n = len(h.ash.calls)
    h.step(5, 5.0)
    h.say("pretty tired honestly", source="microphone", addressed=False)
    check("the answer is spent on the mood check, not sent to the model",
          len(h.ash.calls) == n and "Sorry" in h.said()[-1], str(h.said()))
    check("the mood is now known", h.rt.moments.mood(h.clock[0]) < -0.3)
    check("and the wonderer's environment reads it", h.rt._wonder_env(h.clock[0], 30.0).mood < -0.3)

    h.step(1800, 0.0)
    h.free()
    h.step(2000, 0.0)
    h.free()
    check("someone who is low gets no jokes", not any("bug" in s or "join" in s or "UDP" in s for s in h.said()), str(h.said()))

    h = rt_rig()
    h.say("I told Omar something about work")
    h.free()
    check("an open gap beats the mood check", h.said()[-1] == "Who is Omar to you?", str(h.said()))
    h.step(5, 5.0)
    h.say("my cousin", source="microphone", addressed=False)
    check("and its answer still goes to the wonderer", h.said()[-1] == "Got it -- Omar is your cousin.", str(h.said()[-1]))

    h = rt_rig(disabled=["mood_check", "gap_closure", "system_assessment"])
    h.free()
    check("with all but jokes disabled, a joke is what it says", len(h.out) == 1 and h.rt.moments.status()["spoken"] == 1)

    h = rt_rig(enabled=False)
    h.say("I told Omar something")
    h.free()
    check("with free moments off, the wonderer asks on its own as before, and no joke or mood check comes",
          h.said()[1:] == ["Who is Omar to you?"], str(h.said()))

    h = rt_rig()
    h.say("hello there")
    h.step(2, 0.0)
    h.say("how are you")
    check("while talking, nothing is added", h.rt.moments.status()["spoken"] == 0)

    h = rt_rig()
    h.rt.stats["errors"] = 3
    h.rt.moments.cfg.disabled = ("mood_check", "joke", "gap_closure")
    h.free()
    check("a system note is said when errors happened", any("internal errors" in s for s in h.said()), str(h.said()))
    h.rt.moments.cfg.max_per_hour = 9
    h.step(7 * HOUR, 0.0)
    h.free()
    check("and not again with nothing new", len([s for s in h.said() if "internal errors" in s]) == 1)
    check("status carries free_moments", "free_moments" in h.rt.status())


def f5_mood_from_queries():
    print("\n=== F5. Mood from what the user asks ===")
    from src.brain.brain import Brain
    b = Brain.__new__(Brain)
    b.sentiment_fn = tw.fake_sentiment
    b.last_sentiment = None
    r = b._sentiment("this is awful")
    check("the brain keeps its last reading, with when", b.last_sentiment["sentiment"] == "NEGATIVE"
          and "t" in b.last_sentiment and r["sentiment"] == "NEGATIVE")
    b.sentiment_fn = lambda t: 1 / 0
    b.last_sentiment = None
    b._sentiment("x")
    check("a failing model leaves nothing behind", b.last_sentiment is None)

    h = rt_rig(mood_check=False, jokes=False)
    seq = iter([{"sentiment": "NEGATIVE", "confidence": 0.97}, {"sentiment": "NEGATIVE", "confidence": 0.97}])

    def think(query, image=None, context=""):
        h.ash.calls.append(query)
        h.ash.brain.last_sentiment = {**next(seq), "t": h.clock[0]}
        return tw.Resp("Sure.")
    h.ash.think = think
    check("(no mood yet)", h.rt.moments.mood(h.clock[0]) is None)
    h.say("this deploy is a disaster")
    m0 = h.rt.moments.mood(h.clock[0])
    check("a query the model reads as negative lowers the mood", m0 is not None and m0 < -0.3, str(m0))
    check("and the wonderer sees it", h.rt._wonder_env(h.clock[0], 30.0).mood < -0.3)
    h.clock[0] += 60
    h.rt.moments.set_mood(0.8, h.clock[0])
    h.say("and the build broke too")
    m1 = h.rt.moments.mood(h.clock[0])
    check("a good mood is pulled down by a bad turn, not replaced", -0.3 < m1 < 0.3, str(m1))
    h.rt._after_answer("again", tw.Resp("x"))
    check("the same reading is not folded in twice", h.rt.moments.mood(h.clock[0]) == m1)


if __name__ == "__main__":
    for fn in (f1_registry, f2_builtins, f3_extending, f4_runtime, f5_mood_from_queries):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
