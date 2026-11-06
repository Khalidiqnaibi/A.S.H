"""
tst_session3.py -- the second round of fixes from the session of 2026-11-06.

  T1  Doubtful transcripts are dropped, sure ones kept, per phrase
  T2  The model does not say a sentence it already said in the last ten
      minutes; a rule's line is never trimmed
  T3  A question the user has since said again, or corrected, is not
      answered; unrelated ones all are
  T4  One copy of the embedding model, on the CPU, shared
  T5  Crashes: read from the event log, told once, in the briefing or when
      asked; the first read is history, not news

Run:  PYTHONIOENCODING=utf-8 python tst_session3.py
"""

import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
T0 = tw.T0
HOUR = 3600.0
DAY = 86400.0


def t1_transcripts():
    print("\n=== T1. Doubtful transcripts ===")
    from src.senses.mic import trusted_text

    def seg(text, lp=-0.3, ns=0.02):
        return NS(text=text, avg_logprob=lp, no_speech_prob=ns)
    check("a sure phrase is kept", trusted_text([seg(" Ash, what time is it")]) == "Ash, what time is it")
    check("an unsure one is dropped", trusted_text([seg(" Ash brothers emote", lp=-1.4)]) == "")
    check("right at the limit is kept", trusted_text([seg(" hello", lp=-1.0)]) == "hello")
    check("just past it is dropped", trusted_text([seg(" hello", lp=-1.01)]) == "")
    check("probably not speech, and unsure: dropped", trusted_text([seg(" Thanks for watching", lp=-0.8, ns=0.7)]) == "")
    check("probably not speech but sure of the words: kept", trusted_text([seg(" yes", lp=-0.3, ns=0.7)]) == "yes")
    check("unsure words but surely speech: kept", trusted_text([seg(" maybe", lp=-0.8, ns=0.1)]) == "maybe")
    check("per phrase: the sure one survives", trusted_text([seg(" Ash,"), seg(" mumble mumble", lp=-1.6),
                                                             seg(" what time is it")]) == "Ash, what time is it")
    check("empty phrases are skipped", trusted_text([seg("  "), seg(" ok")]) == "ok")
    check("no scores at all: kept as is", trusted_text([NS(text=" plain")]) == "plain")
    check("nothing: nothing", trusted_text([]) == "")
    import numpy as np
    from src.senses.mic import MicrophoneSensor
    got = {}

    class W:
        def transcribe(self, audio, **kw):
            got.update(kw)
            return [seg(" Ash what's up"), seg(" ghost words", lp=-1.5)], None
    m = MicrophoneSensor.__new__(MicrophoneSensor)
    m._transcriber, m._whisper = None, W()
    check("through the microphone's own transcribe", m.transcribe(np.zeros(16000, dtype=np.int16)) == "Ash what's up")
    check("each phrase read on its own", got.get("condition_on_previous_text") is False)


def t2_repeat():
    print("\n=== T2. Not twice ===")
    from src.daemon.repeat import NOTHING_NEW, RepeatGuard
    g = RepeatGuard(clock=lambda: 0)
    A = "I didn't go to privacy mode because there was no trigger to activate it."
    B = "As you know, my actions are deterministic and I don't invent facts."
    check("said once: as it is", g.check(A + " " + B, 0) == A + " " + B)
    check("again, with something new: only the new part", g.check(A + " I can't promise it will be different.", 60)
          == "I can't promise it will be different.")
    check("again with nothing new: said so", g.check(A, 70) == NOTHING_NEW)
    check("a reworded repeat counts", g.check("I didn't go to privacy mode, because there was no trigger to activate "
                                              "it at all.", 80) == NOTHING_NEW)
    check("related but different sentences are both kept",
          g.check("Today it is sunny and warm in Amman and the humidity is low.", 85)
          == "Today it is sunny and warm in Amman and the humidity is low."
          and g.check("Today it is sunny in Paris with a light breeze.", 86)
          == "Today it is sunny in Paris with a light breeze.")
    g3 = RepeatGuard(clock=lambda: 0)
    g3.check("I turned the microphone off but the camera is still on for you.", 0)
    check("the same words in a different order are a different sentence",
          g3.check("I turned the camera off but the microphone is still on for you.", 5)
          == "I turned the camera off but the microphone is still on for you.")
    check("short words may be said twice", g.check("Okay.", 90) == "Okay." and g.check("Okay.", 91) == "Okay.")
    check("ten minutes later it may be said again", g.check(A, 700) == A)
    g2 = RepeatGuard(clock=lambda: 0)
    g2.note("Privacy mode on. The microphone, camera and screen are off.", 0)
    check("what a rule said counts as said", g2.check("The microphone, camera and screen are off.", 5) == NOTHING_NEW)
    check("empty in, empty out", g2.check("", 5) == "")


def rig(reply="Sure.", slow=True):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            self.calls.append(query)
            r = tw.Resp(reply if callable(reply) is False else reply(query))
            if slow:
                r.pathway = type("P", (), {"value": "slow"})()
            return r
    ash = Ash()
    rt = AmbientRuntime(ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9),
                        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]

    def submit(text, source="console", dt=1.0):
        clock[0] += dt
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=True, salience=0.9,
                              urgency=0.6, ts=clock[0]))

    def say(text, source="console"):
        submit(text, source)
        rt._tick()
    rt.say, rt.submit_only, rt.out, rt.ash_ = say, submit, out, ash
    return rt


def t3_overtaken():
    print("\n=== T3. Overtaken questions ===")
    from src.senses import address
    rt = rig()
    rt.submit_only("what do you see")
    rt.submit_only("what do you see on the camera")
    rt._tick()
    check("said again, a little differently: only the later one is answered",
          rt.ash_.calls == ["what do you see on the camera"], str(rt.ash_.calls))
    check("counted", rt.stats.get("overtaken") == 1)
    rt = rig()
    rt.submit_only("what's the capital of France")
    rt.submit_only("no, I meant Spain")
    rt._tick()
    check("corrected: only the correction is answered", len(rt.ash_.calls) == 1 and "Spain" in rt.ash_.calls[0],
          str(rt.ash_.calls))
    rt = rig()
    rt.submit_only("what's the capital of France")
    rt.submit_only("tell me a fact about volcanoes")
    rt._tick()
    check("two different questions: both are answered", len(rt.ash_.calls) == 2, str(rt.ash_.calls))
    rt = rig()
    rt.submit_only("what do you see")
    rt._tick()
    rt.submit_only("what do you see")
    rt._tick()
    check("the same question a tick later (nothing behind it): answered", len(rt.ash_.calls) == 2)
    rt = rig()
    rt.submit_only("what do you see")
    rt.submit_only("what do you see on the camera", source="screen")
    rt._tick()
    check("a later line that is not from the user does not overtake the question",
          rt.ash_.calls[0] == "what do you see" and rt.stats.get("overtaken", 0) == 0, str(rt.ash_.calls))
    rt = rig()
    rt.submit_only("what do you see")
    rt.submit_only("privacy mode")
    rt._tick()
    check("a command in between is done and the question still answered",
          rt.privacy.paused and rt.ash_.calls == ["what do you see"], str(rt.ash_.calls))


def t3b_repeat_runtime():
    print("\n=== T3b. The repeat guard in the runtime ===")
    rt = rig(reply="I didn't go to privacy mode because there was no trigger to activate it.")
    rt.say("why didn't you do it")
    first = rt.out[-1]
    rt.say("but I did order you to do it, why not")
    check("the model said it twice: the second time, 'that's what I just said'", rt.out[-1] == "That's what I just said.",
          f"{first!r} / {rt.out[-1]!r}")
    rt = rig(reply="Privacy mode off -- sensors are back on.")
    rt.say("privacy off")
    check("(the rule's line)", rt.out[-1] == "Privacy mode off -- sensors are back on.", rt.out[-1])
    rt.say("what did you just do")
    check("the model repeating what a rule just said is trimmed", rt.out[-1] == "That's what I just said.", rt.out[-1])
    rt.say("privacy mode")
    first = rt.out[-1]
    rt.privacy.resume()
    rt.say("privacy mode")
    check("a rule says the same thing every time, untrimmed", first.startswith("Privacy mode on") and rt.out[-1] == first,
          rt.out[-1])


def t4_shared_model():
    print("\n=== T4. One copy of the model ===")
    import importlib
    from tools import shared_model as sm
    import tools.classification as cl
    from src.memory.disambiguation import Disambiguator
    if cl._INTENT_CATALOG.model is None:
        check("(no sentence-transformers here)", True)
        return
    before = sm.loads
    d = Disambiguator()
    check("the disambiguator reuses the classifier's copy", d.model is cl._INTENT_CATALOG.model and sm.loads == before,
          f"{before} -> {sm.loads}")
    check("on the CPU", str(getattr(d.model, "device", "cpu")).startswith("cpu"))
    cl._INTENT_CATALOG._source_mtime = -1          # the intent files look newer: the catalog is rebuilt
    try:
        cl._ensure_catalogs_loaded(force=True)
    except Exception:
        pass
    check("rebuilding the catalog does not load the model again", sm.loads == before, f"{before} -> {sm.loads}")
    check("asking again loads nothing", sm.load(sm._MODELS and next(iter(sm._MODELS))) is d.model and sm.loads == before)
    os.environ["ASH_USE_GPU"] = "1"
    try:
        check("ASH_USE_GPU=1 opts back in", sm.device() is None)
    finally:
        os.environ.pop("ASH_USE_GPU", None)
    check("default is the CPU", sm.device() == "cpu")
    src = open(os.path.join("src", "memory", "disambiguation.py"), encoding="utf-8").read()
    check("nothing creates a SentenceTransformer by itself any more",
          "SentenceTransformer(self.model_name)" not in src)


def t5_crashes():
    print("\n=== T5. Crashes ===")
    from src.daemon.crashwatch import CrashWatch
    p = os.path.join(tempfile.mkdtemp(), "crashes.json")
    t = [T0]
    log = [{"t": T0 - 3 * HOUR, "code": 10}, {"t": T0 - 5 * DAY, "code": 0}, {"t": T0 - 12 * DAY, "code": 0},
           {"t": T0 - 90 * DAY, "code": 0}]
    c = CrashWatch(path=p, clock=lambda: t[0], reader=lambda: list(log), background=False)
    c.refresh(force=True)
    check("the first read is history, not news", c.briefing_part("morning", T0) is None)
    check("...but it can be asked about", "shut down unexpectedly" in c.hear("did my computer crash lately?", T0))
    line = c.hear("any crashes lately", T0)
    check("with the stop code in words, and the count", "a driver touching memory" in line and "3 times in the last 30 days"
          in line and "mdsched.exe" in line, line)
    two = CrashWatch(clock=lambda: T0, reader=lambda: log[:2], background=False)
    two.refresh(force=True)
    check("two in a month is counted, not yet called a pattern", "That's 2 times in the last 30 days." in
          two.hear("any crashes", T0) and "pattern" not in two.hear("any crashes", T0))
    one = CrashWatch(clock=lambda: T0, reader=lambda: log[:1], background=False)
    one.refresh(force=True)
    check("one is just said", "That's" not in one.hear("any crashes", T0))
    t[0] += 6 * HOUR
    log.insert(0, {"t": t[0] - HOUR, "code": 0})
    c.refresh(t[0], force=True)
    part = c.briefing_part("morning", t[0])
    check("a new one is told in the briefing", part and "with no error recorded" in part and "4 times" in part, str(part))
    check("...once", c.briefing_part("morning", t[0]) is None)
    check("not in other briefings", c.briefing_part("focus", t[0]) is None)
    c2 = CrashWatch(path=p, clock=lambda: t[0], reader=lambda: list(log), background=False)
    c2.refresh(force=True)
    check("what was told is kept across a restart", c2.briefing_part("morning", t[0]) is None)
    log.insert(0, {"t": t[0] + 2 * HOUR, "code": 0})
    log.insert(0, {"t": t[0] + 3 * HOUR, "code": 0x12B})
    t[0] += 4 * HOUR
    c2.refresh(t[0], force=True)
    part = c2.briefing_part("evening", t[0])
    check("two at once: said so, the latest first", part.startswith("2 unexpected shutdowns since I last told you")
          and "corrupted memory" in part, part)
    quiet = CrashWatch(clock=lambda: T0, reader=lambda: [{"t": T0 - 40 * DAY, "code": 0}], background=False)
    quiet.refresh(force=True)
    check("nothing this month: said plainly", quiet.hear("did my computer crash", T0) == "No unexpected shutdowns in the "
                                                                                      "last 30 days.")
    called = []
    off = CrashWatch(enabled=False, clock=lambda: T0, reader=lambda: called.append(1) or [], background=False)
    off.refresh(force=True)
    check("off: the log is not even read", called == [])
    broken = CrashWatch(clock=lambda: T0, reader=lambda: None, background=False)
    broken.refresh(force=True)
    check("can't read the log: said so", "can't read the Windows event log" in broken.hear("any crashes", T0))
    unread = CrashWatch(clock=lambda: T0, reader=lambda: [], background=False)
    check("not read yet: reads, and asks to be asked again", "haven't read the crash log yet" in unread.hear("any crashes", T0))
    for text in ("how do I fix a crash in my code", "the game crashed", "I crashed my car", "", None):
        check(f"not a crash question: {str(text)[:30]!r}", c.hear(text, T0) is None)
    check("off: nothing", CrashWatch(enabled=False, reader=lambda: log, background=False).hear("any crashes", T0) is None)
    n = [0]

    def reader():
        n[0] += 1
        return []
    r = CrashWatch(clock=lambda: t[0], reader=reader, background=False, refresh_s=100)
    r.refresh(t[0])
    r.refresh(t[0] + 50)
    r.refresh(t[0] + 150)
    check("read at most every refresh interval", n[0] == 2, str(n[0]))

    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    out = []
    rt = AmbientRuntime(ash=tw.Ash(), periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()),
                        gate=AttentionGate(), privacy=PrivacyPolicy(),
                        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, crashes=True,
                                             crashes_path=os.path.join(tempfile.mkdtemp(), "c.json")),
                        response_sink=lambda x, m: out.append(x))
    rt.crashwatch.reader, rt.crashwatch.background = (lambda: [{"t": time.time() - 3600, "code": 0xA}]), False
    rt.crashwatch.refresh(force=True)
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="did my computer crash?", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("the runtime answers it without the model", out and "shut down unexpectedly" in out[-1] and rt.ash.calls == [],
          str(out))
    check("status", rt.status()["crashes"]["events"] == 1)


if __name__ == "__main__":
    for fn in (t1_transcripts, t2_repeat, t3_overtaken, t3b_repeat_runtime, t4_shared_model, t5_crashes):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
