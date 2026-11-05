"""
tst_corrections.py -- learning from being corrected.

  C1  Reading a correction: "no, I meant ...", "that's not what I asked",
      "wrong one"; and what is not one ("no thanks", "no, I'm fine")
  C2  The memory: record, repeat, cap, persistence, forget
  C3  The nudge: the wrong reading down, the meant one up, only for things
      said the same way, bounded
  C4  The classifier uses it
  C5  The brain is told: the outcome model and the concept graph
  C6  The runtime: a correction is answered as what was meant and learned;
      a bare correction asks what was wanted; nothing changes for chat

Run:  PYTHONIOENCODING=utf-8 python tst_corrections.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain import corrections  # noqa: E402
from src.brain.corrections import CorrectionMemory, read_correction, similarity  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.senses.base import PeripheryRegistry  # noqa: E402

check = tw.check
T0 = tw.T0


def c1_reading():
    print("\n=== C1. Reading a correction ===")
    cases = {
        "no, I meant search my files": "search my files",
        "No I meant the weather in Amman": "the weather in Amman",
        "I meant convert 5 miles to km": "convert 5 miles to km",
        "that's not what I asked": "",
        "that's not what I asked, I wanted the time": "the time",
        "wrong one": "",
        "not that, set a reminder for 5": "set a reminder for 5",
        "no, search my files for TODO": "search my files for TODO",
        "I didn't ask for that": "",
        "you misunderstood": "",
        "what I meant was the date": "the date",
        "no, I wanted to open the readme": "open the readme",
    }
    for text, want in cases.items():
        got = read_correction(text)
        check(f"correction: {text!r}", got == want, repr(got))
    for text in ("no", "no thanks", "nope, I'm good", "no, that's fine", "no I don't think so",
                 "nah not now", "what's the weather", "no way that's amazing", "", None, "x" * 400,
                 "no, I meant " + "x" * 400):
        check(f"not a correction: {str(text)[:30]!r}", read_correction(text) is None, repr(read_correction(text)))


def c2_memory():
    print("\n=== C2. The memory ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.json")
    m = CorrectionMemory(p)
    c = m.record("what's the weather like in my code", "get_weather", "search_files", now=T0)
    check("recorded", c is not None and len(m.items) == 1)
    m.record("what's the weather like in my code", "get_weather", "search_files", now=T0 + 5)
    check("the same correction again strengthens it", len(m.items) == 1 and m.items[0].times == 2)
    check("nothing recorded for an empty or self-same one",
          m.record("", "x") is None and m.record("hello", "a", "a") is None and m.record("the a", "x") is None)
    m2 = CorrectionMemory(p)
    check("survives a restart", len(m2.items) == 1 and m2.items[0].times == 2 and m2.items[0].right == "search_files")
    small = CorrectionMemory(None, max_items=3)
    for i in range(6):
        small.record(f"phrase number {i} here", "a", "b", now=T0 + i)
    check("capped", len(small.items) == 3)
    m2.forget()
    check("forget empties it, on disk too", CorrectionMemory(p).items == [])
    with open(p, "w") as fh:
        fh.write("{nope")
    check("a broken file is set aside", CorrectionMemory(p).items == [] and os.path.exists(p + ".bad"))
    check("no path: nothing written", CorrectionMemory(None).record("open my notes", "x", "y") is not None)


def c3_nudge():
    print("\n=== C3. The nudge ===")
    m = CorrectionMemory(None)
    m.record("what's the weather like in my code", "get_weather", "search_files", now=T0)
    base = [("get_weather", 0.62), ("search_files", 0.50), ("calculate", 0.2)]
    out = m.adjust("what's the weather like in my code", base)
    check("said the same way: the meant reading wins", out[0][0] == "search_files", str(out))
    check("the wrong one goes down", dict(out)["get_weather"] < 0.62)
    check("others are untouched", dict(out)["calculate"] == 0.2)
    out = m.adjust("what is the weather in Amman", base)
    check("said differently: nothing changes", out == base, str(out))
    check("(that sentence is not similar enough)",
          similarity("what's the weather like in my code", "what is the weather in Amman") < 0.6)
    out = m.adjust("weather like in my code", [("get_weather", 0.6)])
    check("a meant reading the catalog did not rank is added", "search_files" in dict(out)
          and dict(out)["get_weather"] < 0.6, str(out))
    m2 = CorrectionMemory(None)
    m2.record("play some music", "open_app", "", now=T0)
    out = m2.adjust("play some music", [("open_app", 0.7), ("conversation", 0.1)])
    check("'not that' with nothing meant only pushes the wrong one down", dict(out)["open_app"] < 0.7
          and len(out) == 2)
    for _ in range(20):
        m.record("what's the weather like in my code", "get_weather", "search_files", now=T0)
    out = m.adjust("what's the weather like in my code", base)
    check("repetition strengthens it, but it stays bounded",
          dict(out)["search_files"] <= 0.50 + 2 * 0.25 + 1e-9, str(out))
    check("scores never go negative", all(s >= 0 for _, s in m.adjust(
        "what's the weather like in my code", [("get_weather", 0.01)])))


def c4_classifier():
    print("\n=== C4. The classifier uses it ===")
    from tools import classification as cl

    class Cat:
        labels = ["get_weather", "search_files", "calculate"]

        def most_similar(self, q, top_k=3):
            return [("get_weather", 0.62), ("search_files", 0.50), ("calculate", 0.2)][:top_k]

    old_cat = cl._INTENT_CATALOG
    old_mem = corrections.default()
    try:
        cl._INTENT_CATALOG = Cat()
        corrections.configure(None)
        check("before: the catalog's first choice", cl._ranked("what's the weather like in my code", 1)[0][0]
              == "get_weather")
        corrections.default().record("what's the weather like in my code", "get_weather", "search_files")
        check("after a correction: the meant one", cl._ranked("what's the weather like in my code", 1)[0][0]
              == "search_files")
        check("other sentences unchanged", cl._ranked("weather in Amman tomorrow", 1)[0][0] == "get_weather")
    finally:
        cl._INTENT_CATALOG = old_cat
        corrections._DEFAULT = old_mem


def c5_brain():
    print("\n=== C5. The brain is told ===")
    from src.brain.brain import Brain

    class Pred:
        def __init__(self):
            self.seen = []

        def record_outcome(self, intent, ok):
            self.seen.append((intent, ok))

    class Concepts:
        def __init__(self):
            self.r = []

        def reward(self, r):
            self.r.append(r)

    b = Brain.__new__(Brain)
    b.predictive, b.concepts = Pred(), Concepts()
    b.note_misroute("get_weather")
    check("the outcome model counts a failure", b.predictive.seen == [("get_weather", False)])
    check("the concept graph is punished", b.concepts.r == [-0.8])
    b.note_misroute("conversation")
    b.note_misroute(None)
    check("chat and nothing are not misroutes", len(b.predictive.seen) == 1)
    b.concepts = None
    b.note_misroute("x")
    check("works with no concept graph", b.predictive.seen[-1] == ("x", False))


class Resp:
    def __init__(self, text, intent=None, tool=None):
        self.text, self.intent, self.tool_used = text, intent, tool
        self.pathway, self.trace = tw.Pathway(), None


class Ash(tw.Ash):
    ROUTES = {"weather": "get_weather", "search": "search_files", "files": "search_files",
              "remind": "set_reminder"}

    def think(self, query, image=None, context=""):
        self.calls.append(query)
        intent = next((v for k, v in self.ROUTES.items() if k in query.lower()), "conversation")
        return Resp(f"[{intent}] ok", intent=intent, tool=None if intent == "conversation" else intent)


class Brain(tw.Brain):
    def __init__(self):
        super().__init__()
        self.misroutes = []

    def note_misroute(self, intent):
        self.misroutes.append(intent)


def rt_rig(enabled=True):
    clock = [T0]
    out = []
    ash = Ash()
    ash.brain = Brain()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, corrections=enabled),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.clock = rt, ash, out, clock

    def say(text, source="console", addressed=True, dt=3):
        clock[0] += dt
        rt.submit(tw.SensorEvent(source=source, modality=tw.Modality.TEXT, text=text, addressed=addressed,
                                 salience=0.9, urgency=0.6))
        rt._tick()

    h.say = say
    h.said = lambda: [o[0] for o in out]
    return h


def c6_runtime():
    print("\n=== C6. The runtime ===")
    old = corrections.default()
    try:
        corrections.configure(None)
        h = rt_rig()
        h.say("what's the weather like in my code")
        check("(first read as weather)", h.said()[-1] == "[get_weather] ok")
        h.say("no, I meant search my files")
        check("the correction is answered as what was meant", h.ash.calls[-1] == "search my files"
              and h.said()[-1] == "[search_files] ok", str(h.ash.calls))
        check("the brain was told the reading failed", h.ash.brain.misroutes == ["get_weather"])
        c = corrections.default().items
        check("and it is remembered with what was meant",
              len(c) == 1 and c[0].wrong == "get_weather" and c[0].right == "search_files"
              and c[0].said == "what's the weather like in my code", str(c))
        check("status carries it", h.rt.status()["corrections"]["heard"] == 1)

        corrections.configure(None)
        h = rt_rig()
        h.say("remind me about the weather")
        h.say("that's not what I asked")
        n = len(h.ash.calls)
        check("a bare correction asks what was wanted, without the model",
              h.said()[-1] == "Sorry -- what did you want?" and n == 1)
        check("and is remembered as 'not that'", corrections.default().items[0].right == "")
        h.say("search my files for the plan", addressed=False)
        check("the next thing said is answered as the meant request",
              h.ash.calls[-1] == "search my files for the plan" and h.said()[-1] == "[search_files] ok",
              str(h.said()))
        check("and the correction learns what was meant", corrections.default().items[0].right == "search_files",
              str(corrections.default().items))

        corrections.configure(None)
        h = rt_rig()
        h.say("how are you")
        h.say("no, I meant what are you up to")
        check("after plain chat, 'no, I meant' is just conversation",
              h.ash.brain.misroutes == [] and corrections.default().items == [])

        h = rt_rig()
        h.say("what's the weather")
        h.clock[0] += 600
        h.say("no, I meant search my files")
        check("too long after: not a correction of that reply", h.ash.brain.misroutes == [])

        h = rt_rig()
        h.say("what's the weather")
        h.say("no, I meant search my files", source="screen", addressed=False)
        check("text on the screen is not a correction", h.ash.brain.misroutes == [])

        h = rt_rig(enabled=False)
        h.say("what's the weather")
        h.say("no, I meant search my files")
        check("switched off: taken as an ordinary sentence",
              h.ash.brain.misroutes == [] and h.ash.calls[-1] == "no, I meant search my files")

        h = rt_rig()
        h.say("what's the weather")
        h.say("no thanks")
        check("'no thanks' is not a correction", h.ash.brain.misroutes == [])
    finally:
        corrections._DEFAULT = old


if __name__ == "__main__":
    for fn in (c1_reading, c2_memory, c3_nudge, c4_classifier, c5_brain, c6_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
