"""
tst_explain.py -- "why did you say that?".

  E1  Reading the question, and what is not it
  E2  The wording for each kind of reason
  E3  The runtime: the note left by an answer, a question, a free-moment
      remark, a watch, an action asked for; and asking twice explains the
      same thing; nothing said yet says so

Run:  PYTHONIOENCODING=utf-8 python tests/tst_explain.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.explain import asked_why, explain  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
T0 = tw.T0


def e1_reading():
    print("\n=== E1. Reading the question ===")
    for text in ("why did you say that?", "Why did you ask that", "why are you asking?", "why did you do that",
                 "how come you said that", "why?", "ash, why did you say that", "what made you say that?",
                 "but why are you telling me this", "why did you open that"):
        check(f"asks: {text!r}", asked_why(text))
    for text in ("why is the sky blue", "why did the build fail", "why does python do that",
                 "tell me why you like it", "", None, "why " * 40):
        check(f"not it: {str(text)[:30]!r}", not asked_why(text))


def e2_wording():
    print("\n=== E2. The wording ===")
    check("nothing yet", explain(None) == "I haven't said anything yet that I can explain.")
    s = explain({"kind": "answer", "query": "what's the weather", "intent": "get_weather", "tool": "get_weather",
                 "pathway": "fast"})
    check("an answer with a tool", s == "You asked \"what's the weather\", and I read it as get weather, so I used the "
                                        "get weather tool. The reply came from a quick reflex, without the language "
                                        "model.", s)
    s = explain({"kind": "answer", "query": "how are you", "intent": "conversation", "pathway": "slow"})
    check("conversation", s.startswith("You said \"how are you\", and I took it as conversation")
          and "language model" in s, s)
    s = explain({"kind": "wonder", "free": True, "wonder": {"kind": "person", "subject": "Omar", "mentions": 3}})
    check("a question about a person", s == "I asked because you'd mentioned Omar 3 times and I didn't know who that "
                                            "is. You'd paused, so it seemed a good moment to ask.", s)
    s = explain({"kind": "wonder", "free": False, "wonder": {"kind": "conflict", "subject": "Omar", "mentions": 1}})
    check("a clash, asked at a pause", "two things you told me about Omar don't match" in s
          and "important enough" in s, s)
    check("no pronoun is guessed", all(p not in explain({"kind": "wonder", "wonder": {"kind": k, "subject": "Sam"}})
                                       .split() for k in ("person", "role", "between") for p in ("he", "she")))
    check("a joke", "something light" in explain({"kind": "moment", "moment": "joke"}))
    check("a follow-up", "before it happened" in explain({"kind": "moment", "moment": "follow_up"}))
    check("a check-in", explain({"kind": "check_in", "minutes": 42}).startswith(
        "You'd been working for about 42 minutes"))
    check("the morning briefing", "this morning" in explain({"kind": "briefing", "briefing": "morning"}))
    check("a watch", explain({"kind": "watch", "what": "chrome closes"}) == "You asked me to tell you when chrome closes.")
    check("something asked for", explain({"kind": "asked", "query": "open outlook"}) == "You asked me to: \"open outlook\".")
    s = explain({"kind": "observation", "noticed": "battery at 15%", "reasons": ["urgency 0.55", "volunteered"]})
    check("an observation, with the reasons", s.startswith("I noticed battery at 15%") and "urgency 0.55" in s, s)
    check("an unknown note", explain({"kind": "?"}) == "I'm not sure I can explain that one.")


class Resp:
    def __init__(self, text, intent=None, tool=None):
        self.text, self.intent, self.tool_used = text, intent, tool
        self.pathway, self.trace = tw.Pathway(), None


class Ash(tw.Ash):
    def think(self, query, image=None, context=""):
        self.calls.append(query)
        if "weather" in query:
            return Resp("Sunny.", "get_weather", "get_weather")
        return Resp("Sure.", "conversation")


def rt_rig():
    clock = [T0]
    out = []
    ash = Ash()
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
                             desktop=True),
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
        clock[0] += 2
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    def free():
        for _ in range(15):
            step(10, 0.0, ticks=1)
        step(15, 25.0)

    h.step, h.say, h.free = step, say, free
    h.said = lambda: [o[0] for o in out]
    return h


def e3_runtime():
    print("\n=== E3. The runtime ===")
    h = rt_rig()
    h.say("why did you say that?")
    check("nothing said yet", h.said()[-1] == "I haven't said anything yet that I can explain.")
    h.say("what's the weather")
    n = len(h.ash.calls)
    h.say("why did you say that?")
    check("an answer is explained, without the model", len(h.ash.calls) == n and
          h.said()[-1].startswith("You asked \"what's the weather\", and I read it as get weather"), h.said()[-1])
    h.say("why?")
    check("asking twice explains the same thing", h.said()[-1] == h.said()[-2])

    h.say("I told Omar about it")
    h.step(400, 300.0)
    h.free()
    check("(a question was asked)", any(s == "Who is Omar to you?" for s in h.said()), str(h.said()))
    h.say("why are you asking?", source="microphone", addressed=False)
    check("a question is explained", h.said()[-1].startswith("I asked because you'd mentioned Omar"), h.said()[-1])
    check("and the question stays open", h.rt.wonder.status()["asked"] == 1)

    h = rt_rig()
    h.say("open outlook")
    h.say("why did you do that")
    check("an action asked for", h.said()[-1] == "You asked me to: \"open outlook\".", h.said()[-1])

    h = rt_rig()
    h.free()
    h.say("why did you say that", source="microphone", addressed=False)
    check("a free-moment check-in", h.said()[-1].startswith("You'd been working for about"), str(h.said()))

    import tst_watch as twa
    h = rt_rig()
    h.rt.cfg.watch = True
    from src.daemon.watch import WatchConfig, Watcher
    probe = twa.FakeProbe()
    h.rt.watcher = Watcher(WatchConfig(disk_trend=False), probe=probe, clock=lambda: h.clock[0])
    h.say("tell me when chrome closes")
    probe.procs.discard("chrome.exe")
    h.step(10, 300.0)
    check("(the watch fired)", h.said()[-1] == "chrome has closed.", str(h.said()))
    h.say("why did you say that?")
    check("a watch is explained", h.said()[-1] == "You asked me to tell you when chrome closes.", h.said()[-1])

    h = rt_rig()
    h.say("why did you say that", source="screen", addressed=False)
    check("text on the screen is not a question", not any("explain" in s for s in h.said()))


if __name__ == "__main__":
    for fn in (e1_reading, e2_wording, e3_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
