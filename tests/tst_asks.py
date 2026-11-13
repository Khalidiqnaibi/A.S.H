"""
tst_asks.py -- what ASH asked, what the user said back, and what that teaches about timing.

  A1  an ask is kept open; a reply settles it as answered / yes / no / later / moved on; nothing asked: nothing to settle
  A2  "I have a question for you when you have a minute" stays standing for hours; "go ahead", "what questions do you have?"
      and a plain yes take it up; something unrelated leaves it standing
  A3  the ledger: one line per ask, a line per outcome; survives a restart; a guest's words are not kept
  A4  the pair goes into episodic memory (and not a guest's)
  A5  timing: a no lowers a situation, a yes raises it; finer situations count more once they have evidence; it persists
  A6  what ASH tells the model; the report in words

Run:  PYTHONIOENCODING=utf-8 python tests/tst_asks.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.asks import Asks, TimingModel, context_of, keys_of  # noqa: E402

check = tw.check
def at(day, hour, minute=0):
    return time.mktime((2026, 3, 10 + day, hour, minute, 0, 0, 0, -1))


T0 = 1_800_000_000.0                      # a Thursday morning, local time does not matter for these


def make(directory=None, episodic=None, **kw):
    t = [T0]
    a = Asks(directory or tempfile.mkdtemp(), clock=lambda: t[0], episodic=episodic, user="Immortal", **kw)
    a.t = t
    return a


class Epi:
    def __init__(self):
        self.rows = []

    def add_episode(self, summary, event_type, related_entities=None, importance=0.5):
        self.rows.append((summary, event_type, importance))


def a1_settling():
    print("\n=== A1. Settling an ask ===")
    a = make()
    check("nothing asked: a reply is nothing to settle", a.hear("yes", T0) is None and a.current(T0) is None)
    q = a.asked("wonder", "What's your father's name?", T0, expects="answer", topic="role")
    check("it is open, and current", a.current(T0) is q and q.status == "open")
    r = a.hear("Afif", T0 + 8)
    check("a short reply is the answer", r.outcome == "answered" and r.reply == "Afif" and q.status == "answered", str(r))
    check("and it is no longer open", a.current(T0 + 9) is None)
    for said, want in (("no", "declined"), ("not now", "later"), ("later", "later"), ("no thanks", "declined"),
                       ("what time is it?", "moved_on"), ("I don't know really", "answered")):
        a2 = make()
        a2.asked("wonder", "What is brain.py for?", T0)
        got = a2.hear(said, T0 + 5)
        check(f"{said!r} -> {want}", got is not None and got.outcome == want, str(got and got.outcome))
    a3 = make()
    a3.asked("moment", "How are you holding up today?", T0)
    check("not said to ASH (overheard): not an answer", a3.hear("yes", T0 + 3, addressed=False) is None and a3.current(T0 + 3) is not None)
    check("two minutes on nobody has answered: it is settled as ignored",
          a3.current(T0 + 400) is None and a3.stats["ignored"] == 1)
    yn = make()
    yn.asked("moment", "Want me to look into that?", T0, expects="yes_no")
    check("a yes to a yes/no is accepted", yn.hear("yes please", T0 + 4).outcome == "accepted")
    many = make(max_open=3)
    for i in range(5):
        many.asked("moment", f"q{i}?", T0 + i)
    check("only a few stay open; the oldest are settled as ignored", len(many.open) == 3 and many.stats["ignored"] == 2)
    check("the latest is the one a reply answers", many.hear("blue", T0 + 6).ask.text == "q4?")


def a2_standing():
    print("\n=== A2. 'I have a question for you when you have a minute' ===")
    a = make()
    q = a.asked("briefing", "Good morning. I have a question for you when you have a minute.", T0, expects="permission")
    check("still standing three hours later", a.standing(T0 + 3 * 3600) is q)
    for said in ("yes", "sure", "go ahead", "What questions do you have", "what's the question?", "ask me", "okay, shoot",
                 "what is it", "I'm listening", "now is fine", "what do you want to know"):
        b = make()
        b.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
        r = b.hear(said, T0 + 2 * 3600)
        check(f"{said!r} takes it up", r is not None and r.outcome == "accepted", str(r and r.outcome))
    b = make()
    b.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
    r = b.hear("open spotify", T0 + 60)
    check("an unrelated request leaves it standing", r is None and b.standing(T0 + 61) is not None)
    check("'not now' puts it off", b.hear("not now", T0 + 90).outcome == "later" and b.standing(T0 + 91) is None)
    c = make()
    c.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
    check("'no' declines it", c.hear("no", T0 + 5).outcome == "declined")
    d = make()
    d.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
    check("after a day it is gone", d.standing(T0 + 9 * 3600) is None)


def a3_ledger():
    print("\n=== A3. The ledger ===")
    d = tempfile.mkdtemp()
    a = make(d)
    ctx = context_of(T0, "msedge.exe", "A.S.H blueprints", idle_s=300, dwell_min=22)
    a.asked("wonder", "What's your father's name?", T0, topic="role", ctx=ctx)
    a.hear("Afif", T0 + 12)
    rows = [json.loads(x) for x in open(os.path.join(d, "asks.jsonl"), encoding="utf-8")]
    check("an 'asked' line and a 'settled' line", [r["event"] for r in rows] == ["asked", "settled"], str([r["event"] for r in rows]))
    check("the situation is in it: the program, how long, reading or typing",
          rows[0]["ctx"]["app"] == "Edge" and rows[0]["ctx"]["mode"] == "reading" and rows[0]["ctx"]["dwell"] == "a_while", str(rows[0]["ctx"]))
    check("the reply, and how long it took", rows[1]["reply"] == "Afif" and rows[1]["latency_s"] == 12.0 and rows[1]["status"] == "answered")
    b = Asks(d, clock=lambda: T0 + 100)
    check("after a restart it still knows what was asked", b.recent and b.recent[-1].text == "What's your father's name?" and b.recent[-1].reply == "Afif")
    n = b.asked("moment", "Another?", T0 + 100)
    check("ids carry on, not restart", n.id == 2, str(n.id))
    g = make()
    g.asked("moment", "How are you?", T0)
    g.hear("fine thanks", T0 + 4, guest=True)
    check("a guest's words are not kept, and teach nothing", g.recent[-1].reply == "" and g.timing.updates == 0)
    nr = make(keep_replies=False)
    nr.asked("moment", "How are you?", T0)
    nr.hear("fine", T0 + 4)
    check("with replies switched off only the outcome is kept", nr.recent[-1].reply == "" and nr.recent[-1].status == "answered")


def a4_episodic():
    print("\n=== A4. Episodic memory ===")
    e = Epi()
    a = make(episodic=e)
    ctx = context_of(T0, "msedge.exe", "Body blueprints", idle_s=10, dwell_min=3)
    a.asked("moment", "How are you holding up today?", T0, ctx=ctx)
    a.hear("tired but fine", T0 + 6)
    check("the question and the answer are one episode", len(e.rows) == 1 and "How are you holding up today?" in e.rows[0][0]
          and "tired but fine" in e.rows[0][0] and "Immortal answered" in e.rows[0][0], str(e.rows))
    check("typed as an ask", e.rows[0][1] == "ask")
    check("where the user was", "Body blueprints" in e.rows[0][0], e.rows[0][0])
    a.asked("moment", "Want a joke?", T0 + 50)
    a.hear("no", T0 + 52)
    check("a no is remembered too", len(e.rows) == 2 and "said no" in e.rows[1][0], str(e.rows[1:]))
    a.asked("moment", "Another?", T0 + 100)
    a.expire(T0 + 1000)
    check("one nobody answered is not an episode", len(e.rows) == 2)
    g = make(episodic=e)
    g.asked("moment", "Hi?", T0)
    g.hear("hello", T0 + 1, guest=True)
    check("nor is a guest's", len(e.rows) == 2)


def a5_timing():
    print("\n=== A5. Timing ===")
    d = tempfile.mkdtemp()
    tm = TimingModel(os.path.join(d, "timing.json"))
    morning_code = {"part": "morning", "app": "Code", "mode": "pausing", "dwell": "long", "hour": 10}
    night_edge = {"part": "night", "app": "Edge", "mode": "reading", "dwell": "long", "hour": 1}
    check("no evidence: neutral", tm.fit(morning_code) == 1.0)
    for _ in range(6):
        tm.update(night_edge, "declined")
        tm.update(morning_code, "accepted")
    check("said no six times at night in Edge: much less welcome there", tm.fit(night_edge) < 0.6, f"{tm.fit(night_edge):.2f}")
    check("said yes six times in the code editor: more welcome", tm.fit(morning_code) > 1.2, f"{tm.fit(morning_code):.2f}")
    other = {"part": "night", "app": "Code", "mode": "typing", "dwell": "just_started", "hour": 2}
    check("another program at night is only somewhat affected", 0.4 < tm.fit(other) <= 1.2, f"{tm.fit(other):.2f}")
    keys = keys_of(night_edge)
    check("the situation is kept at several levels", "part:night|app:Edge" in keys and "*" in keys, str(keys))
    tm2 = TimingModel(os.path.join(d, "timing.json"))
    check("it persists", abs(tm2.fit(night_edge) - tm.fit(night_edge)) < 1e-9 and tm2.updates == 12)
    words = tm.describe()
    check("and says it in words", "leave you alone" in words and "Edge" in words, words)
    later = TimingModel()
    for _ in range(5):
        later.update(night_edge, "later")
    check("'later' counts against too", later.fit(night_edge) < 0.7, f"{later.fit(night_edge):.2f}")
    ign = TimingModel()
    for _ in range(5):
        ign.update(night_edge, "ignored")
    check("being ignored counts only a little against", 0.7 < ign.fit(night_edge) < 1.0, f"{ign.fit(night_edge):.2f}")
    bad = TimingModel(os.path.join(d, "nope", "x.json"))
    check("a missing or unwritable file is no trouble", bad.fit(night_edge) == 1.0)


def a6_context_and_report():
    print("\n=== A6. Context and report ===")
    a = make()
    check("no report before anything", a.report() == "I haven't asked you anything yet.")
    a.asked("moment", "How are you holding up today?", T0)
    check("what the model is told while an ask is waiting", "How are you holding up today?" in a.context_line(T0 + 4)
          and "may be their answer" in a.context_line(T0 + 4))
    a.hear("fine", T0 + 5)
    check("nothing once it is settled", a.context_line(T0 + 6) == "")
    a.asked("wonder", "Q2?", T0 + 20)
    a.hear("no", T0 + 21)
    rep = a.report()
    check("the report counts them", "asked 2 things" in rep and "answered 1" in rep and "said no to 1" in rep, rep)
    check("status", a.status()["asked"] == 2)
    h = a.history()
    check("history is oldest to newest and has the reply", [r["text"] for r in h] == ["How are you holding up today?", "Q2?"] and h[0]["reply"] == "fine")


def rig(briefings=False, **extra):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [at(0, 19) if briefings else T0]
    out = []
    ash = tw.Ash()
    epi = Epi()
    ash.episodic_memory = epi
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    cfg = dict(wonder=True, wonder_cfg={"min_gap_s": 0}, wonder_path=os.path.join(d, "w.json"),
               knowledge_path=os.path.join(d, "k.json"), asks=True, asks_dir=os.path.join(d, "asks"))
    if briefings:
        cfg.update(briefings=True, briefings_cfg={}, briefings_path=os.path.join(d, "b.json"))
    cfg.update(extra)
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=20, quiet_hours=None),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0, **cfg),
                        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock, h.epi, h.dir = rt, ash, out, idle, clock, epi, d

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True):
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed, salience=0.9, urgency=0.6))
        rt._tick()

    def work(seconds, every=10):
        for _ in range(int(seconds / every)):
            step(every, 0.0, ticks=1)

    def free():
        work(150)
        step(15, 25.0)
    h.step, h.say, h.work, h.free = step, say, work, free
    h.said = lambda: [o[0] for o in out]
    return h


def a7_runtime():
    print("\n=== A7. In the runtime ===")
    h = rig()
    h.say("I told Omar something about work")
    h.step(60, 40.0)
    h.free()
    asked = [o for o in h.out if o[0] == "Who is Omar to you?"]
    check("a question was asked in a free moment", len(asked) == 1, str(h.said()))
    check("and it was written down, with the situation", h.rt.asks.open and h.rt.asks.open[-1].text == "Who is Omar to you?"
          and h.rt.asks.open[-1].ctx.get("part"), str(h.rt.asks.open))
    n = len(h.ash.calls)
    h.step(5, 5.0)
    h.say("my cousin", source="microphone", addressed=False)
    check("the answer was spent on the question (no model)", h.said()[-1] == "Got it -- Omar is your cousin." and len(h.ash.calls) == n, h.said()[-1])
    led = [json.loads(x) for x in open(os.path.join(h.dir, "asks", "asks.jsonl"), encoding="utf-8")]
    check("the ledger has the ask and the answer", [r["event"] for r in led] == ["asked", "settled"] and led[1]["status"] == "answered" and led[1]["reply"] == "my cousin", str(led))
    check("the pair is in episodic memory", any("Who is Omar to you?" in e[0] and "my cousin" in e[0] for e in h.epi.rows), str(h.epi.rows))
    h.say("what have you asked me?")
    check("'what have you asked me?' is answered from the ledger, no model", "I've asked 1 thing" in h.said()[-1] and len(h.ash.calls) == n, h.said()[-1])

    # not now: remembered as a no for that situation
    h = rig()
    h.say("I told Omar something")
    h.free()
    h.say("not now")
    st = h.rt.asks.recent[-1]
    check("'not now' is written down as later, in this situation", st.status == "later" and st.ctx.get("app") is not None, str(st))
    check("and teaches the timing model", h.rt.asks.timing.updates == 1)
    check("it is in episodic memory too", any("said later" in e[0] for e in h.epi.rows), str(h.epi.rows))


def a8_the_briefing_question():
    print("\n=== A8. 'I have a question for you when you have a minute' ===")
    h = rig(briefings=True)
    h.say("I told Omar something about work")
    h.clock[0] = at(1, 8)
    h.step(1, 600.0)
    h.step(1, 1.0)
    h.step(20, 20.0)
    brief = [s for s in h.said() if "question for you" in s]
    check("(precondition) the briefing said it has a question", len(brief) == 1, str(h.said()))
    st = h.rt.asks.standing(h.clock[0])
    check("it is standing, waiting to be taken up", st is not None and st.expects == "permission", str(h.rt.asks.open))
    h.step(2.5 * 3600, 30.0)
    check("still standing hours later", h.rt.asks.standing(h.clock[0]) is not None)
    n = len(h.ash.calls)
    h.say("What questions do you have")
    check("'what questions do you have' gets the question, not 'what did you mean'", h.said()[-1].endswith("Who is Omar to you?") and len(h.ash.calls) == n, h.said()[-1])
    check("the permission is settled as taken up", h.rt.asks.stats["accepted"] == 1, str(h.rt.asks.stats))

    h2 = rig(briefings=True)
    h2.say("I told Omar something about work")
    h2.clock[0] = at(1, 8)
    h2.step(1, 600.0)
    h2.step(1, 1.0)
    h2.step(20, 20.0)
    n = len(h2.ash.calls)
    h2.say("yes")
    check("a bare yes takes it up too", h2.said()[-1].endswith("Who is Omar to you?") and len(h2.ash.calls) == n, h2.said()[-1])
    h3 = rig(briefings=True)
    h3.say("I told Omar something about work")
    h3.clock[0] = at(1, 8)
    h3.step(1, 600.0)
    h3.step(1, 1.0)
    h3.step(20, 20.0)
    h3.say("not now")
    check("'not now' puts it off, politely", h3.said()[-1] == "Okay, I'll ask later." and h3.rt.asks.standing(h3.clock[0]) is None, h3.said()[-1])
    h4 = rig(briefings=True)
    h4.say("I told Omar something about work")
    h4.clock[0] = at(1, 8)
    h4.step(1, 600.0)
    h4.step(1, 1.0)
    h4.step(20, 20.0)
    h4.say("what time is it")
    check("something else entirely leaves it standing", h4.rt.asks.standing(h4.clock[0]) is not None)
    h4.say("yes", addressed=False, source="microphone")
    check("a 'yes' that was not said to ASH does not take it up", h4.rt.asks.standing(h4.clock[0]) is not None)


def a9_the_model_knows():
    print("\n=== A9. The model is told what it asked ===")
    h = rig(briefings=True)
    seen = {}
    real = h.ash.think

    def think(query, image=None, context=""):
        seen["context"] = context
        return real(query, image=image, context=context)
    h.ash.think = think
    h.rt.asks.asked("moment", "How are you holding up today?", h.clock[0], expects="answer")
    h.say("honestly a bit tired, the deadline is close")
    check("the line the model is given says what it had asked", "How are you holding up today?" in seen.get("context", "")
          and "their answer" in seen.get("context", ""), seen.get("context", "")[:200])
    h.say("what is the capital of France")
    check("and nothing about it afterwards", "holding up" not in seen.get("context", ""))


def a10_timing_in_the_chooser():
    print("\n=== A10. What the answers taught is used when choosing ===")
    from src.daemon.freemoment import Candidate
    from src.daemon.wondering import Environment
    h = rig(moments=True, moments_cfg={"jokes": False, "mood_check": False, "min_gap_s": 0})
    ctx = h.rt._ask_ctx(h.clock[0], 40.0)
    env = Environment(idle_s=40.0)
    c = Candidate("gap_closure", "Who is Omar to you?", 0.8)
    before = h.rt.moments.fit(c, env)
    for _ in range(6):
        h.rt.asks.asked("moment", "Another question?", h.clock[0], ctx=ctx)
        h.rt.asks.hear("not now", h.clock[0] + 2)
    after = h.rt.moments.fit(c, env)
    check("after six 'not now's in this situation the chooser weighs asking down", before == 1.0 and after < 0.6, f"{before} -> {after}")
    check("it is in the timing file, readable", os.path.exists(os.path.join(h.dir, "asks", "timing.json")))
    check("and in words", "leave you alone" in h.rt.asks.report(), h.rt.asks.report())
    h.say("when do I like to be asked things?")
    check("'when do I like to be asked things?' answers from it", "I've asked 6 things" in h.said()[-1], h.said()[-1])


if __name__ == "__main__":
    for fn in (a1_settling, a2_standing, a3_ledger, a4_episodic, a5_timing, a6_context_and_report, a7_runtime, a8_the_briefing_question, a9_the_model_knows, a10_timing_in_the_chooser):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
