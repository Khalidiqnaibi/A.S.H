"""
tst_backoff.py -- talking less, by learning what is not wanted.

  B1  The score: rejected and dismissed count 1, deferred half, silence
      nothing; three within a week benches for three days; welcome clears it
  B2  Free moments: a benched activity sits out, the others do not
  B3  Questions: a benched kind is not asked at a free moment, an urgent one
      still is, and asking "any questions?" still offers it; kept across a
      restart
  B4  The runtime: waving off jokes benches jokes, not the rest

Run:  PYTHONIOENCODING=utf-8 python tests/tst_backoff.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.backoff import Backoff  # noqa: E402
from src.daemon.freemoment import Candidate, FreeMoments, FreeMomentsConfig  # noqa: E402
from src.daemon.wondering import Environment, Knowledge, Wonderer, WonderingConfig  # noqa: E402

check = tw.check
T0 = tw.T0
DAY = 86400.0
FREE = dict(idle_s=60.0, free_moment=True)


def b1_score():
    print("\n=== B1. The score ===")
    t = [T0]
    b = Backoff(clock=lambda: t[0])
    check("two rejections: not yet", not b.note("joke", "rejected") and not b.note("joke", "dismissed")
          and not b.benched("joke"))
    check("the third benches it", b.note("joke", "rejected") and b.benched("joke"))
    t[0] += 3 * DAY + 1
    check("for three days", not b.benched("joke"))
    b = Backoff(clock=lambda: t[0])
    for _ in range(5):
        b.note("joke", "deferred")
    check("'not now' counts half: five are not enough", not b.benched("joke"))
    b.note("joke", "deferred")
    check("six are", b.benched("joke"))
    b = Backoff(clock=lambda: t[0])
    for _ in range(10):
        b.note("joke", "silence")
    check("silence alone never benches", not b.benched("joke"))
    b = Backoff(clock=lambda: t[0])
    b.note("joke", "rejected")
    b.note("joke", "rejected")
    b.note("joke", "engaged")
    b.note("joke", "rejected")
    check("a welcome clears the score", not b.benched("joke"))
    b = Backoff(clock=lambda: t[0])
    b.note("joke", "rejected")
    b.note("joke", "rejected")
    t[0] += 8 * DAY
    b.note("joke", "rejected")
    check("only the last week counts", not b.benched("joke"))
    check("other things are not touched", not b.benched("story"))
    d = Backoff(clock=lambda: t[0])
    for _ in range(3):
        d.note("joke", "rejected")
    e = Backoff(clock=lambda: t[0])
    e.load(d.dump())
    check("kept across a dump and load", e.benched("joke"))


class Fixed:
    def __init__(self, name, value):
        self.name, self.value, self.weight, self.cooldown_s = name, value, 1.0, 0.0

    def propose(self, env, now):
        return Candidate(self.name, self.name, self.value)

    def spoken(self, now):
        pass

    def hear(self, text, now):
        return None


def b2_moments():
    print("\n=== B2. Free moments ===")
    t = [T0]
    m = FreeMoments(FreeMomentsConfig(min_gap_s=0, max_per_hour=99), clock=lambda: t[0])
    m.register(Fixed("joke", 0.9))
    m.register(Fixed("mood_check", 0.5))
    check("(the joke wins before)", m.pick(Environment(**FREE)).activity == "joke")
    for _ in range(3):
        m.backoff.note("joke", "rejected")
    check("benched: the next one gets the moment", m.pick(Environment(**FREE)).activity == "mood_check")
    check("status says so", m.status()["backoff"]["benched"] == ["joke"])
    t[0] += 3 * DAY + 1
    check("and the joke is back after", m.pick(Environment(**FREE)).activity == "joke")


def b3_questions():
    print("\n=== B3. Questions ===")
    w = tw.rig()
    w.observe("I told Omar about it")
    for _ in range(3):
        w.backoff.note("person", "rejected")
    check("a benched kind is not asked at a free moment", w.next_question(Environment(**FREE), w.t[0]) is None)
    r = w.hear("any questions for me?", w.t[0] + 5)
    check("asked for, it is still offered", r.outcome == "batch" and "Omar" in r.say, r.say)
    w = tw.rig()
    for _ in range(3):
        w.backoff.note("ambiguous", "rejected")
    w.kb.store_relation("Sara Ali", role="friend")
    w.kb.store_relation("Sara Khan", role="colleague")
    w.observe("I told Sara about it")
    q = w.next_question(Environment(idle_s=60.0), w.t[0])
    check("an urgent kind still gets through", q is not None and q.kind == "ambiguous", str(q))
    p = os.path.join(tempfile.mkdtemp(), "w.json")
    t = [T0]
    a = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), path=p, clock=lambda: t[0])
    for _ in range(3):
        a.backoff.note("person", "rejected")
    a._touch()
    b = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), path=p, clock=lambda: t[0])
    check("kept across a restart", b.backoff.benched("person"))


def b4_runtime():
    print("\n=== B4. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=99),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             moments=True, moments_cfg={"min_gap_s": 0, "max_per_hour": 99,
                                                        "disabled": ["system_assessment"]}),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    rt.moments.set_mood(0.8, clock[0])
    rt.moments.activities["joke"].cooldown_s = 0
    rt.moments.activities["mood_check"].cooldown_s = 0

    def step(dt, idle_s, ticks=1):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def free():
        for _ in range(15):
            step(10, 0.0)
        step(15, 25.0, ticks=2)

    def say(text):
        clock[0] += 3
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    from src.daemon.social import Offer
    for i in range(3):
        o = rt.reactions.register("moment", "", f"joke {i}", clock[0], topic="joke")
        say("no thanks")
    check("(three jokes waved off, through the real tracker)", rt.reactions.counts["rejected"] == 3)
    check("jokes are benched", rt.moments.backoff.benched("joke"), str(rt.moments.backoff.status()))
    check("the mood check is not", not rt.moments.backoff.benched("mood_check"))
    from src.daemon.social import SocialLearner
    rt.gate.learner = SocialLearner()          # the gate's own, broad learning is a separate thing: start it fresh
    rt.gate.learner.clock = lambda: clock[0]
    rt.moments._mood = None
    step(1800, 300.0)                          # the "no thanks" exchange is long over
    n = len(out)
    free()
    step(1, 25.0)                              # the tick that says what the free moment picked
    check("the next free moment is something else", len(out) > n and out[-1].startswith("How are you"),
          str(out[n:]))


if __name__ == "__main__":
    for fn in (b1_score, b2_moments, b3_questions, b4_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
