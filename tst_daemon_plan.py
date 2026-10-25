"""
tst_daemon_plan.py -- the ambient loop following the brain's multi-step plans.

Separate from tst_daemon.py (which exercises attention, privacy and the journal
with a stub brain) because these need the real Brain: a plan is only worth
following if the steps really run, in order, one per tick, and stop when they
should.

  * a plan adopted while answering a person is carried on, one step per tick,
    and each step's result reaches the sink
  * never on the tick that just answered someone; a new request supersedes it
  * a plan born from an unprompted remark is dropped (the remark was budgeted,
    the steps would not be)
  * the plan waits while the user is typing, and is not lost by waiting
  * a stub brain with no planner, a planner that raises, and follow_plans=False
    all leave the loop untouched

Run:  python tst_daemon_plan.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
os.environ["ASH_PLANNING"] = "1"

import numpy as np  # noqa: E402

import tst_brain  # noqa: E402
from tst_planning import make_plan, unit  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


TOOLS = {"get_time": "date_time", "calculate": "calculator"}


class Decision:
    def as_dict(self):
        return {"disposition": "respond"}


class Ash:
    """The slice of the ASH facade the ambient loop touches."""

    def __init__(self, brain):
        self.brain = brain
        self.episodic_memory = None

    def think(self, query, image=None, context=""):
        return self.brain.think(query, context=context)


def ev(text, addressed=True, salience=0.8, urgency=0.0):
    return SensorEvent(source="scripted", modality=Modality.TEXT, text=text,
                       salience=salience, urgency=urgency, addressed=addressed)


def build():
    b = tst_brain.build_brain(tempfile.mkdtemp(prefix="ash_brain_"))
    b.predictive.persist = False
    b.predictive.outcomes.clear()
    b.predictive.goal_protos.clear()
    b.predictive.goal_counts.clear()
    b.goal_rollout = True
    calls = {}
    for _name, entry in b.registry._by_intent.items():
        real = entry.func

        def wrap(q, _n=entry.name, _real=real):
            calls[_n] = calls.get(_n, 0) + 1
            return _real(q)
        entry.func = wrap

    def canned(proposals, latent, predictive, goal, goal_intent=None, confidence=1.0, **kw):
        p = make_plan(["get_time", "calculate", "get_time"], goal=None)
        for st in p.path:
            st.tool_name = TOOLS[st.intent]
        return p
    b.system2.plan = canned
    b.predictive.goal_for = lambda intent, min_obs=2: unit(np.random.default_rng(1), 64)
    return b, calls


def runtime(brain, **cfg):
    delivered = []
    rt = AmbientRuntime(
        ash=Ash(brain), periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()),
        gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=999999, **cfg),
        response_sink=lambda t, m: delivered.append((t, m)))
    return rt, delivered


def main():
    print("\n=== D1. A plan adopted while answering a person is carried on ===")
    b, calls = build()
    rt, out = runtime(b)
    rt.submit(ev("what time is it"))
    rt._tick()
    check("the answer was delivered", len(out) == 1 and out[0][1]["source"] == "scripted",
          str(out[:1]))
    check("only step 1 ran so far", calls.get("date_time") == 1 and not calls.get("calculator"),
          str(calls))
    check("the brain holds the rest of the plan", b.follower.active and b.follower.idx == 1)
    check("no plan step on the tick that answered someone", rt.stats["plan_steps"] == 0)

    rt._tick()
    check("next tick: exactly one more step", rt.stats["plan_steps"] == 1 and len(out) == 2,
          str(rt.stats))
    check("it ran the second tool", calls.get("calculator") == 1 and not calls.get("date_time") == 2,
          str(calls))
    check("the step reached the sink as part of the answer",
          out[1][1]["source"] == "plan" and out[1][1]["addressed"] is True
          and out[1][1]["plan"].get("status") == "step", str(out[1][1].get("plan")))
    check("the step's text is its real output", out[1][0] == "Result: 42", out[1][0])

    rt._tick()
    check("last tick finishes the plan", rt.stats["plan_steps"] == 2 and not b.follower.active
          and out[2][1]["plan"].get("status") == "done", str(out[2][1].get("plan")))
    n = len(out)
    rt._tick()
    rt._tick()
    check("afterwards the loop is quiet", len(out) == n and rt.stats["plan_steps"] == 2)
    check("each tool ran the expected number of times",
          calls == {"date_time": 2, "calculator": 1}, str(calls))
    check("no errors along the way", rt.stats["errors"] == 0)
    check("plan steps count as activity", time.time() - rt._last_activity < 5)

    print("\n=== D2. A new request supersedes the plan ===")
    b, calls = build()
    rt, out = runtime(b)
    rt.submit(ev("what time is it"))
    rt._tick()
    check("plan is waiting", b.follower.active)
    b.system2.plan = lambda *a, **k: None      # the new request needs no plan of its own
    rt.submit(ev("hello there"))
    rt._tick()
    check("the new request replaced it", not b.follower.active
          and b.follower.last_ended == "superseded by a new request", b.follower.last_ended)
    before = dict(calls)
    rt._tick()
    check("none of the old plan's steps ran afterwards", dict(calls) == before and
          rt.stats["plan_steps"] == 0, str(calls))

    print("\n=== D3. A plan born from an unprompted remark is dropped ===")
    b, calls = build()
    rt, out = runtime(b)
    rt._respond(ev("what time is it", addressed=False), Decision())
    check("the remark itself was delivered or silent, but no plan survives",
          not b.follower.active and b.follower.last_ended == "unprompted", b.follower.last_ended)
    n = len(out)
    rt._tick()
    check("so no plan step is ever spoken", rt.stats["plan_steps"] == 0 and len(out) == n)

    print("\n=== D4. The plan waits while the user is typing ===")
    b, calls = build()
    rt, out = runtime(b)
    rt.submit(ev("what time is it"))
    rt._tick()
    rt._user_busy = 1.0
    n = len(out)
    for _ in range(3):
        rt._advance_plan()
    check("held while the user types", rt.stats["plan_steps"] == 0 and len(out) == n
          and b.follower.active, str(rt.stats))
    rt._user_busy = 0.6
    rt._advance_plan()
    check("resumes when they pause", rt.stats["plan_steps"] == 1 and len(out) == n + 1)

    print("\n=== D5. Switched off, or no planner at all ===")
    b, calls = build()
    rt, out = runtime(b, follow_plans=False)
    rt.submit(ev("what time is it"))
    rt._tick()
    rt._tick()
    rt._tick()
    check("follow_plans=False leaves the plan alone", rt.stats["plan_steps"] == 0
          and calls.get("calculator", 0) == 0 and b.follower.active)

    class StubBrain:
        def __init__(self):
            self.homeostasis = type("H", (), {"drives": None})()

    class StubAsh:
        brain = StubBrain()
        episodic_memory = None

        def think(self, q, image=None):
            return None

    rt = AmbientRuntime(ash=StubAsh(), periphery=PeripheryRegistry(),
                        journal=Journal(tempfile.mkdtemp()), response_sink=lambda t, m: None)
    try:
        rt._tick()
        rt._advance_plan()
        survived = True
    except Exception as e:  # noqa: BLE001
        survived = False
        print("    ", repr(e))
    check("a brain without a planner is ignored, not a crash", survived and
          rt.stats["plan_steps"] == 0)

    print("\n=== D6. A planner that raises cannot take the loop down ===")
    b, calls = build()
    rt, out = runtime(b)
    rt.submit(ev("what time is it"))
    rt._tick()

    def boom():
        raise RuntimeError("planner on fire")
    b.advance_plan = boom
    try:
        rt._tick()
        survived = True
    except Exception:  # noqa: BLE001
        survived = False
    check("the exception is contained", survived)
    check("it is counted", rt.stats["errors"] == 1, str(rt.stats))
    check("the broken plan is dropped, so it cannot fail forever",
          not b.follower.active and b.follower.last_ended == "error", b.follower.last_ended)
    rt._tick()
    check("and the next tick is clean", rt.stats["errors"] == 1)

    print("\n=== D7. Real thread: plan finishes on its own, answer first ===")
    b, calls = build()
    rt, out = runtime(b)
    rt.submit(ev("what time is it"))
    rt.start()
    t0 = time.time()
    while time.time() - t0 < 5 and rt.stats["plan_steps"] < 2:
        time.sleep(0.05)
    rt.stop()
    check("both remaining steps ran in the background", rt.stats["plan_steps"] == 2
          and not b.follower.active, str(rt.stats))
    check("answer came before the steps",
          [m["source"] for _t, m in out] == ["scripted", "plan", "plan"],
          str([m["source"] for _t, m in out]))

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
