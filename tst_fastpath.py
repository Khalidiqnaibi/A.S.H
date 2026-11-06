"""
tst_fastpath.py -- why every request went to the language model, and the fix.

On 2026-11-06 all 31 replies took the slow path -- even "what's 15 times 23"
(calculate at 0.85) -- because of three feedback loops:

  F1  The critic was trained with -0.4 on every turn that went slow, tool or
      not. Talk is most turns, so it learned "everything is worth -0.4", its
      values sank, more turns went slow... Now a turn that ran no tool does
      not train it; a tool that worked (fast or slow) is rewarded, one that
      failed is not.
  F2  A critic saved under that reward starts again from the warm-start
      priors; one saved under the new reward is kept.
  F3  "Outcome surprise" rose with every first successful use of a tool,
      because the untried 0.5 counted as a prediction. Now surprise is
      weighted by the evidence behind the prediction.
  F4  The novelty ceiling no longer sends a confident match on a harmless,
      reversible tool to the model; consequential or weak matches still go.
  F5  End to end, with the warm-start critic: confident tool requests take
      the fast path, weak ones and talk still deliberate.

Run:  PYTHONIOENCODING=utf-8 python tst_fastpath.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import numpy as np  # noqa: E402

import tst_wondering as tw  # noqa: E402
from src.brain import system1 as S1  # noqa: E402
from src.brain.signals import ActionClass, ActionProposal, Drives, PerceptBundle, Prediction  # noqa: E402

check = tw.check


def policy():
    return S1.System1Policy(classifier=lambda q: {}, registry=type("R", (), {"match": lambda self, i: None})(),
                            persist=False)


def staged(p, tool_success, escalated, query="what's 15 times 23"):
    prop = ActionProposal(intent="calculate", tool_name="calculate", args=query, confidence=0.85)
    prop.__dict__["_features"] = np.ones(len(S1.FEATURE_NAMES), dtype=np.float32) * 0.5
    p.stage_reward(prop, tool_success=tool_success, escalated=escalated)


def f1_reward():
    print("\n=== F1. The critic's reward ===")
    p = policy()
    w0 = p.critic.w.copy()
    staged(p, tool_success=None, escalated=True, query="tell me about yourself")
    r = p.settle_reward("thanks", None)
    check("a conversation turn (slow, no tool) does not train it", r is None and p.critic.updates == 0
          and np.allclose(p.critic.w, w0))
    staged(p, tool_success=True, escalated=False)
    r = p.settle_reward("great, thanks", None)
    check("a tool that worked on the fast path: rewarded", r == 1.0 and p.critic.updates == 1)
    staged(p, tool_success=True, escalated=True)
    r = p.settle_reward("ok", None)
    check("a tool that worked after deliberating: still rewarded", r is not None and r > 0, str(r))
    staged(p, tool_success=False, escalated=False)
    r = p.settle_reward("ok", None)
    check("a tool that failed: punished", r is not None and r < 0, str(r))
    staged(p, tool_success=None, escalated=False, query="hello")
    r = p.settle_reward("hi", "POSITIVE")
    check("a fast, toolless turn still counts the user's reaction", r == 0.3, str(r))


def f2_reset():
    print("\n=== F2. A critic trained on the old reward ===")
    d = tempfile.mkdtemp()
    old = S1.STATE_DIR
    try:
        p = policy()
        p.path = os.path.join(d, "critic.json")
        bad = [1.123, -0.687, -0.038, -1.009, 0.823, 0.189, -0.547, -0.057, -0.263, 0.234, -0.266]
        json.dump({"w": bad, "updates": 41}, open(p.path, "w"))
        warm = S1.Critic().w.copy()
        p.critic = S1.Critic()
        p._load()
        check("old reward: back to the warm-start priors", np.allclose(p.critic.w, warm) and p.critic.updates == 0)
        json.dump({"w": bad, "updates": 41, "reward_version": S1.REWARD_VERSION}, open(p.path, "w"))
        p.critic = S1.Critic()
        p._load()
        check("new reward: kept", np.allclose(p.critic.w, np.array(bad, dtype=np.float32)) and p.critic.updates == 41)
        p.persist = True
        S1.STATE_DIR = d
        p.save()
        check("saved with the reward version", json.load(open(p.path))["reward_version"] == S1.REWARD_VERSION)
    finally:
        S1.STATE_DIR = old


def f3_surprise():
    print("\n=== F3. Outcome surprise needs a prediction ===")
    from src.brain.predictive import PredictiveModel
    pm = PredictiveModel(dim=16, persist=False)
    for intent in ("calculate", "get_time", "unit_convert", "list_directory", "read_file"):
        pm.record_outcome(intent, True)
    check("first successful uses of tools: no surprise", pm._outcome_err < 0.01, f"{pm._outcome_err:.3f}")
    for _ in range(30):
        pm.record_outcome("get_time", True)
    before = pm._outcome_err
    pm.record_outcome("get_time", False)
    check("a reliable tool failing: surprise", pm._outcome_err > before + 0.1, f"{before:.3f} -> {pm._outcome_err:.3f}")
    pm2 = PredictiveModel(dim=16, persist=False)
    pm2.record_outcome("new_tool", False)
    check("a never-tried tool failing once: hardly any", pm2._outcome_err < 0.05, f"{pm2._outcome_err:.3f}")


def f4_novelty():
    print("\n=== F4. Novelty and a confident, harmless tool ===")
    from src.brain.executive import ExecutiveCore
    ex = ExecutiveCore()
    d = Drives()
    d.effort_budget = 0.5
    pred = Prediction()
    novel = PerceptBundle()
    novel.novelty = 0.85

    def gate(conf, reversible=True, cls=ActionClass.COMPUTE, value=1.0):
        p = ActionProposal(intent="x", tool_name="x", confidence=conf, value=value, reversible=reversible,
                           action_class=cls)
        return ex.gate(p, d, pred, novel, margin=1.0, deadlocked=False)

    path, why = gate(0.85)
    check("confident and harmless: novelty does not send it to the model", path.value == "fast", str(why))
    path, why = gate(0.45)
    check("a weak match on a new topic: deliberates", any("novelty" in r for r in why), str(why))
    path, why = gate(0.85, reversible=False, cls=ActionClass.IRREVERSIBLE)
    check("something that can't be undone: deliberates", path.value == "slow" and any("novelty" in r for r in why),
          str(why))
    calm = PerceptBundle()
    calm.novelty = 0.1
    p = ActionProposal(intent="x", tool_name="x", confidence=0.45, value=1.0, action_class=ActionClass.COMPUTE)
    path, why = ex.gate(p, d, pred, calm, margin=1.0, deadlocked=False)
    check("(no novelty, no novelty reason)", not any("novelty" in r for r in why))


def f5_end_to_end():
    print("\n=== F5. End to end ===")
    from src.brain.executive import ExecutiveCore
    ex = ExecutiveCore()
    p = policy()

    def decide(intent, match, novelty, surprise, p_success=0.5):
        d = Drives()
        pred = Prediction()
        pred.surprise = surprise
        b = PerceptBundle()
        b.novelty = novelty
        x = p.features(match, b, d, pred, p_success, True)
        prop = ActionProposal(intent=intent, tool_name=intent, args="", origin="system1",
                              action_class=ActionClass.COMPUTE, confidence=match, value=p.critic.value(x),
                              expected_success=p_success, reversible=True)
        return ex.arbitrate([prop], b, d, pred)

    # (the numbers from the session: match, novelty, surprise, and the tool's
    # track record -- get_time had worked 78% of the time)
    for intent, match, nov, sur, ps in (("calculate", 0.85, 0.74, 0.25, 0.5), ("get_time", 0.60, 0.77, 0.27, 0.78),
                                        ("system_stats", 0.74, 0.30, 0.27, 0.5),
                                        ("list_directory", 0.65, 0.55, 0.30, 0.5)):
        v = decide(intent, match, nov, sur, ps)
        check(f"{intent} @ {match}: fast", v.pathway.value == "fast", f"{v.score:.3f} {v.escalation_reasons}")
    v = decide("system_stats", 0.30, 0.5, 0.3)
    check("a weak match (0.30): deliberates", v.pathway.value == "slow", f"{v.score:.3f} {v.escalation_reasons}")


if __name__ == "__main__":
    for fn in (f1_reward, f2_reset, f3_surprise, f4_novelty, f5_end_to_end):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
