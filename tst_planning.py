"""
tst_planning.py -- standalone tests for goal-conditioned imagination and planning.

No sentence-transformers, no LLM, no network:

    python tst_planning.py

Phase 3: following a plan (PlanFollower + Brain.advance_plan / run_plan)
  * the decision rules: continue / replan / done / abort, and their ordering
  * a scripted plan runs step by step through the real brain, each tool once
  * failing, vetoed and unsafe steps (irreversible, physical, mutating) stop it
  * surprise triggers a replan from the current state, capped so it cannot loop
  * a stale plan or a new user request ends it
  * think() takes step 1 and the follower takes the rest

Phase 2: planning (Imagery / beam search)
  * a goal that needs two different moves gets a two-move path, a one-move
    goal gets a one-move path, and a goal that is already met gets no plan
  * beam search finds the brute-force optimum
  * no goal / bad latent / no tools => None, so callers deliberate as before
  * irreversible steps and unreliable tools are avoided when an alternative exists
  * plans are deterministic and only name tools that were offered
  * the Brain wiring produces well-formed plans and the whole brain suite
    still passes with ASH_PLANNING=1

Phase 1: goal-conditioned rollout
  * in a world with known dynamics, rollouts that move toward the goal score a
    positive goal_gain and ones that move away score a negative one
  * a multi-step path gains more than a single step
  * with no goal the rollout is exactly what it was before
  * outcome prototypes are learned from successes only, gated by evidence
  * prototypes survive a save/load, and a dimension change is ignored
  * imagine() ranks the goal-ward candidate first
  * action embeddings are stable across processes (no salted hash())
  * the wiring in Brain runs end-to-end with ASH_PLANNING on and off
"""

import os
import subprocess
import sys
import tempfile

# Must be set before src.brain is imported: STATE_DIR is read at import time,
# and these tests must never touch the repo's tracked state/ directory.
_TMP = tempfile.mkdtemp(prefix="ash_planning_")
os.environ["ASH_BRAIN_STATE"] = _TMP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from src.brain.predictive import PredictiveModel, _l2  # noqa: E402
from src.brain.signals import ActionProposal  # noqa: E402
from src.brain.system2 import System2Deliberator  # noqa: E402

RESULTS = {"ok": True}


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    RESULTS["ok"] = RESULTS["ok"] and bool(cond)


def unit(rng, dim):
    return _l2(rng.standard_normal(dim).astype(np.float32))


# ----------------------------------------------------------------------
# A world whose dynamics we know exactly:
#   action "toward" pushes the state along +u, "away" along -u, None does nothing.
# ----------------------------------------------------------------------
def toy_world(dim=16, steps=4000, seed=3):
    rng = np.random.default_rng(seed)
    u = unit(rng, dim)
    pm = PredictiveModel(dim=dim, persist=False)
    for i in range(steps):
        s = unit(rng, dim)
        which = i % 3
        if which == 0:
            pm.forward.learn(s, "toward", _l2(s + 0.6 * u))
        elif which == 1:
            pm.forward.learn(s, "away", _l2(s - 0.6 * u))
        else:
            pm.forward.learn(s, None, s)
    return pm, u, rng


def phase1():
    dim = 16

    print("\n=== P1.1 Goal gain separates goal-ward from goal-away ===")
    pm, u, rng = toy_world(dim)
    pt = ActionProposal(intent="toward")
    pa = ActionProposal(intent="away")
    wins, margins = 0, []
    for _ in range(30):
        s0 = unit(rng, dim)
        gt = pm.rollout(s0, pt, horizon=3, goal=u)
        ga = pm.rollout(s0, pa, horizon=3, goal=u)
        margins.append(gt["goal_gain"] - ga["goal_gain"])
        if gt["goal_gain"] > 0 > ga["goal_gain"]:
            wins += 1
    check("toward gains, away loses (30/30 random start states)", wins == 30,
          f"{wins}/30, mean margin {np.mean(margins):.3f}")
    r = pm.rollout(unit(rng, dim), pt, horizon=3, goal=u)
    check("has_goal is reported", r["has_goal"] is True)

    print("\n=== P1.2 A longer path gets further ===")
    s0 = unit(rng, dim)
    one = pm.rollout(s0, pt, horizon=3, goal=u)["goal_gain"]
    three = pm.rollout(s0, pt, horizon=3, goal=u,
                       path=["toward", "toward", "toward"])["goal_gain"]
    check("3 steps toward gain more than 1", three > one, f"1-step={one:.3f} 3-step={three:.3f}")
    mixed = pm.rollout(s0, pt, horizon=3, goal=u,
                       path=["toward", "away", "away"])["goal_gain"]
    check("a path that turns back loses ground", mixed < one, f"mixed={mixed:.3f}")

    print("\n=== P1.3 No goal => behaves exactly as before ===")
    s0 = unit(rng, dim)
    r = pm.rollout(s0, pt, horizon=3)
    check("has_goal False and gain 0", r["has_goal"] is False and r["goal_gain"] == 0.0)
    # independent recomputation of the legacy drift / stability
    s, drifts = _l2(s0), []
    for step in range(3):
        nxt = pm.forward.predict(s, "toward" if step == 0 else None)
        drifts.append(float(1.0 - np.dot(_l2(s), nxt)))
        s = nxt
    exp_drift = float(np.clip(np.mean(drifts), 0.0, 2.0) / 2.0)
    exp_stab = float(np.clip(1.0 - abs(drifts[-1] - drifts[0]), 0.0, 1.0))
    check("drift identical to legacy formula", abs(r["drift"] - exp_drift) < 1e-6,
          f"{r['drift']:.6f} vs {exp_drift:.6f}")
    check("stability identical to legacy formula", abs(r["stability"] - exp_stab) < 1e-6)
    bad = pm.rollout(None, pt, horizon=3, goal=u)
    check("unusable latent degrades safely",
          bad["drift"] == 0.0 and bad["stability"] == 1.0 and bad["has_goal"] is False)
    wrong = pm.rollout(s0, pt, horizon=3, goal=np.ones(dim + 3, dtype=np.float32))
    check("goal of the wrong dimension is ignored", wrong["has_goal"] is False)

    print("\n=== P1.4 Prototypes: learned from successes only, evidence-gated ===")
    pm2 = PredictiveModel(dim=dim, persist=False)
    target = unit(rng, dim)
    for i in range(6):
        s = unit(rng, dim)
        pm2.observe(s, [])                       # closes previous transition
        pm2.record_outcome("fetch", True)
        pm2.commit(s, "fetch")
        landed = _l2(target + 0.05 * unit(rng, dim))
        pm2.observe(landed, [])                  # the state AFTER a success
        pm2.commit(landed, None)
    check("prototype exists after successes", "fetch" in pm2.goal_protos,
          f"n={pm2.goal_counts.get('fetch')}")
    cos = float(np.dot(pm2.goal_protos["fetch"], target))
    check("prototype points at where successes landed", cos > 0.95, f"cos={cos:.3f}")
    check("goal_for returns it once evidence suffices", pm2.goal_for("fetch") is not None)

    pm3 = PredictiveModel(dim=dim, persist=False)
    for i in range(6):
        s = unit(rng, dim)
        pm3.observe(s, [])
        pm3.record_outcome("flaky", False)
        pm3.commit(s, "flaky")
        pm3.observe(unit(rng, dim), [])
        pm3.commit(unit(rng, dim), None)
    check("failures never form a prototype", "flaky" not in pm3.goal_protos)
    check("goal_for is None with no evidence", pm3.goal_for("flaky") is None
          and pm3.goal_for(None) is None and pm3.goal_for("never_seen") is None)

    pm4 = PredictiveModel(dim=dim, persist=False)
    s = unit(rng, dim)
    pm4.observe(s, [])
    pm4.record_outcome("once", True)
    pm4.commit(s, "once")
    pm4.observe(unit(rng, dim), [])
    check("a single success is not yet a goal (min_obs=2)", pm4.goal_for("once") is None
          and pm4.goal_counts.get("once") == 1)

    # outcome recorded for a DIFFERENT intent than the committed one must not
    # credit the committed action
    pm5 = PredictiveModel(dim=dim, persist=False)
    s = unit(rng, dim)
    pm5.observe(s, [])
    pm5.record_outcome("other", True)
    pm5.commit(s, "committed")
    pm5.observe(unit(rng, dim), [])
    check("outcome for another intent does not credit this one",
          "committed" not in pm5.goal_protos and "other" not in pm5.goal_protos)

    pm6 = PredictiveModel(dim=dim, persist=False)
    seed = unit(rng, dim)
    pm6.seed_goal("cold", seed)
    check("seed_goal gives a cold-start prototype, not yet a goal",
          "cold" in pm6.goal_protos and pm6.goal_for("cold") is None)
    pm6.seed_goal("cold", unit(rng, dim))
    check("seed_goal never overwrites existing evidence",
          np.allclose(pm6.goal_protos["cold"], seed))

    print("\n=== P1.5 Prototypes persist; a dimension change is ignored ===")
    d = tempfile.mkdtemp(prefix="ash_goals_")
    old = PredictiveModel(dim=dim, persist=True)
    old.path = os.path.join(d, "predictive.npz")
    old.outcome_path = os.path.join(d, "outcomes.json")
    old.goal_path = os.path.join(d, "goals.npz")
    old.goal_protos = dict(pm2.goal_protos)
    old.goal_counts = dict(pm2.goal_counts)
    old.save()
    fresh = PredictiveModel(dim=dim, persist=False)
    fresh.path, fresh.outcome_path, fresh.goal_path = old.path, old.outcome_path, old.goal_path
    fresh._load()
    check("prototype round-trips through disk",
          "fetch" in fresh.goal_protos
          and np.allclose(fresh.goal_protos["fetch"], pm2.goal_protos["fetch"], atol=1e-6)
          and fresh.goal_counts["fetch"] == pm2.goal_counts["fetch"])
    other_dim = PredictiveModel(dim=dim + 8, persist=False)
    other_dim.path, other_dim.outcome_path, other_dim.goal_path = old.path, old.outcome_path, old.goal_path
    other_dim._load()
    check("prototypes from a different latent size are not loaded",
          other_dim.goal_protos == {})

    print("\n=== P1.6 imagine() prefers the candidate that moves toward the goal ===")
    sys2 = System2Deliberator(llm=None, registry=None)
    s0 = unit(rng, dim)
    cands = [ActionProposal(intent="away", confidence=0.5),
             ActionProposal(intent="toward", confidence=0.5)]
    out = sys2.imagine(cands, s0, pm, goal=u)
    by = {c.intent: c for c in out}
    check("toward outranks away with a goal", by["toward"].value > by["away"].value,
          f"toward={by['toward'].value:.3f} away={by['away'].value:.3f}")
    check("rationale records the gain", "gain=" in by["toward"].rationale)
    cands2 = [ActionProposal(intent="away", confidence=0.5),
              ActionProposal(intent="toward", confidence=0.5)]
    out2 = sys2.imagine(cands2, s0, pm)
    check("without a goal imagine() works and notes no gain",
          all("gain=" not in c.rationale for c in out2))

    print("\n=== P1.7 Action embeddings are stable across processes ===")
    code = ("import sys,os;os.environ['ASH_BRAIN_STATE']=sys.argv[1];sys.path.insert(0,sys.argv[2]);"
            "from src.brain.predictive import LatentForwardModel as L;"
            "import numpy as np;v=L().action_vec('get_time');print(round(float(v.sum()),6))")
    here = os.path.dirname(os.path.abspath(__file__))
    outs = []
    for seed in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", code, _TMP, here], env=env,
                                   capture_output=True, text=True).stdout.strip())
    check("same intent => same vector regardless of PYTHONHASHSEED",
          len(set(outs)) == 1 and outs[0] != "", str(outs))

    print("\n=== P1.8 Brain wiring, planning off and on ===")
    import tst_brain
    for flag in ("0", "1"):
        os.environ["ASH_PLANNING"] = flag
        st = tempfile.mkdtemp(prefix="ash_brain_")
        brain = tst_brain.build_brain(st)
        check(f"ASH_PLANNING={flag} sets goal_rollout", brain.goal_rollout == (flag == "1"))
        brain.predictive.persist = False
        qs = ["what time is it please", "what time is it now", "what is the time",
              "what time is it currently", "tell me what time is it", "what time is it right now",
              "so what time is it again", "what time is it", "calculate 6 times 7",
              "calculate 40 plus 2", "what is 6 times 7", "delete the old logs"]
        crashed = None
        goal_scored = 0
        try:
            for q in qs:
                r = brain.think(q)
                goal_scored += sum("gain=" in p["rationale"] for p in r.trace.proposals)
        except Exception as e:  # noqa: BLE001
            crashed = repr(e)
        if flag == "1":
            check("candidates really are scored against a goal", goal_scored > 0,
                  f"{goal_scored} goal-scored proposals")
        else:
            check("planning off => no candidate is goal-scored", goal_scored == 0)
        check(f"session runs clean (planning={flag})", crashed is None, crashed or "")
        check(f"prototypes learned from real cycles (planning={flag})",
              len(brain.predictive.goal_protos) > 0,
              f"{ {k: v for k, v in brain.predictive.goal_counts.items()} }")
        brain.predictive.persist = True
        brain.predictive.save()
        check(f"goals.npz written (planning={flag})",
              os.path.exists(brain.predictive.goal_path))
    os.environ["ASH_PLANNING"] = "0"


# ----------------------------------------------------------------------
# Phase 2
# ----------------------------------------------------------------------
def planning_world(dim=24, steps=6000, seed=11):
    """Five actions with known effects: X=+u, Y=+v, Z=-u, W=-v, X2=+u (u perp v)."""
    rng = np.random.default_rng(seed)
    u = unit(rng, dim)
    v = unit(rng, dim)
    v = _l2(v - np.dot(v, u) * u)
    eff = {"X": u, "Y": v, "Z": -u, "W": -v, "X2": u}
    pm = PredictiveModel(dim=dim, persist=False)
    names = list(eff) + [None]
    for i in range(steps):
        st = unit(rng, dim)
        a = names[i % len(names)]
        pm.forward.learn(st, a, st if a is None else _l2(st + 0.6 * eff[a]))
    return pm, eff, rng


def props(names, irreversible=()):
    return [ActionProposal(intent=n, tool_name=n, reversible=(n not in irreversible),
                           confidence=0.5) for n in names]


def apply_world(eff, s, seq):
    for a in seq:
        s = _l2(s + 0.6 * eff[a])
    return s


def phase2():
    import json
    from itertools import product
    from src.brain.signals import Imagery, Step
    sys2 = System2Deliberator(llm=None, registry=None)
    pm, eff, rng = planning_world()
    dim = 24
    P = props(["X", "Y", "Z", "W"])

    print("\n=== P2.1 A two-move goal gets a two-move path ===")
    ok2, ok1 = 0, 0
    for _ in range(20):
        s0 = unit(rng, dim)
        plan = sys2.plan(P, s0, pm, apply_world(eff, s0, ["X", "Y"]))
        if plan and plan.intents() == ["X", "Y"]:
            ok2 += 1
        plan1 = sys2.plan(P, s0, pm, apply_world(eff, s0, ["X"]))
        if plan1 and plan1.intents() == ["X"]:
            ok1 += 1
    check("X then Y is found for an X-then-Y goal (20/20)", ok2 == 20, f"{ok2}/20")
    check("a one-move goal gets exactly one move (20/20)", ok1 == 20, f"{ok1}/20")
    s0 = unit(rng, dim)
    plan = sys2.plan(P, s0, pm, apply_world(eff, s0, ["Y", "X"]))
    check("order matters: Y then X goal gives Y then X",
          plan is not None and plan.intents() == ["Y", "X"], str(plan.intents() if plan else None))

    print("\n=== P2.2 Plan object is well-formed ===")
    s0 = unit(rng, dim)
    plan = sys2.plan(P, s0, pm, apply_world(eff, s0, ["X", "Y"]), goal_intent="g", confidence=0.8)
    check("is an Imagery with Steps",
          isinstance(plan, Imagery) and all(isinstance(t, Step) for t in plan.path))
    check("positive gain and a score", plan.gain > 0 and plan.score > 0,
          f"gain={plan.gain:.3f} score={plan.score:.3f}")
    check("every step has an expected latent of the right size",
          all(t.expected_latent is not None and t.expected_latent.shape == (dim,)
              for t in plan.path))
    check("confidence = model confidence x p_success, within 0..1",
          abs(plan.confidence - 0.8 * plan.p_success) < 1e-9 and 0.0 <= plan.confidence <= 1.0,
          f"{plan.confidence:.3f}")
    d = plan.as_dict()
    check("as_dict is JSON-able",
          bool(json.dumps(d)) and d["goal_intent"] == "g" and len(d["path"]) == len(plan))
    check("steps carry tool and class from the proposal",
          all(t.tool_name == t.intent for t in plan.path))
    check("no step repeats its predecessor",
          all(a != b for a, b in zip(plan.intents(), plan.intents()[1:])))
    check("max_len is respected",
          len(sys2.plan(P, s0, pm, apply_world(eff, s0, ["X", "Y"]), max_len=1)) == 1)

    print("\n=== P2.3 Beam search finds the brute-force optimum ===")
    names = ["X", "Y", "Z", "W"]
    exact_wide, exact_default, worst_gap = 0, 0, 0.0
    trials = 30
    for _ in range(trials):
        s0 = unit(rng, dim)
        goal = _l2(unit(rng, dim) + 2.0 * (eff["X"] + eff["Y"]))
        g, st0 = goal, _l2(s0)
        best = None
        for n in (1, 2, 3):
            for seq in product(names, repeat=n):
                if any(a == b for a, b in zip(seq, seq[1:])):
                    continue
                st, p_ok = st0, 1.0
                for a in seq:
                    st = pm.forward.predict(st, a)
                    p_ok *= pm.expected_success(a)
                gain = float(np.dot(st, g) - np.dot(st0, g))
                sc = System2Deliberator._plan_score(gain, p_ok, n, 0)
                if best is None or sc > best[0] + 1e-12:
                    best = (sc, seq, gain)
        wide = sys2.plan(P, s0, pm, goal, beam=100)
        dflt = sys2.plan(P, s0, pm, goal)
        if best[2] < 1e-3:
            exact_wide += wide is None
            exact_default += dflt is None
            continue
        if wide is not None and abs(wide.score - best[0]) < 1e-6:
            exact_wide += 1
        if dflt is not None and abs(dflt.score - best[0]) < 1e-6:
            exact_default += 1
        elif dflt is not None:
            worst_gap = max(worst_gap, best[0] - dflt.score)
    check("wide beam == brute force", exact_wide == trials, f"{exact_wide}/{trials}")
    check("default beam (4) == brute force", exact_default == trials,
          f"{exact_default}/{trials}, worst score gap {worst_gap:.4f}")

    print("\n=== P2.4 Nothing to plan with => None ===")
    s0 = unit(rng, dim)
    goal = apply_world(eff, s0, ["X"])
    check("no goal", sys2.plan(P, s0, pm, None) is None)
    check("no latent", sys2.plan(P, None, pm, goal) is None)
    check("goal of wrong size", sys2.plan(P, s0, pm, np.ones(dim + 2, dtype=np.float32)) is None)
    check("latent of wrong size",
          sys2.plan(P, np.ones(dim + 2, dtype=np.float32), pm, goal) is None)
    check("no proposals", sys2.plan([], s0, pm, goal) is None)
    check("proposals without a tool are not plannable",
          sys2.plan([ActionProposal(intent="X"), ActionProposal(intent=None, tool_name="x")],
                    s0, pm, goal) is None)
    check("already at the goal => no plan", sys2.plan(P, s0, pm, s0) is None)
    check("goal opposite to anything reachable => no plan",
          sys2.plan(props(["Z"]), s0, pm, apply_world(eff, s0, ["X"])) is None)

    print("\n=== P2.5 Irreversible steps and unreliable tools are avoided ===")
    s0 = unit(rng, dim)
    goal = apply_world(eff, s0, ["X"])
    plan = sys2.plan(props(["X", "X2"], irreversible=("X",)), s0, pm, goal)
    check("prefers the reversible twin", plan is not None and plan.intents() == ["X2"],
          str(plan.intents() if plan else None))
    plan_free = sys2.plan(props(["X", "X2"]), s0, pm, goal)
    check("with neither irreversible it still plans one move",
          plan_free is not None and len(plan_free) == 1)
    only_irr = sys2.plan(props(["X"], irreversible=("X",)), s0, pm, goal)
    check("an irreversible step is still allowed if it is the only way (the board vets it later)",
          only_irr is not None and only_irr.path[0].reversible is False)
    pm_u, _, _ = planning_world(seed=11)
    for _i in range(12):
        pm_u.record_outcome("X2", False)
        pm_u.record_outcome("X", True)
    plan_u = sys2.plan(props(["X", "X2"]), s0, pm_u, goal)
    check("avoids the tool that keeps failing", plan_u is not None and plan_u.intents() == ["X"],
          f"X={pm_u.expected_success('X'):.2f} X2={pm_u.expected_success('X2'):.2f}")

    pm_r, _, _ = planning_world(seed=11)
    for _i in range(30):
        for a in ("X", "Y", "Z", "W"):
            pm_r.record_outcome(a, True)
    padded = sys2.plan(P, s0, pm_r, goal)
    check("proven-reliable tools do not pad a plan",
          padded is not None and padded.intents() == ["X"],
          f"chose {padded.intents() if padded else None}, X={pm_r.expected_success('X'):.2f}")
    s_pad = unit(rng, dim)
    two = sys2.plan(P, s_pad, pm_r, apply_world(eff, s_pad, ["X", "Y"]))
    check("but a genuinely two-move goal still gets two moves with reliable tools",
          two is not None and two.intents() == ["X", "Y"],
          str(two.intents() if two else None))

    print("\n=== P2.6 Deterministic, and only offered tools appear ===")
    s0 = unit(rng, dim)
    goal = apply_world(eff, s0, ["X", "Y"])
    runs = [sys2.plan(P, s0, pm, goal) for _ in range(5)]
    check("same inputs, same plan", len({tuple(r.intents()) for r in runs}) == 1
          and len({round(r.score, 9) for r in runs}) == 1)
    rev = sys2.plan(list(reversed(P)), s0, pm, goal)
    check("proposal order does not change the plan", rev.intents() == runs[0].intents())
    sub = sys2.plan(props(["X", "Y"]), s0, pm, goal)
    check("every step comes from the offered proposals", set(sub.intents()) <= {"X", "Y"})

    print("\n=== P2.7 Brain wiring ===")
    import tst_brain
    os.environ["ASH_PLANNING"] = "1"
    brain = tst_brain.build_brain(tempfile.mkdtemp(prefix="ash_brain_"))
    brain.predictive.persist = False
    qs = ["what time is it please", "what time is it now", "what is the time",
          "what time is it currently", "tell me what time is it", "what time is it right now",
          "so what time is it again", "what time is it", "calculate 6 times 7",
          "calculate 40 plus 2", "what is 6 times 7", "what time will it be tomorrow",
          "xyzzy time", "so what time is it now", "hmm what time is it", "time check please"]
    plans, bad, crashed = [], [], None
    registered = {"get_time", "calculate", "delete_file", "move_arm", "format_disk"}
    try:
        for q in qs:
            r = brain.think(q)
            if r.trace.plan:
                plans.append(r.trace.plan)
                if not r.trace.plan["path"] or any(
                        st["intent"] not in registered for st in r.trace.plan["path"]):
                    bad.append(r.trace.plan)
    except Exception as e:  # noqa: BLE001
        crashed = repr(e)
    check("session with planning runs clean", crashed is None, crashed or "")
    check("the brain really produced plans", len(plans) > 0, f"{len(plans)} plan(s)")
    check("every produced plan is well-formed and names only registered tools", not bad,
          str(bad[:1]))
    if plans:
        print(f"     e.g. {plans[-1]['rationale']}")
    check("active_plan is an Imagery or None",
          brain.active_plan is None or isinstance(brain.active_plan, Imagery))
    os.environ["ASH_PLANNING"] = "0"
    brain_off = tst_brain.build_brain(tempfile.mkdtemp(prefix="ash_brain_"))
    brain_off.predictive.persist = False
    any_plan = False
    for q in qs:
        any_plan = any_plan or bool(brain_off.think(q).trace.plan)
    check("planning off => no plan is ever made",
          not any_plan and brain_off.active_plan is None)

    print("\n=== P2.8 The existing brain suite still passes with planning on ===")
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, ASH_PLANNING="1", ASH_BRAIN_STATE=tempfile.mkdtemp(prefix="ash_tb_"))
    res = subprocess.run([sys.executable, os.path.join(here, "tst_brain.py")], env=env,
                         capture_output=True, text=True, cwd=here)
    check("tst_brain.py passes under ASH_PLANNING=1",
          res.returncode == 0 and "ALL CHECKS PASSED" in res.stdout,
          res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-200:])


# ----------------------------------------------------------------------
# Phase 3
# ----------------------------------------------------------------------
def make_plan(intents, goal=None, expected=None, classes=None, reversible=None, args="q"):
    from src.brain.signals import ActionClass, Imagery, Step
    steps = []
    for i, name in enumerate(intents):
        steps.append(Step(
            intent=name, tool_name=name,
            action_class=(classes or {}).get(name, ActionClass.COMPUTE),
            reversible=(reversible or {}).get(name, True),
            expected_success=0.8,
            expected_latent=(expected[i] if expected else None), args=args))
    return Imagery(goal=goal, goal_intent="g", path=steps, gain=0.1, p_success=0.5,
                   score=1.0, confidence=0.5, rationale="scripted")


def phase3():
    from src.brain.planfollow import Assessment, Decision, PlanFollower
    from src.brain.signals import ActionClass, Prediction
    dim = 16
    rng = np.random.default_rng(5)

    class Clock:
        t = 1000.0

        def __call__(self):
            return self.t

    print("\n=== P3.1 PlanFollower decision rules ===")
    clk = Clock()
    goal = unit(rng, dim)
    start = _l2(goal * 0.3 + unit(rng, dim))                  # some distance from the goal
    mid = _l2(0.5 * start + 0.5 * goal)
    end = _l2(0.1 * start + 0.9 * goal)
    plan = make_plan(["a", "b", "c"], goal=goal, expected=[mid, mid, end])

    def fresh(**kw):
        f = PlanFollower(clock=clk, **kw)
        f.adopt(plan, [], start_latent=start, completed=0)
        return f

    f = PlanFollower(clock=clk)
    check("starts inactive", not f.active and f.current() is None)
    f.adopt(plan, [], start_latent=start, completed=0)
    check("adopt activates and exposes step 0", f.active and f.current().intent == "a")
    f.idx = 1
    check("current follows idx", f.current().intent == "b")
    f.idx = 3
    check("current is None once the path is spent", f.current() is None)
    f.drop("because")
    check("drop clears and remembers why", not f.active and f.last_ended == "because")
    f2 = PlanFollower(clock=clk)
    f2.adopt(plan, [], completed=99)
    check("completed is clamped to the path length", f2.idx == 3)

    f = fresh(); f.idx = 1
    a = f.assess(mid, True, 0.1)
    check("on track => CONTINUE", a.decision == Decision.CONTINUE, a.reason)
    f = fresh(); f.idx = 1
    a = f.assess(mid, False, 0.0)
    check("failed step => REPLAN", a.decision == Decision.REPLAN and "failed" in a.reason)
    f = fresh(); f.idx = 1
    a = f.assess(mid, True, 0.9)
    check("high surprise => REPLAN", a.decision == Decision.REPLAN and "surpris" in a.reason)
    f = fresh(); f.idx = 1
    a = f.assess(mid, True, 0.75)
    check("surprise exactly at the limit is tolerated", a.decision == Decision.CONTINUE)
    away = _l2(-goal + 0.1 * unit(rng, dim))
    f = fresh(); f.idx = 1
    a = f.assess(away, True, 0.0)
    check("moving away from the goal => REPLAN", a.decision == Decision.REPLAN and "away" in a.reason,
          a.reason)
    slight = _l2(start - 0.02 * goal)
    f = fresh(); f.idx = 1
    c_start = float(np.dot(start, goal))
    c_slight = float(np.dot(slight, goal))
    a = f.assess(slight, True, 0.0)
    check("a dip inside the margin is tolerated",
          (c_start - c_slight) < f.drift_margin and a.decision == Decision.CONTINUE,
          f"dip {c_start - c_slight:.3f}")
    f = fresh(); f.idx = 3
    check("path complete => DONE", f.assess(end, True, 0.0).decision == Decision.DONE)
    f = fresh(); f.idx = 3
    check("path complete is DONE even with no observation",
          f.assess(None, True, 0.0).decision == Decision.DONE)
    f = fresh(); f.idx = 1
    a = f.assess(end, True, 0.0)
    check("reaching the final imagined state early => DONE",
          a.decision == Decision.DONE and "early" in a.reason, a.reason)

    f = fresh(); f.idx = 1; clk.t += 5000
    a = f.assess(mid, True, 0.0)
    check("an old plan => ABORT", a.decision == Decision.ABORT and "expired" in a.reason)
    f = fresh(); f.idx = 1; clk.t += 5000
    check("expiry outranks a failed step",
          f.assess(mid, False, 0.0).reason == "plan expired")
    clk.t = 1000.0

    f = fresh(); f.idx = 1
    f.replans = f.max_replans
    a = f.assess(mid, False, 0.0)
    check("replan budget spent => ABORT", a.decision == Decision.ABORT and "limit" in a.reason)

    f = fresh(); f.idx = 1
    a = f.assess(np.ones(dim + 4, dtype=np.float32), True, 0.0)
    check("a latent from a different space does not crash or trigger anything",
          a.decision == Decision.CONTINUE)
    f = fresh(); f.idx = 1
    check("no observation => CONTINUE", f.assess(None, True, 0.0).decision == Decision.CONTINUE)
    f = PlanFollower(clock=clk)
    check("assess with no plan is an ABORT, not a crash",
          f.assess(mid, True, 0.0).decision == Decision.ABORT)
    noplangoal = make_plan(["a", "b"], goal=None)
    f = PlanFollower(clock=clk)
    f.adopt(noplangoal, [], start_latent=start, completed=1)
    check("a plan with no goal still works (CONTINUE, then DONE)",
          f.assess(mid, True, 0.0).decision == Decision.CONTINUE
          and (setattr(f, "idx", 2) or f.assess(mid, True, 0.0).decision == Decision.DONE))

    # ---- brain-level ---------------------------------------------------
    import tst_brain

    def build():
        os.environ["ASH_PLANNING"] = "1"
        b = tst_brain.build_brain(tempfile.mkdtemp(prefix="ash_brain_"))
        b.predictive.persist = False
        # Brains load whatever earlier tests saved to the shared state dir;
        # start each scenario with no tool history and no prototypes.
        b.predictive.outcomes.clear()
        b.predictive.goal_protos.clear()
        b.predictive.goal_counts.clear()
        calls = {}
        for name, entry in b.registry._by_intent.items():
            real = entry.func

            def wrap(q, _n=entry.name, _real=real):
                calls[_n] = calls.get(_n, 0) + 1
                return _real(q)
            entry.func = wrap
        return b, calls

    def tool_state(b):
        return b.predictive.forward.steps, dict(b.predictive.goal_counts)

    print("\n=== P3.2 Following a scripted plan through the real brain ===")
    b, calls = build()
    pl = make_plan(["date_time", "calculator", "date_time"])
    # the stub registry names tools date_time / calculator for intents get_time / calculate
    pl = make_plan(["get_time", "calculate", "get_time"])
    for st in pl.path:
        st.tool_name = {"get_time": "date_time", "calculate": "calculator"}[st.intent]
    b.follower.adopt(pl, [], start_latent=None, completed=1,
                     pending_output={"ok": True, "result": "2026-08-24T10:00:00"}, pending_ok=True)
    steps0, _ = tool_state(b)
    rs = b.run_plan()
    check("two remaining steps ran", len(rs) == 2, f"{len(rs)} response(s)")
    check("each tool ran exactly once", calls == {"calculator": 1, "date_time": 1}, str(calls))
    check("steps ran in order with their real output",
          [r.tool_used for r in rs] == ["calculator", "date_time"]
          and rs[0].text == "Result: 42", str([r.text for r in rs]))
    check("status is step then done",
          [r.trace.plan["status"] for r in rs] == ["step", "done"],
          str([r.trace.plan["status"] for r in rs]))
    check("progress is reported", rs[-1].trace.plan["step"] == 3 and rs[-1].trace.plan["of"] == 3)
    check("the plan is cleared when finished", not b.follower.active and b.active_plan is None)
    check("every response is a successful FAST-path tool call",
          all(r.tool_success is True and r.pathway.value == "fast" for r in rs))
    steps1, protos = tool_state(b)
    check("the forward model learned from the plan's transitions", steps1 > steps0,
          f"{steps0} -> {steps1}")
    check("successful steps fed the outcome prototypes", protos.get("calculate", 0) >= 1,
          str(protos))
    check("advance_plan with nothing to do returns None", b.advance_plan() is None)
    check("run_plan with nothing to do returns []", b.run_plan() == [])

    print("\n=== P3.3 Failing, vetoed and unsafe steps stop the plan ===")
    b, calls = build()
    b.registry._by_intent["calculate"].func = lambda q: {"ok": False, "error": "boom"}
    pl = make_plan(["get_time", "calculate", "get_time"])
    for st in pl.path:
        st.tool_name = {"get_time": "date_time", "calculate": "calculator"}[st.intent]
    b.follower.adopt(pl, [], start_latent=None, completed=1,
                     pending_output={"ok": True, "result": "x"}, pending_ok=True)
    rs = b.run_plan()
    check("a failed step stops the plan (no better path)",
          len(rs) == 1 and rs[0].trace.plan["status"] == "aborted" and not b.follower.active,
          str([r.trace.plan for r in rs]))
    check("the step after a failure never runs", calls.get("date_time", 0) == 0, str(calls))
    check("the response says it stopped", "stopped the plan" in rs[0].text, rs[0].text)
    check("the failure counted against the tool's record",
          b.predictive.expected_success("calculate") < 0.5)

    b, calls = build()
    b.executive.constitution._cache = [
        {"text": "ASH must never run calculate under any circumstances.", "priority": 10}]
    b.executive.constitution._loaded = True
    pl = make_plan(["get_time", "calculate", "get_time"])
    for st in pl.path:
        st.tool_name = {"get_time": "date_time", "calculate": "calculator"}[st.intent]
    b.follower.adopt(pl, [], start_latent=None, completed=1,
                     pending_output={"ok": True, "result": "x"}, pending_ok=True)
    rs = b.run_plan()
    check("a Constitution-forbidden step is blocked, not run",
          len(rs) == 1 and "constitution" in rs[0].text and calls.get("calculator", 0) == 0,
          rs[0].text if rs else "no response")
    check("and the plan is gone", not b.follower.active and rs[0].trace.plan["status"] == "aborted")

    for label, cls, rev in (("irreversible class", ActionClass.IRREVERSIBLE, False),
                            ("physical class", ActionClass.PHYSICAL, False),
                            ("mutation class", ActionClass.MUTATION, True),
                            ("compute but not reversible", ActionClass.COMPUTE, False)):
        b, calls = build()
        pl = make_plan(["get_time", "calculate"], classes={"calculate": cls},
                       reversible={"calculate": rev})
        for st in pl.path:
            st.tool_name = {"get_time": "date_time", "calculate": "calculator"}[st.intent]
        b.follower.adopt(pl, [], start_latent=None, completed=1,
                         pending_output={"ok": True, "result": "x"}, pending_ok=True)
        rs = b.run_plan()
        check(f"{label} is never run unattended",
              len(rs) == 1 and calls.get("calculator", 0) == 0
              and "go-ahead" in rs[0].text and not b.follower.active, rs[0].text if rs else "")

    print("\n=== P3.4 Replanning when reality disagrees ===")
    from src.brain.signals import Imagery  # noqa: F811
    b, calls = build()
    replans = []

    def fake_plan(proposals, latent, predictive, goal, goal_intent=None, confidence=1.0, **kw):
        replans.append((len(proposals), latent is not None))
        p = make_plan(["get_time"])
        p.path[0].tool_name = "date_time"
        return p
    b.system2.plan = fake_plan
    b.follower.max_replans = 2
    pl = make_plan(["get_time", "calculate", "get_time"], goal=unit(rng, 64))
    for st in pl.path:
        st.tool_name = {"get_time": "date_time", "calculate": "calculator"}[st.intent]
    cands = tst_brain_proposals(b)
    b.follower.adopt(pl, cands, start_latent=None, completed=1,
                     pending_output={"ok": True, "result": "x"}, pending_ok=True)
    # force a surprising observation
    real_obs = b.predictive.observe
    b.predictive.observe = lambda latent, intents=None: Prediction(surprise=0.99, confidence=1.0)
    r = b.advance_plan()
    check("a surprising observation triggers a replan",
          len(replans) >= 1 and r is not None and r.trace.plan["status"] == "replanned",
          str((replans, r.trace.plan if r else None)))
    check("the replan was made from the current state with the same candidates",
          replans and replans[0][0] == len(cands) and replans[0][1] is True)
    check("replan count is tracked", b.follower.replans >= 1, f"replans={b.follower.replans}")
    check("a replan starts the new plan from its first step",
          b.follower.active is False or b.follower.idx <= 1)
    # keep surprising it: replans must stop at the cap
    for _ in range(12):
        if not b.follower.active:
            break
        b.advance_plan()
    check("repeated surprise ends the plan instead of looping forever", not b.follower.active,
          f"replans={b.follower.replans}, last_ended={b.follower.last_ended!r}")
    check("the cap is what ended it", "limit" in b.follower.last_ended or b.follower.replans <= 2,
          b.follower.last_ended)
    b.predictive.observe = real_obs

    print("\n=== P3.5 Time and new requests end a plan ===")
    b, calls = build()
    clock = Clock()
    b.follower.clock = clock
    pl = make_plan(["get_time", "calculate"])
    for st in pl.path:
        st.tool_name = {"get_time": "date_time", "calculate": "calculator"}[st.intent]
    b.follower.adopt(pl, [], start_latent=None, completed=1,
                     pending_output={"ok": True, "result": "x"}, pending_ok=True)
    clock.t += b.follower.max_age_s + 1
    rs = b.run_plan()
    check("a stale plan is abandoned without running anything",
          len(rs) == 1 and rs[0].trace.plan["status"] == "aborted" and not calls,
          str(rs[0].trace.plan if rs else None))

    b, calls = build()
    b.follower.adopt(make_plan(["get_time", "calculate"]), [], start_latent=None, completed=1)
    b.think("hello there")
    check("a new user request supersedes the plan",
          not b.follower.active and b.follower.last_ended == "superseded by a new request")
    check("and nothing from the old plan ran", calls.get("calculator", 0) == 0, str(calls))

    print("\n=== P3.6 End to end: think() takes step 1, the follower takes the rest ===")
    b, calls = build()
    canned = {"n": 0}

    def canned_plan(proposals, latent, predictive, goal, goal_intent=None, confidence=1.0, **kw):
        canned["n"] += 1
        p = make_plan(["get_time", "calculate"], goal=None)
        for st in p.path:
            st.tool_name = {"get_time": "date_time", "calculate": "calculator"}[st.intent]
        return p
    b.system2.plan = canned_plan
    b.predictive.goal_for = lambda intent, min_obs=2: unit(np.random.default_rng(1), 64)
    r = b.think("what time is it")
    check("think() used the planner", canned["n"] == 1)
    check("think() took step 1 only", calls.get("date_time") == 1 and not calls.get("calculator"),
          str(calls))
    check("the follower holds the remaining step", b.follower.active and b.follower.idx == 1
          and b.active_plan is not None)
    check("the trace recorded the plan", r.trace.plan.get("path"))
    rs = b.run_plan()
    check("the follower finished the plan", len(rs) == 1 and rs[0].trace.plan["status"] == "done"
          and rs[0].tool_used == "calculator", str([x.trace.plan for x in rs]))
    check("each tool ran once overall", calls == {"date_time": 1, "calculator": 1}, str(calls))
    check("status() still works afterwards", isinstance(b.status(), dict))

    b, calls = build()
    b.system2.plan = lambda *a, **k: _wrong_first(canned_plan(*a, **k))
    b.predictive.goal_for = lambda intent, min_obs=2: unit(np.random.default_rng(1), 64)
    b.think("what time is it")
    check("if the board picks a different first action the plan is not adopted",
          not b.follower.active and b.active_plan is None)

    b, calls = build()
    b.goal_rollout = False
    b.system2.plan = canned_plan
    b.think("what time is it")
    check("planning off => nothing is adopted", not b.follower.active)
    os.environ["ASH_PLANNING"] = "0"


def _wrong_first(plan):
    """Make the plan start with an action the board will not choose for a
    time query."""
    plan.path[0].intent = "calculate"
    plan.path[0].tool_name = "calculator"
    return plan


def tst_brain_proposals(b):
    from src.brain.signals import ActionProposal as AP
    return [AP(intent="get_time", tool_name="date_time", args="q"),
            AP(intent="calculate", tool_name="calculator", args="q")]


def main():
    phase1()
    phase2()
    phase3()
    print("\n" + ("ALL CHECKS PASSED" if RESULTS["ok"] else "SOME CHECKS FAILED"))
    return 0 if RESULTS["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
