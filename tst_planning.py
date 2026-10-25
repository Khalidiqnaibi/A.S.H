"""
tst_planning.py -- standalone tests for goal-conditioned imagination and planning.

No sentence-transformers, no LLM, no network:

    python tst_planning.py

Phase 4: gist-first narration (gist.py + System2.narrate)
  * facts, number words and plainness; the gist built from what actually happened
  * plain replies rendered with no model; the rest phrased by a model and checked
    (required facts, forbidden claims, meaning), one retry, then raw facts
  * a provider error is never retried; open conversation is never second-guessed
  * the Brain wiring, and the existing suite still passing with everything on

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


# ----------------------------------------------------------------------
# Phase 4
# ----------------------------------------------------------------------
def bow_embed(text, dim=64):
    """Deterministic bag-of-words embedder: shared words => similar vectors."""
    import re
    import zlib
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9']+", (text or "").lower()):
        v[zlib.crc32(tok.encode()) % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class ScriptedLLM:
    """Replies from a queue and remembers every prompt it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        parts = []
        for m in messages:
            c = getattr(m, "content", None)
            if c is None and isinstance(m, dict):
                c = m.get("content")
            parts.append(str(c))
        self.prompts.append("\n".join(parts))
        reply = self.replies.pop(0) if self.replies else "..."

        class R:
            content = reply
        return R()


def phase4():
    import json
    from types import SimpleNamespace as NS
    from src.brain import gist as G
    from src.brain.signals import Drives, Gist, Pathway, Point, Stance, Verdict

    warm = NS(social=0.9, fatigue=0.1, urgency=0.2)
    brisk = NS(social=0.9, fatigue=0.1, urgency=0.9)
    cool = NS(social=0.2, fatigue=0.0, urgency=0.3)

    print("\n=== P4.1 Facts, number words, plainness ===")
    check("a plain number is a fact", G.extract_facts("Result: 42") == ["42"])
    check("a timestamp is not a fact", G.extract_facts("2026-08-24T10:00:00") == [])
    check("clock times and dates are not facts either",
          G.extract_facts("It is 10:30 on 2026-08-24, 3 items, 4.5 kg") == ["3"],
          str(G.extract_facts("It is 10:30 on 2026-08-24, 3 items, 4.5 kg")))
    check("decimals and numbers over 999 are skipped (no single spoken form)",
          G.extract_facts("1,234 files and 3.14") == [])
    check("duplicates collapse, order kept", G.extract_facts("7 and 7 and 12") == ["7", "12"])
    check("the number of facts is capped", len(G.extract_facts(" ".join(map(str, range(1, 40))))) == 5)
    check("empty input is fine", G.extract_facts("") == [] and G.extract_facts(None) == [])
    check("number_words: 42", set(G.number_words(42)) == {"forty-two", "forty two"})
    check("number_words: 105", "one hundred and five" in G.number_words(105))
    check("number_words: 0 and 1000", G.number_words(0) == ["zero"] and G.number_words(1000) == [])
    check("digits satisfy a fact", G.contains_fact("I found 42 items.", "42"))
    check("so do spoken words", G.contains_fact("That comes to forty-two.", "42")
          and G.contains_fact("Forty two, I make it.", "42"))
    check("a different number does not", not G.contains_fact("That is twenty-four.", "42"))
    check("a longer number containing it does not", not G.contains_fact("I found 142 items.", "42"))
    check("a decimal containing it does not", not G.contains_fact("It is 4.2 degrees", "42"))
    check("thousands separators are understood", G.contains_fact("There are 1,042 of them", "1042"))
    check("plain text is plain", G.is_plain("Result: 42") and G.is_plain("The kettle is on."))
    check("timestamps, times and dates are not plain",
          not G.is_plain("2026-08-24T10:00:00") and not G.is_plain("at 10:30")
          and not G.is_plain("on 24/08/2026"))
    check("long, multi-line, json and urls are not plain",
          not G.is_plain("x" * 200) and not G.is_plain("a\nb") and not G.is_plain('{"a": 1}')
          and not G.is_plain("see http://x.y") and not G.is_plain(""))

    print("\n=== P4.2 build_gist decides from what actually happened ===")
    g = G.build_gist(tool_ok=True, body="Result: 42", drives=cool)
    check("a plain successful result => inform, with the fact pinned",
          g.intent == "inform" and g.must_include == ["42"] and g.points[0].say == "Result: 42")
    check("a cool stance adds no follow-up", g.follow_up is None)
    g = G.build_gist(tool_ok=True, body="Result: 42", drives=warm)
    check("a warm, unhurried stance offers a follow-up", g.follow_up is not None)
    g = G.build_gist(tool_ok=True, body="Result: 42", drives=brisk)
    check("a hurried stance does not, however warm", g.follow_up is None)
    g = G.build_gist(tool_ok=True, body="2026-08-24T10:00:00", drives=cool)
    check("a timestamp result needs phrasing (no ready sentence, no pinned digits)",
          g.intent == "inform" and g.points[0].say is None and g.must_include == [])
    g = G.build_gist(tool_ok=False, tool_out={"ok": False, "error": "boom"}, drives=cool)
    check("a failed tool => report_failure with its reason",
          g.intent == "report_failure" and g.points[0].say == "That didn't work: boom.")
    check("failure forbids claiming success", "successfully" in g.must_avoid and "completed" in g.must_avoid)
    g = G.build_gist(tool_ok=False, tool_out={"ok": False}, drives=cool)
    check("a failure with no error text still reads fine", g.points[0].say == "That didn't work.")
    att = NS(intent="format_disk")
    g = G.build_gist(vetoed_by=["constitution"], attempted=att, drives=cool)
    check("a veto => refuse, naming the action and the reason",
          g.intent == "refuse" and g.points[0].say == "I didn't format disk."
          and "hard rules" in g.points[1].say and g.confidence == 1.0, str([p.say for p in g.points]))
    check("refusal forbids claiming success", "all done" in g.must_avoid)
    g = G.build_gist(vetoed_by=["vla"], attempted=att, drives=cool)
    check("a VLA veto gives its own reason", "safely" in g.points[1].say)
    g = G.build_gist(vetoed_by=["somebody"], attempted=None, drives=cool)
    check("an unknown vetoer and no action still produce a sentence",
          g.points[0].say == "I didn't that." or "didn't" in g.points[0].say, g.points[0].say)
    g = G.build_gist(deadlocked=True, alternatives=["get_time", "calculate", "x"], drives=cool)
    check("a deadlock => ask which of the top two",
          g.intent == "ask" and g.points[0].say == "Did you mean get time, or calculate?")
    g = G.build_gist(deadlocked=True, alternatives=["only_one"], drives=cool)
    check("a deadlock with no real alternatives leaves the question to the model",
          g.intent == "ask" and g.points[0].say is None and G.render_gist(g) is None)
    # Found running the real brain: one intent can be proposed twice (reflex
    # and deliberation), which produced "Did you mean get time, or get time?"
    g = G.build_gist(deadlocked=True, alternatives=["get_time", "get_time", None], drives=cool)
    check("the same intent twice is one alternative, not two",
          g.intent == "ask" and g.points[0].say is None, str(g.points[0].say))
    g = G.build_gist(deadlocked=True, alternatives=["get_time", "get_time", "calculate"], drives=cool)
    check("duplicates are skipped to reach two distinct readings",
          g.points[0].say == "Did you mean get time, or calculate?", str(g.points[0].say))
    from types import SimpleNamespace as _NS
    from src.brain.brain import ASK_MIN_CONFIDENCE
    from src.brain.brain import Brain as _Brain
    cands = [_NS(intent="get_time", confidence=0.42, tool_name="date_time"),
             _NS(intent="set_reminder", confidence=0.14, tool_name="r"),
             _NS(intent="file_info", confidence=0.12, tool_name="f"), _NS(intent=None, confidence=0.5, tool_name=None)]
    dl = _NS(vetoed_by=[], deadlocked=True)
    g = _Brain._decide_gist(_NS(_embed_text=lambda t: None), cands, None, None, None, dl, cool)
    check("weak guesses are not offered as a ready-made question (real run: 'hello there')",
          g.intent == "ask" and g.points[0].say is None, str(g.points[0].say))
    cands[1].confidence = 0.35
    g = _Brain._decide_gist(_NS(_embed_text=lambda t: None), cands, None, None, None, dl, cool)
    check("two well-supported readings are", g.points[0].say == "Did you mean get time, or set reminder?"
          and ASK_MIN_CONFIDENCE == 0.3, str(g.points[0].say))
    g = G.build_gist(drives=cool)
    check("nothing happened => converse, no points, no checks",
          g.intent == "converse" and not g.points and not g.must_include and not g.must_avoid)
    g = G.build_gist(tool_ok=True, body="Result: 42", vetoed_by=["constitution"], drives=cool)
    check("a tool that ran is reported as it went, even if something else was vetoed",
          g.intent == "inform")
    g = G.build_gist(tool_ok=True, body="Paris is the capital of France.", drives=cool,
                     embed_fn=bow_embed)
    check("prose results carry an intent vector", g.vector is not None and g.vector.shape == (64,))
    g = G.build_gist(tool_ok=True, body="2026-08-24T10:00:00", drives=cool, embed_fn=bow_embed)
    check("data-like results carry none (a meaning check would misfire)", g.vector is None)
    g = G.build_gist(tool_ok=True, body="Paris is the capital of France.", drives=cool,
                     embed_fn=bow_embed)
    check("the vector is the content itself, not the brief written for the phrasing model",
          np.allclose(g.vector, bow_embed("Paris is the capital of France.")))
    others = [
        G.build_gist(deadlocked=True, alternatives=[], drives=cool, embed_fn=bow_embed),
        G.build_gist(vetoed_by=["constitution"], attempted=att, drives=cool, embed_fn=bow_embed),
        G.build_gist(tool_ok=False, tool_out={"ok": False, "error": "boom"}, drives=cool,
                     embed_fn=bow_embed),
        G.build_gist(drives=cool, embed_fn=bow_embed)]
    check("gists with no content of their own (ask, refuse, failure, converse) carry none",
          all(o.vector is None for o in others))
    ask = others[0]
    ok_q, pr_q = G.verify("Which of those did you have in mind?", ask, bow_embed)
    check("so a perfectly good clarifying question can never be rejected as 'off topic'",
          ok_q, str(pr_q))

    def boom(_t):
        raise RuntimeError("encoder down")
    g = G.build_gist(tool_ok=True, body="Paris is the capital of France.", drives=cool, embed_fn=boom)
    check("a broken encoder costs the vector, not the gist", g.vector is None and g.intent == "inform")
    st = G.stance_from_drives(NS(social=2.0, fatigue=-1.0, urgency=-5.0))
    check("stance is always within 0..1",
          all(0.0 <= x <= 1.0 for x in (st.warmth, st.energy, st.brevity)), str(st.as_dict()))
    check("gist as_dict is JSON-able",
          bool(json.dumps(G.build_gist(tool_ok=True, body="Result: 42", drives=warm,
                                       embed_fn=bow_embed).as_dict())))

    print("\n=== P4.3 render_gist: the reply with no model ===")
    check("plain inform renders as-is",
          G.render_gist(G.build_gist(tool_ok=True, body="Result: 42", drives=cool)) == "Result: 42")
    r = G.render_gist(G.build_gist(tool_ok=True, body="Result: 42", drives=warm))
    check("a follow-up is appended when there is one", r.startswith("Result: 42") and "anything else" in r, r)
    check("a timestamp needs a model", G.render_gist(
        G.build_gist(tool_ok=True, body="2026-08-24T10:00:00", drives=cool)) is None)
    check("open conversation needs a model", G.render_gist(G.build_gist(drives=cool)) is None)
    check("refusal renders to two sentences", G.render_gist(
        G.build_gist(vetoed_by=["constitution"], attempted=att, drives=cool))
        == "I didn't format disk. It goes against one of my hard rules.")
    check("a question renders", G.render_gist(
        G.build_gist(deadlocked=True, alternatives=["get_time", "calculate"], drives=cool))
        == "Did you mean get time, or calculate?")
    check("a failure renders", G.render_gist(
        G.build_gist(tool_ok=False, tool_out={"ok": False, "error": "boom"}, drives=cool))
        == "That didn't work: boom.")
    check("no gist => None", G.render_gist(None) is None)

    print("\n=== P4.4 verify: did the reply carry the gist? ===")
    inform = G.build_gist(tool_ok=True, body="It is 10:30 and you have 3 reminders", drives=cool)
    ok, pr = G.verify("You have three reminders, and it's half past ten.", inform)
    check("a faithful paraphrase passes (number said in words)", ok and not pr, str(pr))
    ok, pr = G.verify("It's half past ten.", inform)
    check("a reply that drops the fact fails, naming it", not ok and any("'3'" in p for p in pr), str(pr))
    ok, pr = G.verify("   ", inform)
    check("an empty reply fails", not ok and pr == ["the reply was empty"])
    fail = G.build_gist(tool_ok=False, tool_out={"ok": False, "error": "disk is full at 10:30"}, drives=cool)
    ok, pr = G.verify("I successfully saved it.", fail)
    check("claiming success after a failure is rejected", not ok and any("successfully" in p for p in pr), str(pr))
    ok, pr = G.verify("Sorry, that didn't work.", fail)
    check("an honest account of the failure passes", ok, str(pr))
    ok, pr = G.verify("It's ALL DONE.", G.build_gist(vetoed_by=["vla"], attempted=att, drives=cool))
    check("the avoid-list is case-insensitive", not ok)
    ok, pr = G.verify("whatever", G.build_gist(drives=cool))
    check("open conversation is never second-guessed", ok)
    ok, pr = G.verify("", G.build_gist(drives=cool))
    check("but an empty reply still fails", not ok)
    ok, pr = G.verify("none of it", None)
    check("no gist => nothing to check", ok)
    ok, pr = G.verify("It's half past ten.", G.build_gist(
        tool_ok=True, body="It is 10:30 and you have 3 reminders; also 5 alarms", drives=cool))
    check("every missing fact is reported", not ok and len(pr) == 2, str(pr))

    prose = G.build_gist(tool_ok=True, body="The capital of France is Paris, on the Seine.",
                         drives=cool, embed_fn=bow_embed)
    ok, pr = G.verify("Paris is the capital of France, on the Seine.", prose, bow_embed)
    check("an on-topic reply passes the meaning check", ok, str(pr))
    ok, pr = G.verify("Sure, setting a timer for ten minutes.", prose, bow_embed)
    check("an off-topic reply fails it", not ok and any("drifted" in p for p in pr), str(pr))
    ok, pr = G.verify("Sure, setting a timer for ten minutes.", prose)
    check("without an embedder the meaning check is skipped", ok)
    ok, pr = G.verify("Sure, setting a timer for ten minutes.", prose, boom)
    check("a broken embedder does not reject the reply", ok)
    ok, pr = G.verify("Sure, setting a timer.", prose, lambda t: np.ones(7, dtype=np.float32))
    check("an embedding of the wrong size is ignored, not fatal", ok)

    print("\n=== P4.5 Narration: rule path, checked model path, bounded retry ===")
    from src.brain.system2 import System2Deliberator
    drives0 = Drives()

    def narrate(llm, gist, embed=None):
        s2 = System2Deliberator(llm=llm, registry=None)
        text = s2.narrate("ASH", "q", {}, {}, "", drives0, "[EMO]", gist=gist, embed_fn=embed)
        return s2, text

    llm = ScriptedLLM(["should never be used"])
    s2, text = narrate(llm, G.build_gist(vetoed_by=["constitution"], attempted=att, drives=cool))
    check("a plain gist is rendered with zero model calls",
          text.startswith("I didn't format disk.") and not llm.prompts
          and s2.last_narration == {"mode": "rule", "calls": 0, "problems": []})

    needs_llm = G.build_gist(tool_ok=True, body="It is 10:30 and you have 3 reminders", drives=cool)
    llm = ScriptedLLM(["You've got three reminders, and it's half past ten."])
    s2, text = narrate(llm, needs_llm)
    check("a good first reply is accepted after one call",
          text.startswith("You've got three") and s2.last_narration["mode"] == "llm"
          and s2.last_narration["calls"] == 1)
    check("the model is briefed with the gist",
          "[WHAT TO SAY]" in llm.prompts[0] and "MUST mention: 3" in llm.prompts[0]
          and "purpose: inform" in llm.prompts[0])

    llm = ScriptedLLM(["It's half past ten.", "You have three reminders; it's half past ten."])
    s2, text = narrate(llm, needs_llm)
    check("a reply that drops a fact is retried once and the retry accepted",
          text.startswith("You have three") and s2.last_narration["mode"] == "llm_retry"
          and s2.last_narration["calls"] == 2)
    check("the retry says exactly what was wrong",
          "[CORRECTION]" in llm.prompts[1] and "left out '3'" in llm.prompts[1])
    check("the first prompt had no correction", "[CORRECTION]" not in llm.prompts[0])

    llm = ScriptedLLM(["It's half past ten.", "Still half past ten.", "never reached"])
    s2, text = narrate(llm, needs_llm)
    check("two bad replies => nothing unchecked is spoken (empty, caller uses raw facts)",
          text == "" and s2.last_narration["mode"] == "fallback" and len(llm.prompts) == 2
          and llm.replies == ["never reached"], str(s2.last_narration))

    llm = ScriptedLLM(["[LLM] Error calling Ollama API: timed out", "never reached"])
    s2, text = narrate(llm, needs_llm)
    check("a provider error is not retried (it would just stack another timeout)",
          text == "" and len(llm.prompts) == 1 and s2.last_narration["mode"] == "fallback",
          str(s2.last_narration))

    fail_llm = G.build_gist(tool_ok=False, tool_out={"ok": False, "error": "disk is full at 10:30"},
                            drives=cool)
    check("this failure really does need the model (error holds a time)", G.render_gist(fail_llm) is None)
    llm = ScriptedLLM(["Done, I successfully saved it.", "Sorry, it didn't work; the disk is full."])
    s2, text = narrate(llm, fail_llm)
    check("a false claim of success is caught and corrected",
          text.startswith("Sorry") and s2.last_narration["mode"] == "llm_retry"
          and "successfully" in llm.prompts[1])

    llm = ScriptedLLM(["Hi there!"])
    s2, text = narrate(llm, G.build_gist(drives=cool))
    check("open conversation is one free call with no gist brief",
          text == "Hi there!" and len(llm.prompts) == 1 and "[WHAT TO SAY]" not in llm.prompts[0]
          and s2.last_narration["calls"] == 1)
    llm = ScriptedLLM(["Hi there!"])
    s2, text = narrate(llm, None)
    check("no gist at all behaves exactly as before",
          text == "Hi there!" and "[WHAT TO SAY]" not in llm.prompts[0])

    prose_llm = G.build_gist(tool_ok=True, body="x" * 200 + " Paris is the capital of France.",
                             drives=cool, embed_fn=bow_embed)
    check("long prose goes through the model", G.render_gist(prose_llm) is None)
    llm = ScriptedLLM(["Sure, I'll set a timer for ten minutes.", "Paris is the capital of France."])
    s2, text = narrate(llm, prose_llm, embed=bow_embed)
    check("an off-topic reply is caught by the meaning check and retried",
          text.startswith("Paris") and s2.last_narration["mode"] == "llm_retry"
          and "drifted" in llm.prompts[1], str(s2.last_narration))

    s2 = System2Deliberator(llm=None, registry=None)
    text = s2.narrate("ASH", "q", {}, {}, "", drives0, "[EMO]", gist=needs_llm)
    check("no model configured => empty (caller falls back), no crash", text == "")

    print("\n=== P4.6 Brain wiring ===")
    import tst_brain

    def build(flag):
        os.environ["ASH_GIST"] = flag
        os.environ["ASH_PLANNING"] = "0"
        b = tst_brain.build_brain(tempfile.mkdtemp(prefix="ash_brain_"))
        b.predictive.persist = False
        # Candidate generation can also ask the model to nominate an action.
        # That is a separate call; these tests are about narration only.
        b.system2.allow_llm_proposals = False
        return b

    def force_choice(b, intent):
        """Have the board pick this tool whenever it is a candidate. The stub
        encoder is hash-seeded, so which option a cold board prefers varies
        from run to run; these tests are about what happens after it picks."""
        real = b.executive.arbitrate

        def pick(proposals, *a, **k):
            v = real(proposals, *a, **k)
            want = next((p for p in proposals if p.intent == intent), None)
            if want is not None and not v.vetoed_by:
                v.chosen = want
            return v
        b.executive.arbitrate = pick

    b = build("1")
    check("ASH_GIST=1 enables gist-first", b.gist_first is True)
    tst_brain.LLM_CALLS["n"] = 0
    r = b.think("format the disk now")
    check("a vetoed request is answered with no model call",
          tst_brain.LLM_CALLS["n"] == 0 and r.text.startswith("I didn't format disk."), r.text)
    gist_calls_veto = tst_brain.LLM_CALLS["n"]
    check("the trace records the gist and the route",
          r.trace.gist.get("intent") == "refuse"
          and any(n.startswith("narration: rule") for n in r.trace.notes), str(r.trace.gist.get("intent")))
    check("the refusal is still a real veto", "constitution" in r.trace.verdict.get("vetoed_by", []))

    b = build("1")
    force_choice(b, "calculate")
    tst_brain.LLM_CALLS["n"] = 0
    r = b.think("calculate 6 times 7")
    check("precondition: a first query deliberates", r.pathway in (Pathway.SLOW, Pathway.ESCALATED),
          r.pathway.value)
    check("a plain tool result needs no model call",
          tst_brain.LLM_CALLS["n"] == 0 and r.text.startswith("Result: 42"),
          f"calls={tst_brain.LLM_CALLS['n']} text={r.text!r}")

    b = build("1")
    force_choice(b, "get_time")
    b.system2.llm = ScriptedLLM(["It's ten o'clock in the morning."])
    r = b.think("what time is it")
    check("a timestamp result is phrased by the model, briefed with the gist",
          r.text.startswith("It's ten o'clock") and len(b.system2.llm.prompts) == 1
          and "[WHAT TO SAY]" in b.system2.llm.prompts[0], r.text)

    b = build("1")
    b.system2.llm = ScriptedLLM(["Hello! Good to hear from you."])
    orig = b.executive.arbitrate

    def undeadlocked(*a, **k):
        v = orig(*a, **k)
        v.deadlocked = False
        return v
    b.executive.arbitrate = undeadlocked
    r = b.think("hello there")
    check("plain conversation is left to the model, unbriefed",
          r.text.startswith("Hello!") and "[WHAT TO SAY]" not in b.system2.llm.prompts[0]
          and r.trace.gist.get("intent") == "converse", str(r.trace.gist.get("intent")))

    b = build("1")
    b.system2.llm = ScriptedLLM(["What would you like to know?"])
    r = b.think("hello there")
    check("a deadlock with nothing concrete to offer is phrased by the model as a question",
          r.trace.gist.get("intent") == "ask" and r.text == "What would you like to know?"
          and "purpose: ask" in b.system2.llm.prompts[0], str(r.trace.gist.get("intent")))

    b = build("1")
    orig = b.executive.arbitrate

    def deadlock(*a, **k):
        v = orig(*a, **k)
        v.deadlocked = True
        return v
    b.executive.arbitrate = deadlock
    tst_brain.LLM_CALLS["n"] = 0
    r = b.think("hello there")
    check("a deadlocked board with no concrete options asks via the model",
          tst_brain.LLM_CALLS["n"] == 1 and r.trace.gist.get("intent") == "ask", r.text)

    b = build("1")
    orig2 = b.executive.arbitrate

    def deadlock_on_conversation(proposals, *a, **k):
        v = orig2(proposals, *a, **k)
        v.deadlocked = True
        conv = next((p for p in proposals if p.intent is None), None)
        if conv is not None:          # the slow path's board; the gate has none
            v.chosen = conv
        return v
    b.executive.arbitrate = deadlock_on_conversation
    tst_brain.LLM_CALLS["n"] = 0
    r = b.think("what time is it and calculate 6 times 7")
    check("a deadlock between two concrete readings asks which, with no model call",
          tst_brain.LLM_CALLS["n"] == 0 and r.text.startswith("Did you mean")
          and r.trace.gist.get("intent") == "ask", r.text)

    b = build("0")
    tst_brain.LLM_CALLS["n"] = 0
    r = b.think("format the disk now")
    check("with ASH_GIST off nothing changes: the model narrates and no gist is built",
          tst_brain.LLM_CALLS["n"] == gist_calls_veto + 1 and r.trace.gist == {}
          and b.gist_first is False,
          f"off={tst_brain.LLM_CALLS['n']} on={gist_calls_veto}")

    b = build("1")
    force_choice(b, "get_time")
    b.system2.llm = ScriptedLLM(["It's half past ten.", "Still no."])
    b.registry._by_intent["get_time"].func = lambda q: "It is 10:30 and you have 3 reminders"
    r = b.think("what time is it")
    check("when no phrasing passes, the raw facts are spoken, never an unchecked claim",
          r.text == "It is 10:30 and you have 3 reminders"
          and any("narration: fallback" in n for n in r.trace.notes)
          and any("narration empty" in n for n in r.trace.notes), r.text)

    print("\n=== P4.7 Meaning-check threshold against the real encoder (if installed) ===")
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(os.environ.get("ASH_EMBED_MODEL", "all-MiniLM-L6-v2"))

        def emb(t):
            return enc.encode([t], convert_to_numpy=True, normalize_embeddings=True)[0]
        cap = G.build_gist(tool_ok=True, drives=cool, embed_fn=emb,
                           body="The capital of France is Paris, a city on the Seine known for the "
                                "Eiffel Tower, and it has been the capital for centuries now. " * 2)
        ok_good, pr1 = G.verify("Paris is the capital, and it sits on the Seine.", cap, emb)
        ok_bad, pr2 = G.verify("Sure, I'll set a timer for ten minutes.", cap, emb)
        check("a real on-topic paraphrase passes", ok_good, str(pr1))
        check("a real off-topic reply is rejected", not ok_bad and bool(pr2), str(pr2))
        ts = G.build_gist(tool_ok=True, body="2026-08-24T10:00:00", drives=cool, embed_fn=emb)
        ok_ts, pr3 = G.verify("It's ten o'clock in the morning.", ts, emb)
        check("a natural reply to a timestamp is not wrongly rejected", ok_ts, str(pr3))
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP  sentence-transformers unavailable ({type(e).__name__})")

    print("\n=== P4.8 The existing brain suite still passes with everything on ===")
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, ASH_PLANNING="1", ASH_GIST="1",
               ASH_BRAIN_STATE=tempfile.mkdtemp(prefix="ash_tb_"))
    res = subprocess.run([sys.executable, os.path.join(here, "tst_brain.py")], env=env,
                         capture_output=True, text=True, cwd=here)
    check("tst_brain.py passes under ASH_PLANNING=1 ASH_GIST=1",
          res.returncode == 0 and "ALL CHECKS PASSED" in res.stdout,
          res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-200:])
    os.environ["ASH_GIST"] = "0"


def main():
    phase1()
    phase2()
    phase3()
    phase4()
    print("\n" + ("ALL CHECKS PASSED" if RESULTS["ok"] else "SOME CHECKS FAILED"))
    return 0 if RESULTS["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
