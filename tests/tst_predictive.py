"""
tst_predictive.py -- the predictive model survives bad input and is never read with pickle.

  P1  A NaN or infinite frame is skipped; a model whose weights already went bad starts again instead of staying poisoned
  P2  A saved model that is the wrong shape or not finite is refused (and ASH starts fresh)
  P3  The saved files hold text, not objects, so they load without pickle; an old-format file is still read once and rewritten

Run:  PYTHONIOENCODING=utf-8 python tests/tst_predictive.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain import predictive as pm  # noqa: E402

check = tw.check
DIM = 16


def vec(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def model(tmp=None, persist=False):
    if tmp:
        pm.STATE_DIR = tmp
    return pm.PredictiveModel(dim=DIM, persist=persist)


def p1_bad_input():
    print("\n=== P1. A bad frame ===")
    m = pm.LatentForwardModel(dim=DIM)
    s, t = vec(1), vec(2)
    before = m.W_s.copy()
    bad = s.copy()
    bad[3] = np.nan
    check("a NaN frame is skipped, the weights are untouched", m.learn(bad, "x", t) == 0.0 and np.array_equal(m.W_s, before) and m.steps == 0)
    inf = t.copy()
    inf[0] = np.inf
    check("...and so is an infinite next frame", m.learn(s, "x", inf) == 0.0 and m.healthy())
    check("a prediction from a NaN frame is None, not NaN", m.predict(bad, "x") is None)
    m.learn(s, "x", t)
    check("a good frame still learns", m.steps == 1 and m.healthy())
    m.W_s[0, 0] = np.nan                                 # poisoned already (an old file, an overflow)
    check("poisoned weights: predict says nothing", m.predict(s, "x") is None)
    m.learn(s, "x", t)
    check("...and the next learn starts the weights again instead of staying NaN", m.healthy() and m.predict(s, "x") is not None)


def p2_refused():
    print("\n=== P2. A saved model that is wrong ===")
    m = pm.LatentForwardModel(dim=DIM)
    d = m.state_dict()
    ok = pm.LatentForwardModel(dim=DIM)
    ok.load_state(d)
    check("a good state loads", ok.healthy())
    for label, mutate in (("not finite", lambda x: x["W_s"].__setitem__((1, 1), np.nan)),
                          ("wrong shape", lambda x: x.update(W_s=np.zeros((3, 3), dtype=np.float32)))):
        d2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in d.items()}
        mutate(d2)
        fresh = pm.LatentForwardModel(dim=DIM)
        try:
            fresh.load_state(d2)
            refused = False
        except ValueError:
            refused = True
        check(f"{label}: refused, and the model it would have replaced is unharmed", refused and fresh.healthy())


def p3_no_pickle():
    print("\n=== P3. No pickle ===")
    tmp = tempfile.mkdtemp()
    m = model(tmp, persist=True)
    for i in range(5):
        m.forward.learn(vec(i), "greet", vec(i + 1))
    m.goal_protos["greet"] = vec(9)
    m.goal_counts["greet"] = 3
    m.save()
    for name in ("predictive.npz", "goals.npz"):
        try:
            with np.load(os.path.join(tmp, name), allow_pickle=False) as f:
                ok = all(f[k] is not None for k in f.files)
        except Exception as e:
            ok = False
            print("   ", name, type(e).__name__, e)
        check(f"{name} loads with allow_pickle=False", ok)
    m2 = model(tmp, persist=True)
    check("a saved model comes back (steps and goals)", m2.forward.steps == 5 and "greet" in m2.goal_protos and m2.goal_counts["greet"] == 3)
    check("...with its action table", "greet" in m2.forward.action_table)

    old = tempfile.mkdtemp()
    state = m.forward.state_dict()
    state["actions"] = np.array(list(m.forward.action_table.keys()), dtype=object)
    np.savez_compressed(os.path.join(old, "predictive.npz"), **state)
    np.savez_compressed(os.path.join(old, "goals.npz"), intents=np.array(["greet"], dtype=object), protos=np.stack([vec(9)]),
                        counts=np.array([3], dtype=np.int64))
    m3 = model(old, persist=True)
    check("an old-format file (object arrays) is still read once", m3.forward.steps == 5 and "greet" in m3.goal_protos)
    m3.save()
    with np.load(os.path.join(old, "goals.npz"), allow_pickle=False) as f:
        check("...and rewritten in the safe form", f["intents"].dtype.kind == "U")


def p4_save_gap():
    print("\n=== P4. Saving is not every twenty cycles of ambient noise ===")
    from types import SimpleNamespace
    from collections import deque
    from src.brain.brain import Brain
    from src.brain.signals import Pathway
    saves = []
    fake = SimpleNamespace(
        pathway_counts={}, trace_history=deque(maxlen=8), answer_history=deque(maxlen=8), last_trace=None, _last_cycle_cost=0,
        sensory=SimpleNamespace(save_requests=lambda: None), system1=SimpleNamespace(save=lambda: saves.append("s1")),
        predictive=SimpleNamespace(save=lambda: saves.append("pred")), PERSIST_MIN_GAP_S=Brain.PERSIST_MIN_GAP_S, cycles=0)
    trace = SimpleNamespace(pathway=None)
    for _ in range(100):
        Brain._finish_cycle(fake, Pathway.OBSERVE, trace)
    check("100 cycles in a burst write the models once, not five times", saves == ["s1", "pred"], str(saves))
    fake._last_persist -= Brain.PERSIST_MIN_GAP_S + 1
    for _ in range(20):
        Brain._finish_cycle(fake, Pathway.OBSERVE, trace)
    check("a minute later, the next twenty cycles write again", saves == ["s1", "pred", "s1", "pred"], str(saves))


if __name__ == "__main__":
    p1_bad_input()
    p2_refused()
    p3_no_pickle()
    p4_save_gap()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
