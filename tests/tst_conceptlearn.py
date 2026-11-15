"""
tst_conceptlearn.py -- what the mind page found wrong with the concept graph, and the fix for each.

  C1  time and running strength survive a restart (created, last fired, activation sum; edge creation and last use)

Everything runs on a small made-up network in a temp folder; no saved graph of the user's is read or written.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_conceptlearn.py
"""

import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import numpy as np  # noqa: E402

import tst_wondering as tw  # noqa: E402
from src.concepts.model import Origin  # noqa: E402
from src.concepts.network import ConceptNetwork  # noqa: E402

check = tw.check
DIM = 32


def rnd(seed, base=None, noise=0.0):
    r = np.random.default_rng(seed)
    v = (base if base is not None else r.standard_normal(DIM)) + noise * r.standard_normal(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def net(**kw):
    return ConceptNetwork(dim=DIM, persist=False, **kw)


def c1_time_survives():
    print("\n=== C1. Time and strength survive a restart ===")
    a = ConceptNetwork(dim=DIM)
    c = a.add_concept(rnd(1), origin=Origin.GROWN, seed_text="first")
    d = a.add_concept(rnd(2), origin=Origin.GROWN, seed_text="second")
    c.created = time.time() - 3 * 86400
    for _ in range(5):
        c.touch(0.6)
    a.synapse(c.id, d.id).w_slow = 0.2
    a.synapse(c.id, d.id).created = time.time() - 86400
    a.synapse(c.id, d.id).last_potentiated = time.time() - 600
    folder = tempfile.mkdtemp()
    a.save(folder)
    b = ConceptNetwork(dim=DIM)
    ok = b.load(folder)
    r = b.concepts[c.id]
    check("it loads", ok)
    check("the creation time is the real one (three days ago), not the moment of loading", abs(r.created - c.created) < 2 and time.time() - r.created > 2.9 * 86400)
    check("last fired and the running strength are kept", abs(r.last_active - c.last_active) < 2 and abs(r.activation_sum - 3.0) < 1e-3 and abs(r.mean_activation() - 0.6) < 1e-3)
    s = b.synapses[(c.id, d.id)]
    check("the wire keeps when it was made and when it was last used", abs(s.created - (time.time() - 86400)) < 5 and abs(s.last_potentiated - (time.time() - 600)) < 5)
    check("a concept never fired loads with 0 as last_active and no mean", b.concepts[d.id].last_active == 0.0 and b.concepts[d.id].mean_activation() == 0.0)
    # An older file without the new keys still loads, as before.
    import json
    gp = os.path.join(folder, "graph.json")
    meta = json.load(open(gp))
    for m in meta["concepts"]:
        for k in ("created", "last_active", "activation_sum"):
            m.pop(k, None)
    for m in meta["synapses"]:
        for k in ("created", "last"):
            m.pop(k, None)
    json.dump(meta, open(gp, "w"))
    old = ConceptNetwork(dim=DIM)
    check("a file saved before this still loads, with creation time = now", old.load(folder) and abs(old.concepts[c.id].created - time.time()) < 60)


if __name__ == "__main__":
    for fn in (c1_time_survives,):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
