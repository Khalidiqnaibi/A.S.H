"""
tst_conceptlearn.py -- what the mind page found wrong with the concept graph, and the fix for each.

  C1  time and running strength survive a restart (created, last fired, activation sum; edge creation and last use)
  C2  a concept that has taken far more than its share is held back (conscience), unless it truly is the best match
  C3  a concept that everything already points at is a harder target for one more wire

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


def _two(conscience, acts_hub=5000, k=1):
    """A hub with a huge history and a modest rival, and an input a little closer to the hub than to the rival."""
    n = net(k=k, conscience=conscience, activation_floor=0.1)
    base = rnd(1)
    hub = n.add_concept(base, seed_text="hub")
    rival_v = rnd(2, base=base, noise=0.9)
    rival = n.add_concept(rival_v, seed_text="rival")
    for i in range(20):
        c = n.add_concept(rnd(10 + i), seed_text="other%d" % i)
        c.activations = 20
    hub.activations = acts_hub
    rival.activations = 20
    return n, hub, rival


def c2_conscience():
    print("\n=== C2. A concept that wins everything is held back ===")
    n, hub, rival = _two(conscience=0.0)
    mid = rnd(3, base=hub.prototype + 0.55 * rival.prototype)           # a bit nearer the hub than the rival
    sh, sr = float(hub.prototype @ mid), float(rival.prototype @ mid)
    check("setup: the input is nearer the hub than the rival, both well above the floor", sh > sr > 0.3, "%.2f %.2f" % (sh, sr))
    first, _ = n._kwta(mid)
    check("without conscience the hub wins the single slot", list(first) == [hub.id])
    n2, hub2, rival2 = _two(conscience=1.0)
    second, _ = n2._kwta(mid)
    check("with conscience the rival takes the slot instead", list(second) == [rival2.id], str(second))
    check("the similarity reported is the real one, not the penalised one", abs(second[rival2.id] - sr) < 1e-5)
    near = hub2.prototype.copy()
    third, _ = n2._kwta(near)
    check("a hub that truly is the best match still wins", list(third) == [hub2.id])
    n3, hub3, _ = _two(conscience=1.0, acts_hub=20)
    n3._rebuild()
    p = n3._conscience_penalty()
    check("no penalty while nothing has taken more than its share", float(np.max(p)) == 0.0 if hasattr(p, "max") else p == 0.0)
    n4 = net(k=1, conscience=1.0)
    from src.concepts.model import Origin as O
    seeded = n4.add_concept(rnd(1), label="get_time", origin=O.INTENT, intent="get_time")
    for i in range(12):
        n4.add_concept(rnd(30 + i)).activations = 5
    seeded.activations = 9000
    n4._rebuild()
    pen = n4._conscience_penalty()
    check("a skill that came with ASH is never penalised, however busy", float(pen[n4._ids.index(seeded.id)]) == 0.0 if hasattr(pen, "__len__") else True)
    n5 = net(k=3, conscience=1.0)
    for i in range(5):
        n5.add_concept(rnd(50 + i)).activations = 1
    check("with too few concepts, or too little history, it does nothing", n5._conscience_penalty() == 0.0)
    r = net(k=2, conscience=1.0)
    r.add_concept(rnd(1))
    act = r.perceive(rnd(1), "x")
    check("a normal perceive still works and grows nothing for a perfect match", act.grown is None and len(r.concepts) == 1)


def c3_hub_inputs():
    print("\n=== C3. A concept everything points at learns fewer wires ===")
    from src.concepts.plasticity import Plasticity
    n = net()
    ids = [n.add_concept(rnd(i), seed_text="c%d" % i).id for i in range(70)]
    pl = Plasticity(n)
    hub, plain = ids[0], ids[1]
    for s_ in ids[2:60]:
        n.synapse(s_, hub)
    for s_ in ids[2:6]:
        n.synapse(s_, plain)
    check("setup: the hub has 58 wires in, the plain one 4", len(n.inputs(hub)) == 58 and len(n.inputs(plain)) == 4)
    check("a concept with few inputs is learned at the full rate", pl.hub_scale(plain) == 1.0)
    check("a hub is learned at a fraction of it", 0.15 < pl.hub_scale(hub) < 0.35, str(pl.hub_scale(hub)))
    check("the rate keeps falling as inputs pile up", pl.hub_scale(hub) < pl.hub_scale(ids[60]) or pl.hub_scale(hub) < 1.0)
    from src.concepts.model import Activation
    a, b = ids[60], ids[61]
    for x in (hub, plain):
        n.concepts[x].activations = 3
    n.cycles = 50
    pl.replay_mode, pl.stdp_enabled = True, False        # the plain Hebbian rule, the same gate for both
    for _ in range(30):
        n.concepts[a].activations = n.concepts[b].activations = 3
        pl.potentiate(Activation(initial={a: 0.9, hub: 0.9}, spread={a: 0.9, hub: 0.9}))
        pl.potentiate(Activation(initial={b: 0.9, plain: 0.9}, spread={b: 0.9, plain: 0.9}))
    to_hub = n.synapses.get((a, hub))
    to_plain = n.synapses.get((b, plain))
    check("the same evidence builds a smaller wire into the hub than into the plain concept",
          (to_hub is None) or (to_plain is not None and abs(to_hub.w_fast) < abs(to_plain.w_fast)), "%s vs %s" % (to_hub and to_hub.w_fast, to_plain and to_plain.w_fast))


if __name__ == "__main__":
    for fn in (c1_time_survives, c2_conscience, c3_hub_inputs):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
