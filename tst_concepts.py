"""
tst_concepts.py -- tests for the associative layer.

The interesting failures in a self-rewiring system are all slow ones: it
works for a week and then quietly stops discriminating. So these tests target
the specific mechanisms that prevent that, not just "does it activate".

  * k-WTA produces a sparse code, and priming carries context forward
  * resonance grows a unit for a genuinely novel input and not otherwise
  * PMI rejects a concept that co-occurs with everything (the base-rate trap)
  * Oja bounds weights under sustained co-activation (the runaway trap)
  * homeostatic scaling caps a hub's total outgoing weight (the hub trap)
  * decay + pruning actually delete edges (the "rewireable" claim)
  * the third factor: rewarded pairs consolidate harder than unrewarded ones
  * frozen core edges survive every plasticity pass untouched
  * spreading activation infers a concept the input never matched
  * consolidation rolls back when it degrades routing
  * associative recall finds an episode sharing no vocabulary with the query
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())

from src.concepts import (  # noqa: E402
    ConceptNetwork, Origin, Plasticity, PlasticityConfig,
)
from src.concepts.bindings import ConceptMemoryIndex  # noqa: E402

DIM = 32
RNG = np.random.default_rng(4)


def vec(seed=None):
    r = np.random.default_rng(seed) if seed is not None else RNG
    v = r.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v, noise=0.12):
    w = v + noise * RNG.standard_normal(DIM).astype(np.float32)
    return w / np.linalg.norm(w)


def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and cond

    # ---------------------------------------------------------------
    print("\n=== 1. Sparse activation and priming ===")
    net = ConceptNetwork(dim=DIM, k=5, vigilance=0.7, persist=False)
    protos = [vec(i) for i in range(20)]
    for i, p in enumerate(protos):
        net.add_concept(p, label=f"n{i}", origin=Origin.INTENT)

    act = net.perceive(near(protos[3]), allow_growth=False)
    check("activation is sparse", len(act.spread) <= 8, f"{len(act.spread)} of 20 active")
    check("the right concept wins", max(act.spread.items(), key=lambda kv: kv[1])[0] ==
          net.by_label("n3").id)

    before = len(net.perceive(near(protos[7]), allow_growth=False).spread)
    primed = net.perceive(near(protos[7]), allow_growth=False)
    check("priming carries context forward", net.by_label("n7").id in primed.spread,
          f"{before} -> {len(primed.spread)} active")

    # ---------------------------------------------------------------
    print("\n=== 2. Resonance growth ===")
    net2 = ConceptNetwork(dim=DIM, k=5, vigilance=0.7, persist=False)
    for i, p in enumerate(protos[:8]):
        net2.add_concept(p, label=f"n{i}", origin=Origin.INTENT)
    n0 = len(net2.concepts)

    net2.perceive(near(protos[2], noise=0.05), text="familiar")
    check("familiar input grows nothing", len(net2.concepts) == n0,
          f"{len(net2.concepts)} concepts")

    stranger = vec(999)
    a = net2.perceive(stranger, text="something entirely new")
    check("novel input grows a unit", a.grown is not None and len(net2.concepts) == n0 + 1,
          f"grew {a.grown}")
    check("grown unit is unnamed", net2.concepts[a.grown].label is None)
    check("grown unit records provenance",
          "entirely new" in net2.concepts[a.grown].seed_text)
    check("describe() identifies an unnamed unit",
          net2.describe(a.grown).startswith("c"), net2.describe(a.grown))

    # ---------------------------------------------------------------
    print("\n=== 3. PMI rejects the base-rate trap ===")
    net3 = ConceptNetwork(dim=DIM, k=6, vigilance=0.99, persist=False)
    pl3 = Plasticity(net3, PlasticityConfig(pmi_floor=0.15))
    ids = [net3.add_concept(vec(100 + i), label=f"c{i}", origin=Origin.INTENT).id
           for i in range(6)]

    # c0 is present in EVERY cycle; c1 and c2 always appear together but only
    # in half of them. Naive Hebb would wire c0 to everything most strongly.
    from src.concepts.model import Activation
    for step in range(120):
        spread = {ids[0]: 0.9}
        if step % 2 == 0:
            spread[ids[1]] = 0.9
            spread[ids[2]] = 0.9
        else:
            spread[ids[3]] = 0.9
            spread[ids[4 + (step % 2)]] = 0.5
        net3.cycles += 1
        for cid, v in spread.items():
            net3.concepts[cid].touch(v)
        a = Activation(initial=spread, spread=spread)
        net3._record_cooccurrence(a)
        pl3.potentiate(a)

    w_ubiq = abs(net3.synapses[(ids[0], ids[1])].weight()) if (ids[0], ids[1]) in net3.synapses else 0.0
    w_real = abs(net3.synapses[(ids[1], ids[2])].weight()) if (ids[1], ids[2]) in net3.synapses else 0.0
    check("real association beats the ubiquitous one", w_real > w_ubiq,
          f"c1-c2={w_real:.4f} vs c0-c1={w_ubiq:.4f}")
    check("PMI of the ubiquitous pair is near zero",
          pl3.pmi(ids[0], ids[1]) < pl3.pmi(ids[1], ids[2]),
          f"{pl3.pmi(ids[0], ids[1]):.3f} vs {pl3.pmi(ids[1], ids[2]):.3f}")

    # ---------------------------------------------------------------
    print("\n=== 4. Oja bounds runaway ===")
    # The pair co-occurs in half the cycles, with a distractor pair filling
    # the rest. A pair that co-occurred in *every* cycle would have PMI of
    # exactly zero and never wire at all -- correct, but degenerate, and not
    # what this test is about. See Plasticity.pmi.
    net4 = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
    pl4 = Plasticity(net4)
    a4 = net4.add_concept(vec(1), label="a", origin=Origin.INTENT).id
    b4 = net4.add_concept(vec(2), label="b", origin=Origin.INTENT).id
    d4 = net4.add_concept(vec(3), label="d", origin=Origin.INTENT).id
    e4 = net4.add_concept(vec(4), label="e", origin=Origin.INTENT).id

    trace = []
    for i in range(800):
        pair = {a4: 1.0, b4: 1.0} if i % 2 == 0 else {d4: 1.0, e4: 1.0}
        net4.cycles += 1
        for cid, v in pair.items():
            net4.concepts[cid].touch(v)
        act4 = Activation(initial=pair, spread=pair)
        net4._record_cooccurrence(act4)
        pl4.potentiate(act4)
        if i % 200 == 40:
            syn4 = net4.synapses.get((a4, b4))
            trace.append(syn4.weight() if syn4 else 0.0)

    check("an edge forms for a real association", (a4, b4) in net4.synapses)
    final = net4.synapses[(a4, b4)].weight()
    check("weight converges instead of diverging", final < 5.0,
          f"after 400 co-activations w={final:.3f}")
    check("growth decelerates", (trace[-1] - trace[-2]) < (trace[1] - trace[0]) + 1e-9,
          f"trace={[round(t, 3) for t in trace]}")

    # ---------------------------------------------------------------
    print("\n=== 5. Homeostatic scaling caps hubs ===")
    net5 = ConceptNetwork(dim=DIM, k=8, vigilance=0.99, persist=False)
    pl5 = Plasticity(net5, PlasticityConfig(weight_budget=2.0))
    hub = net5.add_concept(vec(50), label="hub", origin=Origin.INTENT).id
    spokes = [net5.add_concept(vec(60 + i), label=f"s{i}", origin=Origin.INTENT).id
              for i in range(12)]
    for s in spokes:
        net5.link(hub, s, 0.9)
    total_before = sum(abs(x.weight()) for x in net5._out[hub].values())
    pl5._homeostatic_scaling()
    total_after = sum(abs(x.weight()) for x in net5._out[hub].values())
    check("hub outgoing weight is capped", total_after <= 2.01,
          f"{total_before:.2f} -> {total_after:.2f}")
    ratios = [net5.synapses[(hub, s)].weight() for s in spokes]
    check("scaling is multiplicative (relative strengths preserved)",
          max(ratios) - min(ratios) < 1e-6)

    # ---------------------------------------------------------------
    print("\n=== 6. Decay and pruning make it rewireable ===")
    net6 = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
    pl6 = Plasticity(net6, PlasticityConfig(decay_per_sleep=0.5, prune_below=0.05))
    x = net6.add_concept(vec(70), label="x", origin=Origin.INTENT).id
    y = net6.add_concept(vec(71), label="y", origin=Origin.INTENT).id
    net6.link(x, y, 0.30)
    n_before = len(net6.synapses)
    for _ in range(6):
        pl6.consolidate()
    check("unreinforced edges are deleted", len(net6.synapses) < n_before,
          f"{n_before} -> {len(net6.synapses)} synapses")

    # ---------------------------------------------------------------
    print("\n=== 7. Third factor: reward gates consolidation ===")
    def run_pair(reward):
        n = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
        p = Plasticity(n)
        i1 = n.add_concept(vec(80), label="p", origin=Origin.INTENT).id
        i2 = n.add_concept(vec(81), label="q", origin=Origin.INTENT).id
        i3 = n.add_concept(vec(82), label="r", origin=Origin.INTENT).id
        for step in range(60):
            # Alternate so the pair's base rate stays below 1.0; see test 4.
            pair = {i1: 1.0, i2: 1.0} if step % 2 == 0 else {i3: 1.0}
            n.cycles += 1
            for cid, v in pair.items():
                n.concepts[cid].touch(v)
            a = Activation(initial=pair, spread=pair)
            n._record_cooccurrence(a)
            p.potentiate(a)
            if reward and step % 2 == 0:
                p.reward(1.0)
        p.consolidate()
        syn = n.synapses.get((i1, i2))
        return syn.w_slow if syn else 0.0

    rewarded = run_pair(True)
    plain = run_pair(False)
    check("rewarded associations consolidate harder", rewarded > plain,
          f"rewarded w_slow={rewarded:.4f} vs unrewarded={plain:.4f}")
    check("unrewarded associations still form something", plain > 0,
          f"{plain:.4f}")

    # ---------------------------------------------------------------
    print("\n=== 8. Frozen core edges are untouchable ===")
    net8 = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
    pl8 = Plasticity(net8, PlasticityConfig(decay_per_sleep=0.9, prune_below=0.5))
    rule = net8.add_concept(vec(90), label="rule:never_format", origin=Origin.CORE,
                            frozen=True).id
    danger = net8.add_concept(vec(91), label="format_disk", origin=Origin.INTENT,
                              intent="format_disk").id
    net8.link(rule, danger, weight=-1.0, bidirectional=False, frozen=True)

    other = net8.add_concept(vec(92), label="other", origin=Origin.INTENT).id
    for step in range(80):
        pair = {rule: 1.0, danger: 1.0} if step % 2 == 0 else {other: 1.0}
        net8.cycles += 1
        for cid, v in pair.items():
            net8.concepts[cid].touch(v)
        a = Activation(initial=pair, spread=pair)
        net8._record_cooccurrence(a)
        pl8.potentiate(a)
        if step % 2 == 0:
            pl8.reward(1.0)      # actively try to reward it into positive
    pl8.consolidate()
    pl8.consolidate()

    syn = net8.synapses.get((rule, danger))
    check("frozen inhibitory edge survives", syn is not None)
    check("frozen edge weight is unchanged", syn is not None and abs(syn.weight() + 1.0) < 1e-9,
          f"w={syn.weight() if syn else 'gone'}")

    # ---------------------------------------------------------------
    print("\n=== 9. Spreading activation infers ===")
    net9 = ConceptNetwork(dim=DIM, k=3, vigilance=0.99, spread_steps=2,
                          spread_gain=0.8, persist=False)
    battery = net9.add_concept(vec(10), label="low_battery", origin=Origin.INTENT).id
    charger = net9.add_concept(vec(11), label="charger", origin=Origin.ENTITY).id
    desk = net9.add_concept(vec(12), label="desk", origin=Origin.ENTITY).id
    net9.link(battery, charger, 0.9)
    net9.link(charger, desk, 0.8)

    a9 = net9.perceive(net9.concepts[battery].prototype, allow_growth=False)
    check("direct associate is inferred", charger in a9.spread and charger not in a9.initial,
          f"inferred={[net9.describe(c) for c, _ in a9.inferred()]}")
    check("two-hop associate is reached", desk in a9.spread,
          f"desk activation={a9.spread.get(desk, 0):.3f}")
    check("inferred is weaker than matched", a9.spread[charger] < a9.spread[battery])

    # ---------------------------------------------------------------
    print("\n=== 10. Consolidation rolls back a regression ===")
    net10 = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
    pl10 = Plasticity(net10)
    ia = net10.add_concept(vec(20), label="A", origin=Origin.INTENT, intent="A").id
    net10.link(ia, ia, 0.5, bidirectional=False)
    w0 = net10.synapses[(ia, ia)].w_slow

    scores = iter([1.0, 0.2])       # accuracy collapses after the pass

    def bad_evaluator():
        return next(scores, 0.2)

    for _ in range(10):
        net10.cycles += 1
        net10.concepts[ia].touch(1.0)
        a = Activation(initial={ia: 1.0}, spread={ia: 1.0})
        pl10.potentiate(a)
    rep = pl10.consolidate(evaluator=bad_evaluator)
    check("regression is detected and reverted", rep.get("ROLLED_BACK") is True, str(rep)[:90])
    check("weights restored", abs(net10.synapses[(ia, ia)].w_slow - w0) < 1e-9,
          f"{w0:.4f} -> {net10.synapses[(ia, ia)].w_slow:.4f}")

    # ---------------------------------------------------------------
    print("\n=== 11. Associative recall across vocabulary ===")
    net11 = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, spread_gain=0.9, persist=False)
    idx = ConceptMemoryIndex(net11)
    deadline = net11.add_concept(vec(30), label="deadline", origin=Origin.ENTITY).id
    stress = net11.add_concept(vec(31), label="stress", origin=Origin.ENTITY).id
    net11.link(deadline, stress, 0.95)

    laid = net11.perceive(net11.concepts[stress].prototype, allow_growth=False)
    idx.index("ep1", "user was short-tempered on Tuesday", laid)

    query = net11.perceive(net11.concepts[deadline].prototype, allow_growth=False)
    hits = idx.recall(query)
    check("recall crosses to an associated episode",
          any("short-tempered" in h for h, _ in hits),
          f"{[h[:40] for h, _ in hits]}")

    # ---------------------------------------------------------------
    print("\n=== 12. Persistence ===")
    d = tempfile.mkdtemp()
    net.persist = True
    net.save(d)
    net_r = ConceptNetwork(dim=DIM, persist=False)
    check("graph round-trips through disk", net_r.load(d),
          f"{len(net_r.concepts)} concepts restored")
    check("labels survive", net_r.by_label("n3") is not None)

    print("\n=== Final network state ===")
    st = net3.status()
    print(f"     {st['concepts']} concepts, {st['synapses']} synapses, "
          f"density {st['density']}, mean w {st['mean_weight']}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
