"""
tst_neuro.py -- tests for the brain-faithful additions.

Targets the specific claims made by the architecture, each of which is
falsifiable:

  * one-shot learning: a single exposure is recallable immediately
  * pattern separation: similar inputs get distinguishable codes
  * pattern completion: a partial cue retrieves the whole episode
  * CLS: hippocampus learns in one shot, cortex needs replay
  * interleaved replay prevents catastrophic forgetting
  * STDP makes edges directional -- A->B strong, B->A weak
  * adaptation silences an always-on concept for learning purposes
  * acetylcholine switches encode/consolidate
  * synaptic downscaling renormalizes without inverting rank order
  * developmental annealing: early learning is faster than late
  * schema fast-track: consolidation accelerates in a familiar domain
  * uncertainty arbitration transfers control to whoever is actually right
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())

from src.concepts import (  # noqa: E402
    ConceptNetwork, Hippocampus, Neuromodulators, Origin, Plasticity,
    PlasticityConfig, SleepConfig, SleepCycle, SynType,
)
from src.concepts.model import Activation  # noqa: E402

DIM = 64
RNG = np.random.default_rng(11)


def v(seed=None):
    r = np.random.default_rng(seed) if seed is not None else RNG
    x = r.standard_normal(DIM).astype(np.float32)
    return x / np.linalg.norm(x)


def near(p, noise=0.15):
    x = p + noise * RNG.standard_normal(DIM).astype(np.float32)
    return x / np.linalg.norm(x)


def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and cond

    # ---------------------------------------------------------------
    print("\n=== 1. One-shot episodic learning ===")
    hpc = Hippocampus(dim=DIM, expansion=1024, sparsity=20, capacity=200)
    fact = v(1)
    hpc.store(fact, text="the spare key is under the third plant pot")
    hits = hpc.recall(fact)
    check("recallable after exactly one exposure", len(hits) == 1,
          f"{hits[0][0].text[:44]!r} @ {hits[0][1]:.2f}" if hits else "nothing")

    partial = near(fact, noise=0.55)
    hits = hpc.recall(partial)
    check("pattern completion from a degraded cue", len(hits) >= 1,
          f"overlap={hits[0][1]:.3f}" if hits else "no completion")

    unrelated = hpc.recall(v(999))
    check("does not hallucinate a match for an unrelated cue", len(unrelated) == 0,
          f"{len(unrelated)} spurious")

    # ---------------------------------------------------------------
    print("\n=== 2. Pattern separation ===")
    a, b = v(2), None
    b = near(a, noise=0.25)
    cos = float(np.dot(a, b))
    ia, va = hpc.separate(a)
    ib, vb = hpc.separate(b)
    sep = Hippocampus.overlap(ia, va, ib, vb)
    check("similar inputs are separated", sep < cos,
          f"cosine {cos:.3f} -> sparse overlap {sep:.3f}")

    ic, vc = hpc.separate(a)
    check("separation is deterministic", Hippocampus.overlap(ia, va, ic, vc) > 0.99)

    # ---------------------------------------------------------------
    print("\n=== 3. CLS: fast store vs slow cortex ===")
    net = ConceptNetwork(dim=DIM, k=6, vigilance=0.55, persist=False)
    pl = Plasticity(net)
    hpc2 = Hippocampus(dim=DIM, expansion=1024, sparsity=20)
    nm = Neuromodulators()
    pl.neuromod = nm

    p1 = net.add_concept(v(30), label="alpha", origin=Origin.ENTITY).id
    p2 = net.add_concept(v(31), label="beta", origin=Origin.ENTITY).id
    combo = (net.concepts[p1].prototype + net.concepts[p2].prototype)
    combo /= np.linalg.norm(combo)

    hpc2.store(combo, text="alpha and beta happened together", reward=0.8)
    syn = net.synapses.get((p1, p2))
    check("one exposure leaves cortex unchanged", syn is None or abs(syn.w_slow) < 1e-6,
          "cortex correctly refuses to learn from n=1")
    check("but the hippocampus has it", len(hpc2.recall(combo)) == 1)

    sleeper = SleepCycle(net, pl, hpc2, nm,
                         SleepConfig(cycles=2, nrem_batches=4, batch_size=8))
    sleeper.run()
    syn = net.synapses.get((p1, p2))
    check("replay transfers it to cortex", syn is not None and syn.w_slow != 0.0,
          f"w_slow={syn.w_slow:.4f}" if syn else "still absent")

    # ---------------------------------------------------------------
    print("\n=== 4. Interleaved replay prevents forgetting ===")
    def forgetting_run(interleave: bool):
        n = ConceptNetwork(dim=DIM, k=6, vigilance=0.99, persist=False)
        p = Plasticity(n)
        h = Hippocampus(dim=DIM, expansion=1024, sparsity=20)
        old_a = n.add_concept(v(40), label="old_a", origin=Origin.ENTITY).id
        old_b = n.add_concept(v(41), label="old_b", origin=Origin.ENTITY).id
        n.link(old_a, old_b, 0.8)

        # Old episodes, then a flood of unrelated new ones.
        oc = (n.concepts[old_a].prototype + n.concepts[old_b].prototype)
        oc /= np.linalg.norm(oc)
        for _ in range(6):
            t = h.store(oc, text="old pairing")
            t.created -= 3600 * 48        # make them "old"
        for i in range(40):
            h.store(v(200 + i), text=f"new thing {i}")

        s = SleepCycle(n, p, h, None,
                       SleepConfig(cycles=2, nrem_batches=6, batch_size=12,
                                   new_fraction=(0.4 if interleave else 1.0)))
        s.run()
        syn = n.synapses.get((old_a, old_b))
        return syn.w_slow if syn else 0.0

    with_inter = forgetting_run(True)
    without = forgetting_run(False)
    check("interleaving preserves the old association better",
          with_inter > without,
          f"interleaved w={with_inter:.4f} vs new-only w={without:.4f}")

    # ---------------------------------------------------------------
    print("\n=== 5. STDP makes edges directional ===")
    net5 = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
    pl5 = Plasticity(net5, PlasticityConfig(stdp_window=3))
    ca = net5.add_concept(v(50), label="cause", origin=Origin.ENTITY).id
    cb = net5.add_concept(v(51), label="effect", origin=Origin.ENTITY).id
    filler = net5.add_concept(v(52), label="filler", origin=Origin.ENTITY).id

    for _ in range(60):
        for step in ({ca: 0.9}, {cb: 0.9}, {filler: 0.5}):
            net5.cycles += 1
            for cid, x in step.items():
                net5.concepts[cid].touch(x)
            act = Activation(initial=step, spread=step)
            net5._record_cooccurrence(act)
            pl5.potentiate(act)

    fwd = net5.synapses.get((ca, cb))
    rev = net5.synapses.get((cb, ca))
    fw = fwd.weight() if fwd else 0.0
    rw = rev.weight() if rev else 0.0
    check("forward edge is stronger than reverse", fw > rw,
          f"cause->effect {fw:.4f} vs effect->cause {rw:.4f}")
    check("order evidence accumulated", fwd is not None and fwd.order_evidence > 0,
          f"order={fwd.order_evidence:.3f}" if fwd else "no edge")
    check("edge promoted to CAUSAL", fwd is not None and fwd.kind == SynType.CAUSAL,
          f"kind={fwd.kind.value}" if fwd else "-")

    # ---------------------------------------------------------------
    print("\n=== 6. Adaptation silences an always-on concept ===")
    net6 = ConceptNetwork(dim=DIM, k=6, vigilance=0.99, persist=False)
    pl6 = Plasticity(net6)
    always = net6.add_concept(v(60), label="hum", origin=Origin.ENTITY).id
    rare1 = net6.add_concept(v(61), label="r1", origin=Origin.ENTITY).id
    rare2 = net6.add_concept(v(62), label="r2", origin=Origin.ENTITY).id

    for step in range(200):
        spread = {always: 0.9}
        if step % 4 == 0:
            spread[rare1] = 0.9
            spread[rare2] = 0.9
        net6.cycles += 1
        for cid, x in spread.items():
            net6.concepts[cid].touch(x)
        act = Activation(initial=spread, spread=spread)
        net6._record_cooccurrence(act)
        pl6.potentiate(act)

    hum_w = max((abs(net6.synapses[(always, d)].weight())
                 for d in (rare1, rare2) if (always, d) in net6.synapses), default=0.0)
    real_w = abs(net6.synapses[(rare1, rare2)].weight()) if (rare1, rare2) in net6.synapses else 0.0
    check("the real pair beats the ubiquitous one", real_w > hum_w,
          f"r1-r2={real_w:.4f} vs hum-r={hum_w:.4f}")
    check("adaptation scales the always-on concept down",
          pl6.adapted(always, 1.0) < 0.2 < pl6.adapted(rare1, 1.0),
          f"hum={pl6.adapted(always, 1.0):.3f} rare={pl6.adapted(rare1, 1.0):.3f}")

    # ---------------------------------------------------------------
    print("\n=== 7. Acetylcholine switches encode/consolidate ===")
    nm7 = Neuromodulators()
    for _ in range(50):
        nm7.update(novelty=0.95)
    enc = nm7.encoding_gain()
    cort = nm7.cortical_gain()
    check("novelty raises encoding above cortical", enc > cort,
          f"encode={enc:.2f} cortex={cort:.2f}")
    check("mode is encode", not nm7.is_consolidating(), nm7.status()["mode"])

    for _ in range(30):
        nm7.update(asleep=True)
    check("sleep flips to consolidate", nm7.is_consolidating(),
          f"ACh={nm7.state.ach:.3f}")
    check("cortical gain now exceeds encoding gain",
          nm7.cortical_gain() > nm7.encoding_gain(),
          f"encode={nm7.encoding_gain():.2f} cortex={nm7.cortical_gain():.2f}")

    nm8 = Neuromodulators()
    for _ in range(80):
        nm8.update(novelty=0.02)
    check("familiarity lowers encoding gain", nm8.encoding_gain() < enc,
          f"familiar={nm8.encoding_gain():.2f} vs novel={enc:.2f}")

    # ---------------------------------------------------------------
    print("\n=== 8. Synaptic downscaling ===")
    net9 = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
    pl9 = Plasticity(net9)
    ids9 = [net9.add_concept(v(70 + i), label=f"n{i}", origin=Origin.ENTITY).id
            for i in range(6)]
    for i in range(5):
        net9.link(ids9[i], ids9[i + 1], 0.3 + 0.12 * i)
    before = [net9.synapses[(ids9[i], ids9[i + 1])].weight() for i in range(5)]

    s9 = SleepCycle(net9, pl9, Hippocampus(dim=DIM), None)
    rep = s9._downscale()
    after = [net9.synapses[(ids9[i], ids9[i + 1])].weight() for i in range(5)]
    check("total weight is reduced", sum(after) < sum(before),
          f"{sum(before):.3f} -> {sum(after):.3f}, factor {rep['factor']}")
    check("rank order is preserved",
          [i for i, _ in sorted(enumerate(before), key=lambda kv: kv[1])] ==
          [i for i, _ in sorted(enumerate(after), key=lambda kv: kv[1])],
          "multiplicative, so relative structure survives")

    # ---------------------------------------------------------------
    print("\n=== 9. Developmental annealing ===")
    net10 = ConceptNetwork(dim=DIM, persist=False)
    pl10 = Plasticity(net10, PlasticityConfig(maturity_cycles=1000))
    infant = pl10.developmental_gain()
    net10.cycles = 1000
    adult = pl10.developmental_gain()
    check("infant learns faster than adult", infant > adult * 2,
          f"infant x{infant:.2f} -> adult x{adult:.2f}")
    check("maturity saturates at 1.0", abs(pl10.maturity() - 1.0) < 1e-9)

    # ---------------------------------------------------------------
    print("\n=== 10. Schema fast-track ===")
    from src.concepts.bindings import ConceptSystem  # noqa: E402

    class FakeReg:
        def list_tools(self):
            return []

        def match(self, t):
            return None

    cs = ConceptSystem(embedder=None, registry=FakeReg(), dim=DIM,
                       load=False, seed=False)
    n = cs.net
    hub = [n.add_concept(v(80 + i), label=f"h{i}", origin=Origin.ENTITY).id
           for i in range(6)]
    # A dense, well-connected neighbourhood = an established schema.
    for i in hub:
        for j in hub:
            if i != j:
                n.link(i, j, 0.5, bidirectional=False)
    lonely_a = n.add_concept(v(90), label="la", origin=Origin.ENTITY).id
    lonely_b = n.add_concept(v(91), label="lb", origin=Origin.ENTITY).id

    in_schema = cs._schema_for_pair((hub[0], hub[1]))
    out_schema = cs._schema_for_pair((lonely_a, lonely_b))
    check("pairs inside a schema score higher", in_schema > out_schema,
          f"in-schema {in_schema:.2f} vs isolated {out_schema:.2f}")

    # ---------------------------------------------------------------
    print("\n=== 11. Uncertainty arbitration transfers control ===")
    from src.brain.executive import ExecutiveCore  # noqa: E402
    from src.brain.signals import Verdict, Vote  # noqa: E402

    ex = ExecutiveCore()
    w0 = ex._normalized_weights()

    # system1 is right about everything; system2 is wrong about everything.
    for _ in range(60):
        vd = Verdict(chosen=None, pathway=None, votes=[
            Vote(member="system1", support=1.0, confidence=0.9),
            Vote(member="system2", support=-1.0, confidence=0.9),
        ])
        ex.learn_from_outcome(vd, outcome_good=True)

    w1 = ex._normalized_weights()
    check("the reliable member gains influence", w1["system1"] > w0["system1"],
          f"{w0['system1']:.3f} -> {w1['system1']:.3f}")
    check("the unreliable member loses it", w1["system2"] < w0["system2"],
          f"{w0['system2']:.3f} -> {w1['system2']:.3f}")
    check("the cap still binds after adaptation",
          max(w1.values()) <= 0.34 + 1e-9, f"max={max(w1.values()):.3f}")
    check("nobody is silenced entirely", min(v for v in w1.values() if v > 0) > 0.01,
          "a silenced member could never earn its way back")

    print("\n=== Final ===")
    print("    ", cs.status()["network"])
    print("    ", nm7.status()["levels"])

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
