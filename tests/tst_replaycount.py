"""
tst_replaycount.py -- "seen" is what was lived, not what was replayed; noises do not become concepts.

The mind page showed wires "seen 1,232x" and "permanent" between sentences said a handful of times: the sleep replays the same traces
hundreds of times (with replacement, from a small pool), and each replay counted the concepts as active and the wires between them as
potentiated again -- so one afternoon's remark became "evidence" -- and "Back on some.", "Uh-oh.", "I can sound." (noises from the
microphone) each grew a concept that became a hub 111 other concepts pointed at.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_replaycount.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.concepts.network import ConceptNetwork  # noqa: E402
from src.concepts.plasticity import Plasticity  # noqa: E402
from src.concepts.model import Origin  # noqa: E402
from src.concepts.telemetry import is_telemetry  # noqa: E402

check = tw.check
DIM = 16


def rnd(i):
    v = np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def r1_count_flag():
    print("\n=== R1. Replay does not count ===")
    n = ConceptNetwork(dim=DIM)
    a = n.add_concept(rnd(1), seed_text="a", origin=Origin.GROWN)
    n.perceive(rnd(1), allow_growth=False, learn_prototypes=False)
    check("a live perception counts the concept", a.activations == 1, str(a.activations))
    for _ in range(50):
        n.perceive(rnd(1), allow_growth=False, learn_prototypes=False, count=False)
    check("fifty replays count it no more", a.activations == 1, str(a.activations))


def r2_potentiation():
    print("\n=== R2. Replay does not make a wire 'seen' ===")
    n = ConceptNetwork(dim=DIM)
    ids = [n.add_concept(rnd(i), seed_text=f"c{i}", origin=Origin.GROWN).id for i in range(1, 4)]
    pl = Plasticity(n)
    v = rnd(1) + rnd(2) + rnd(3)

    def seen():
        return max((s.potentiations for s in n.synapses.values()), default=0)
    pl.replay_mode = True
    for _ in range(40):
        act = n.perceive(v, allow_growth=False, learn_prototypes=False, count=False)
        pl.potentiate(act)
    check("forty replays: no wire has been 'seen'", seen() == 0, str(seen()))
    for _ in range(40):
        act = n.perceive(v, allow_growth=False, learn_prototypes=False, count=False)
        pl.replay_token = 7
        pl.potentiate(act)
        pl.replay_token = None
    check("one stored memory replayed forty times is one sighting", seen() == 1, str(seen()))
    act = n.perceive(v, allow_growth=False, learn_prototypes=False, count=False)
    pl.replay_token = 8
    pl.potentiate(act)
    check("a second, different memory holding the pair is a second", seen() == 2, str(seen()))
    pl.replay_mode = False
    act = n.perceive(v, allow_growth=False, learn_prototypes=False)
    pl.potentiate(act)
    check("one live sighting: seen once more", seen() == 3, str(seen()))


def r3_noises():
    print("\n=== R3. Noises are not concepts ===")
    for t in ["Uh-oh.", "Back on some.", "I can sound.", "Mm.", "Ugh, ugh."]:
        check(f"from the microphone, no concept: {t!r}", is_telemetry(t, "microphone"))
    for t in ["hello", "yes", "open Chrome"]:
        check(f"typed, it is on purpose: {t!r}", not is_telemetry(t, "console"))
    for t in ["what time is it", "remind me to call mom tomorrow", "the lecturer said a vector is a keyword"]:
        check(f"a real sentence from the microphone is still learned from: {t!r}", not is_telemetry(t, "microphone"))


def r4_cap():
    print("\n=== R4. A hub earns its wires ===")
    n = ConceptNetwork(dim=DIM)
    hub = n.add_concept(rnd(500), seed_text="hub", origin=Origin.GROWN).id
    srcs = [n.add_concept(rnd(600 + i), seed_text=f"s{i}", origin=Origin.GROWN).id for i in range(60)]
    for i, s_ in enumerate(srcs):
        n.link(s_, hub, 0.1 + i * 0.01)
    frozen = n.synapse(srcs[0], hub)
    frozen.frozen = True
    pl = Plasticity(n)
    check("setup: 60 wires in", len(n._in[hub]) == 60)
    removed = pl._cap_inputs()
    check("down to the cap, 20 removed", len(n._in[hub]) == pl.cfg.max_inputs and removed == 20, f"{len(n._in[hub])} {removed}")
    check("the strongest stayed", all(abs(n._in[hub][s_].weight()) >= 0.1 + 20 * 0.01 - 1e-6 for s_ in n._in[hub] if s_ != srcs[0]))
    check("a frozen wire stayed", srcs[0] in n._in[hub])
    check("the synapse table and the in/out indexes agree", all((s_, hub) in n.synapses for s_ in n._in[hub]) and all(hub in n._out[s_] for s_ in n._in[hub]))
    check("run again: nothing", pl._cap_inputs() == 0)


def r5_causal():
    print("\n=== R5. Arrows are causal only across days ===")
    from src.concepts.model import SynType, Activation
    from src.concepts.plasticity import PlasticityConfig
    n = ConceptNetwork(dim=DIM, k=4, vigilance=0.99, persist=False)
    pl = Plasticity(n, PlasticityConfig(stdp_window=3))
    a = n.add_concept(rnd(70), label="a", origin=Origin.ENTITY).id
    b = n.add_concept(rnd(71), label="b", origin=Origin.ENTITY).id
    f = n.add_concept(rnd(72), label="f", origin=Origin.ENTITY).id

    def run(rounds):
        for _ in range(rounds):
            for step in ({a: 0.9}, {b: 0.9}, {f: 0.5}):
                n.cycles += 1
                for cid, x in step.items():
                    n.concepts[cid].touch(x)
                act = Activation(initial=step, spread=step)
                n._record_cooccurrence(act)
                pl.potentiate(act)
    run(80)
    fwd = n.synapses.get((a, b))
    check("within one afternoon it has plenty of order evidence but is not causal", fwd is not None and fwd.order_evidence >= pl.cfg.causal_threshold
          and fwd.kind != SynType.CAUSAL, f"{fwd.order_evidence if fwd else None} {fwd.kind if fwd else None}")
    fwd.created -= 3 * 86400
    run(5)
    check("once it spans days it is", n.synapses[(a, b)].kind == SynType.CAUSAL, str(n.synapses[(a, b)].kind))
    # two arrows that claim to cause each other lose it in the sleep
    x, y = n.synapse(a, f), n.synapse(f, a)
    x.kind = y.kind = SynType.CAUSAL
    x.order_evidence = y.order_evidence = 0.9
    check("both ways, no clear direction: the audit takes it away from both", pl._audit_causal() >= 2 and x.kind != SynType.CAUSAL and y.kind != SynType.CAUSAL)


def r6_lonely():
    print("\n=== R6. A concept that connects to nothing goes ===")
    import time as _t
    n = ConceptNetwork(dim=DIM)
    old = _t.time() - 5 * 86400
    lonely = n.add_concept(rnd(80), origin=Origin.GROWN, seed_text="read the screen in a window")
    lonely.created = old
    wired = n.add_concept(rnd(81), origin=Origin.GROWN, seed_text="wired")
    wired.created = old
    other = n.add_concept(rnd(82), origin=Origin.GROWN, seed_text="other")
    other.created = old
    n.link(wired.id, other.id, 0.3)
    newone = n.add_concept(rnd(83), origin=Origin.GROWN, seed_text="new")
    busy = n.add_concept(rnd(84), origin=Origin.GROWN, seed_text="busy but wireless")
    busy.created, busy.activations = old, 50
    labelled = n.add_concept(rnd(85), origin=Origin.GROWN, label="named", seed_text="named")
    labelled.created = old
    seed = n.add_concept(rnd(86), origin=Origin.INTENT, label="calc", intent="calc")
    seed.created = old
    pl = Plasticity(n)
    check("one is dropped", pl._prune_lonely() == 1 and lonely.id not in n.concepts, str(sorted(n.concepts)))
    check("a wired one, a new one, a busy one, a labelled one and a seeded one stay",
          all(c in n.concepts for c in (wired.id, other.id, newone.id, busy.id, labelled.id, seed.id)))
    check("run again: nothing", pl._prune_lonely() == 0)


if __name__ == "__main__":
    r1_count_flag()
    r2_potentiation()
    r3_noises()
    r4_cap()
    r5_causal()
    r6_lonely()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
