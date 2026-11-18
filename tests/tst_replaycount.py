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
    pl.replay_mode = False
    act = n.perceive(v, allow_growth=False, learn_prototypes=False)
    pl.potentiate(act)
    check("one live sighting: seen once", seen() == 1, str(seen()))


def r3_noises():
    print("\n=== R3. Noises are not concepts ===")
    for t in ["Uh-oh.", "Back on some.", "I can sound.", "Mm.", "Ugh, ugh."]:
        check(f"from the microphone, no concept: {t!r}", is_telemetry(t, "microphone"))
    for t in ["hello", "yes", "open Chrome"]:
        check(f"typed, it is on purpose: {t!r}", not is_telemetry(t, "console"))
    for t in ["what time is it", "remind me to call mom tomorrow", "the lecturer said a vector is a keyword"]:
        check(f"a real sentence from the microphone is still learned from: {t!r}", not is_telemetry(t, "microphone"))


if __name__ == "__main__":
    r1_count_flag()
    r2_potentiation()
    r3_noises()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
