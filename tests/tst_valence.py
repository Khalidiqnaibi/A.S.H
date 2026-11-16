"""
tst_valence.py -- a concept remembers how the user felt when it was what they were talking about.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_valence.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain.brain import Brain  # noqa: E402
from src.concepts.network import ConceptNetwork  # noqa: E402
from src.concepts.model import Origin  # noqa: E402

check = tw.check
DIM = 8


def vec(i):
    v = np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def v1_feel():
    print("\n=== V1. Averaging ===")
    net = ConceptNetwork(dim=DIM)
    c = net.add_concept(vec(1), origin=Origin.GROWN, seed_text="exam stress")
    for _ in range(6):
        c.feel(-1.0)
    check("a topic that keeps coming up in a bad mood turns negative", c.valence < -0.7, str(c.valence))
    for _ in range(3):
        c.feel(1.0)
    check("a few good moments move it, not erase it", -0.7 < c.valence < 0.0, str(c.valence))
    c2 = net.add_concept(vec(2), origin=Origin.GROWN, seed_text="x")
    c2.feel(1.0, weight=0.0)
    check("a reading with no confidence changes nothing (but is counted)", c2.valence == 0.0 and c2.valence_n == 1)


def v2_brain():
    print("\n=== V2. The brain feeds it ===")
    net = ConceptNetwork(dim=DIM)
    a = net.add_concept(vec(1), origin=Origin.GROWN, seed_text="a").id
    b = net.add_concept(vec(2), origin=Origin.GROWN, seed_text="b").id
    froz = net.add_concept(vec(3), origin=Origin.INTENT, label="calc", intent="calc")
    froz.frozen = True
    brain = NS(concepts=NS(net=net), _SENTIMENT_VALUE=Brain._SENTIMENT_VALUE, _topic={})
    act = NS(top=lambda n: [(a, 1.0), (b, 0.8), (froz.id, 0.5)])
    for _ in range(4):
        Brain._feel(brain, act, {"sentiment": "NEGATIVE", "confidence": 0.9})
    check("the concepts it lit took the mood", net.concepts[a].valence < -0.5 and net.concepts[b].valence < -0.5)
    check("a frozen one did not", froz.valence == 0.0)
    Brain._feel(brain, act, {"sentiment": "unknown", "confidence": 0.0})
    check("'unknown' is not a reading", net.concepts[a].valence_n == 4)
    brain._topic = {a: 1.0}
    check("the topic's valence is read back", Brain.topic_valence(brain) < -0.5, str(Brain.topic_valence(brain)))
    brain._topic = {}
    check("no topic, no opinion", Brain.topic_valence(brain) == 0.0)


def v3_persist():
    print("\n=== V3. Saved ===")
    net = ConceptNetwork(dim=DIM)
    c = net.add_concept(vec(1), origin=Origin.GROWN, seed_text="x")
    for _ in range(4):
        c.feel(-1.0)
    d = tempfile.mkdtemp()
    net.save(d)
    n2 = ConceptNetwork(dim=DIM)
    n2.load(d)
    c2 = n2.concepts[c.id]
    check("valence and its count survive a restart", abs(c2.valence - c.valence) < 1e-3 and c2.valence_n == 4, f"{c2.valence} {c2.valence_n}")


if __name__ == "__main__":
    v1_feel()
    v2_brain()
    v3_persist()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
