"""
tst_conceptcleanup.py -- concepts grown from the machine's own readings are taken out of a saved graph, and nothing else.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_conceptcleanup.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import concept_cleanup as cc  # noqa: E402
from src.concepts.network import ConceptNetwork  # noqa: E402
from src.concepts.model import Origin  # noqa: E402

check = tw.check
DIM = 32


def rnd(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def build():
    d = tempfile.mkdtemp()
    net = ConceptNetwork(dim=DIM)
    ids = {}
    ids["intent"] = net.add_concept(rnd(1), label="calculate", origin=Origin.INTENT, intent="calculate").id
    ids["mem"] = net.add_concept(rnd(2), origin=Origin.GROWN, seed_text="memory at 93%").id
    ids["screen"] = net.add_concept(rnd(3), origin=Origin.GROWN, seed_text="screen changed while in Discord").id
    ids["switch"] = net.add_concept(rnd(4), origin=Origin.GROWN, seed_text="switched to Chrome (was on A.S.H for 40s)").id
    ids["person"] = net.add_concept(rnd(5), origin=Origin.GROWN, seed_text="remind me to call my mother tomorrow").id
    ids["lecture"] = net.add_concept(rnd(6), origin=Origin.GROWN, seed_text="a vector is a keyword embedding").id
    net.link(ids["mem"], ids["person"], 0.5)
    net.link(ids["person"], ids["lecture"], 0.5)
    net.save(d)
    return d, ids


def c1():
    print("\n=== C1. The cleanup ===")
    d, ids = build()
    p = cc.plan(d)
    dropped = {c["id"] for c in p["drop"]}
    check("the three readings are found", dropped == {ids["mem"], ids["screen"], ids["switch"]}, str(dropped))
    check("a seeded concept, a person's sentence and a lecture line are not", not ({ids["intent"], ids["person"], ids["lecture"]} & dropped))
    before = open(os.path.join(d, "graph.json"), "rb").read()
    check("looking changes nothing", open(os.path.join(d, "graph.json"), "rb").read() == before and p["total"] == 6)
    backup = cc.apply(p)
    check("a backup of the folder was made", os.path.isdir(backup) and os.path.exists(os.path.join(backup, "graph.json")))
    net = ConceptNetwork(dim=DIM)
    check("the graph loads again", net.load(d))
    left = set(net.concepts)
    check("only the readings are gone", left == {ids["intent"], ids["person"], ids["lecture"]}, str(left))
    check("no wire points at a removed concept, and the wire between the two kept ones survived", all(s in left and t in left for (s, t) in net.synapses) and (ids["person"], ids["lecture"]) in net.synapses)
    check("run again: nothing to do", not cc.plan(d)["drop"])


if __name__ == "__main__":
    c1()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
