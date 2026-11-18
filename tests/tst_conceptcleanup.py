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
    ids["own"] = net.add_concept(rnd(7), origin=Origin.GROWN, seed_text="You've been on Stremio-shell-ng for 15 minutes without typing.").id
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
    check("the three readings and ASH's own nudge are found", dropped == {ids["mem"], ids["screen"], ids["switch"], ids["own"]}, str(dropped))
    check("a seeded concept, a person's sentence and a lecture line are not", not ({ids["intent"], ids["person"], ids["lecture"]} & dropped))
    before = open(os.path.join(d, "graph.json"), "rb").read()
    check("looking changes nothing", open(os.path.join(d, "graph.json"), "rb").read() == before and p["total"] == 7)
    backup = cc.apply(p)
    check("a backup of the folder was made", os.path.isdir(backup) and os.path.exists(os.path.join(backup, "graph.json")))
    net = ConceptNetwork(dim=DIM)
    check("the graph loads again", net.load(d))
    left = set(net.concepts)
    check("only the readings are gone", left == {ids["intent"], ids["person"], ids["lecture"]}, str(left))
    check("no wire points at a removed concept, and the wire between the two kept ones survived", all(s in left and t in left for (s, t) in net.synapses) and (ids["person"], ids["lecture"]) in net.synapses)
    check("run again: nothing to do", not cc.plan(d)["drop"])


def c2_clamp():
    print("\n=== C2. Counts brought down ===")
    d, ids = build()
    net = ConceptNetwork(dim=DIM)
    net.load(d)
    syn = net.synapse(ids["person"], ids["lecture"])
    syn.potentiations = 1232
    other = net.synapse(ids["lecture"], ids["intent"])
    other.potentiations = 2
    w = syn.w_slow
    check("only the inflated ones are clamped", cc.clamp_counts(net) == 1 and syn.potentiations == 3 and other.potentiations == 2)
    check("their weights are untouched", syn.w_slow == w)
    p = cc.plan(d)
    p["clamp"] = True
    cc.apply(p)
    net2 = ConceptNetwork(dim=DIM)
    net2.load(d)
    check("through the tool, saved", all(s.potentiations <= 3 for s in net2.synapses.values()))


if __name__ == "__main__":
    c1()
    c2_clamp()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
