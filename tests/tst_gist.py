"""
tst_gist.py -- replay compiles episodes into concepts: an entity that keeps turning up where a concept was active becomes part of it.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_gist.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.concepts.gist import compile_gist, entities_of  # noqa: E402
from src.concepts.network import ConceptNetwork  # noqa: E402
from src.concepts.model import Origin  # noqa: E402

check = tw.check
DIM = 8


def vec(i):
    v = np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def ep(concepts, entities):
    return NS(concepts=concepts, related_entities=entities)


def g1_compile():
    print("\n=== G1. Compiling ===")
    net = ConceptNetwork(dim=DIM)
    a = net.add_concept(vec(1), origin=Origin.GROWN, seed_text="my dad's clinic").id
    b = net.add_concept(vec(2), origin=Origin.GROWN, seed_text="lecture").id
    c = net.add_concept(vec(3), origin=Origin.GROWN, seed_text="few").id
    eps = [ep([a], ["afif", "x1"]), ep([a], ["afif"]), ep([a], ["afif", "hebron"]), ep([a], ["afif", "hebron"]), ep([a, b], ["x2", "hebron"]),
           ep([b], ["y"]), ep([b], ["z"]), ep([b], ["w"]),
           ep([c], ["q"]), ep([c], ["q"])]
    n = compile_gist(net, eps)
    check("the concept learned who keeps turning up", set(net.concepts[a].entities) == {"afif", "hebron"}, str(net.concepts[a].entities))
    check("a one-off in a concept's episodes is not learned", net.concepts[b].entities == {})
    check("two episodes is too few", net.concepts[c].entities == {})
    check("only the changed concept counts", n == 1, str(n))
    check("run again: nothing changes", compile_gist(net, eps) == 0)
    check("entities_of returns them, strongest first", entities_of(net, [b, a]) == ["afif", "hebron"], str(entities_of(net, [b, a])))


def g2_persist():
    print("\n=== G2. Saved and loaded ===")
    net = ConceptNetwork(dim=DIM)
    a = net.add_concept(vec(1), origin=Origin.GROWN, seed_text="x").id
    net.concepts[a].entities = {"afif": 4}
    d = tempfile.mkdtemp()
    net.save(d)
    net2 = ConceptNetwork(dim=DIM)
    check("it loads", net2.load(d))
    check("the entities survived", net2.concepts[a].entities == {"afif": 4}, str(net2.concepts[a].entities))


if __name__ == "__main__":
    g1_compile()
    g2_persist()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
