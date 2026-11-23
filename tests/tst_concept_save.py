"""
tst_concept_save.py -- the concept graph is saved as one file, so a crash cannot leave two generations side by side.

It was prototypes.npz then graph.json, each atomic but not together: a crash between them left the ids and vectors of one generation with the
metadata of another (concepts with no label or history, synapses to ids that did not exist). The metadata rides inside the npz now, in one swap;
the last good copy is kept; and the co-occurrence statistics the PMI correction reads survive a restart (they started from nothing every time).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_concept_save.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

OK = True
DIM = 16


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def vec(i):
    v = np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def net():
    from src.concepts.network import ConceptNetwork
    return ConceptNetwork(dim=DIM)


def main():
    from src.concepts.model import Origin
    print("\n=== C1. One generation ===")
    d = tempfile.mkdtemp()
    n = net()
    a = n.add_concept(vec(1), seed_text="first generation", origin=Origin.GROWN)
    b = n.add_concept(vec(2), seed_text="second", origin=Origin.GROWN)
    n.link(a.id, b.id, 0.4)
    n.co_counts[(a.id, b.id)] = 7
    n.save(d)
    check("saved", os.path.exists(os.path.join(d, "prototypes.npz")) and os.path.exists(os.path.join(d, "graph.json")))
    check("the metadata is inside the npz, beside the vectors", "graph" in np.load(os.path.join(d, "prototypes.npz")).files)
    n.concepts[a.id].seed_text = "second generation"
    n.save(d)
    # a crash between the two writes of the old way: the NEW vectors with the OLD graph.json
    old = json.load(open(os.path.join(d, "graph.json"), encoding="utf-8"))
    for c in old["concepts"]:
        c["seed_text"] = "STALE metadata from another generation"
    json.dump(old, open(os.path.join(d, "graph.json"), "w", encoding="utf-8"))
    m = net()
    ok = m.load(d)
    check("it loads", ok and len(m.concepts) == 2)
    check("the metadata is the one saved WITH the vectors, not the stale graph.json", m.concepts[a.id].seed_text == "second generation", m.concepts[a.id].seed_text)
    check("the wire and the co-occurrence count came back", len(m.synapses) == 2 and m.co_counts.get((a.id, b.id)) == 7, str(m.co_counts))

    print("\n=== C2. A damaged file ===")
    p = os.path.join(d, "prototypes.npz")
    check("the last good copy was kept", os.path.exists(p + ".bak"))
    open(p, "wb").write(b"not an npz")
    m2 = net()
    check("a torn prototypes.npz: loaded from the backup, not started fresh", m2.load(d) and len(m2.concepts) == 2)
    os.remove(p)
    os.remove(p + ".bak")
    check("nothing readable: False (start fresh), no crash", net().load(d) is False)

    print("\n=== C3. An older file still loads ===")
    d2 = tempfile.mkdtemp()
    n3 = net()
    x = n3.add_concept(vec(3), seed_text="old format", origin=Origin.GROWN)
    n3.save(d2)
    pz = np.load(os.path.join(d2, "prototypes.npz"))
    np.savez_compressed(os.path.join(d2, "prototypes.npz"), ids=pz["ids"], protos=pz["protos"])      # no 'graph' inside: written before this change
    m3 = net()
    check("the metadata is read from graph.json", m3.load(d2) and m3.concepts[x.id].seed_text == "old format")

    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
