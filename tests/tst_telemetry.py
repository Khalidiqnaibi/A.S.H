"""
tst_telemetry.py -- the machine's own readings are perceived but never become concepts.

"memory at 93%", battery, "screen changed", window switches and motion made 23 concepts in the graph and took a quarter of
all activation. They are the body's state (homeostasis already has them); the graph is for what the person means.

  T1  What counts as telemetry (and what never does: a person's words, whatever carries them)
  T2  A telemetry percept does not grow a concept, drag a prototype, or strengthen wires; a person's sentence still does
  T3  The brain's ambient path passes it on

Run:  PYTHONIOENCODING=utf-8 python tests/tst_telemetry.py
"""

import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.concepts import ConceptSystem  # noqa: E402

check = tw.check
DIM = 32


def rnd(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def is_telemetry():
    from src.concepts.telemetry import is_telemetry as f
    return f


def t1_what():
    print("\n=== T1. What is telemetry ===")
    f = is_telemetry()
    for text, source in [("memory at 93%", "system"), ("CPU at 91%", "system"), ("battery at 15% and unplugged", "system"),
                         ("screen changed while in Discord", "screen"), ("switched to Chrome (was on A.S.H for 480s)", "screen"),
                         ("motion in camera view (delta 0.16)", "camera")]:
        check(f"{text!r} from {source} is telemetry", f(text, source))
    for text, source in [("the lecturer said a vector is a keyword", "microphone"), ("what time is it", "console"),
                         ("memory at 93%", "microphone"), ("remind me to check memory usage", "console"),
                         ("I use 90% of my memory for tabs", "microphone")]:
        check(f"{text!r} from {source} is not", not f(text, source))


def t2_graph():
    print("\n=== T2. In the graph ===")
    cs = ConceptSystem(None, None, dim=DIM, load=False, seed=False)
    cs.net.vigilance = 0.9
    n0 = len(cs.net.concepts)
    for i in range(10):
        cs.perceive(rnd(100 + i), text=f"memory at {80 + i}%", telemetry=True)
    check("ten different telemetry percepts grow no concept", len(cs.net.concepts) == n0, f"{n0} -> {len(cs.net.concepts)}")
    w0 = cs.plasticity.stats.get("potentiated", 0) if hasattr(cs.plasticity, "stats") else 0
    cs.perceive(rnd(200), text="battery at 10%", telemetry=True)
    w1 = cs.plasticity.stats.get("potentiated", 0) if hasattr(cs.plasticity, "stats") else 0
    check("...and strengthen no wires", w1 == w0)
    for i in range(10):
        cs.perceive(rnd(300 + i), text=f"a sentence about topic {i}")
    check("whereas a person's sentences still grow the graph", len(cs.net.concepts) > n0, str(len(cs.net.concepts)))
    act = cs.perceive(rnd(100), text="memory at 80%", telemetry=True)
    check("a telemetry percept is still perceived (an activation to explain)", act is not None and bool(act.spread))


def t3_brain():
    print("\n=== T3. The ambient path ===")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "brain", "brain.py"), encoding="utf-8").read()
    check("Brain.observe hands telemetry to the concept layer as telemetry",
          re.search(r"self\._conceive\(bundle, telemetry=is_telemetry\(text, source\)\)", src) is not None and "from src.concepts.telemetry import is_telemetry" in src)


if __name__ == "__main__":
    t1_what()
    t2_graph()
    t3_brain()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
