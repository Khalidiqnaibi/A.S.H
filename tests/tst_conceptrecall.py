"""
tst_conceptrecall.py -- recall follows the concepts, not only the words.

An episode remembers the concepts that were active when it was laid down. When the same concepts light up again an old episode that
shares no words with the question comes back, as an association -- at most two, never the sentence that is this very input.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_conceptrecall.py
"""

import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_MEMORY_DIR", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.memory.memory_router import MemoryRouter  # noqa: E402

check = tw.check


def ep(i, summary, concepts, age=0):
    return NS(episode_id=i, summary=summary, concepts=concepts, timestamp=time.time() - age, importance=0.5)


def router(eps, active):
    r = MemoryRouter.__new__(MemoryRouter)
    r.episodic = NS(store=NS(episodes={e.episode_id: e for e in eps}))
    r.active_concepts = active
    return r


def c1_overlap():
    print("\n=== C1. Shared concepts bring an episode back ===")
    eps = [ep("a", "Immortal: afif runs the clinic in the old town", [1, 2, 3], 100),
           ep("b", "Immortal: the weather is nice", [9, 8, 7], 50),
           ep("c", "Immortal: one concept only", [1, 40, 41], 10)]
    r = router(eps, [1, 2, 3, 5])
    got = [e.episode_id for e in r._by_concepts("where does my dad work")]
    check("the one that shares three comes back", got == ["a"], str(got))
    check("one concept in common is not enough", "c" not in got)


def c2_order_and_cap():
    print("\n=== C2. Most in common first, at most two, not this very input ===")
    eps = [ep("a", "x one", [1, 2], 300), ep("b", "x two", [1, 2, 3], 200), ep("c", "x three", [1, 2, 3, 4], 100),
           ep("d", "where does my dad work", [1, 2, 3, 4], 1)]
    r = router(eps, [1, 2, 3, 4])
    got = [e.episode_id for e in r._by_concepts("where does my dad work")]
    check("two at most, the biggest overlap first, the input itself skipped", got == ["c", "b"], str(got))


def c3_nothing_active():
    print("\n=== C3. Nothing active, nothing found ===")
    r = router([ep("a", "x", [1, 2, 3])], [])
    check("no active concepts", r._by_concepts("hi") == [])
    r = router([ep("a", "x", [1, 2, 3])], [1])
    check("one active concept is too little", r._by_concepts("hi") == [])
    r = router([NS(episode_id="o", summary="old", timestamp=0.0, importance=0.5)], [1, 2])
    check("an old episode without concepts is fine", r._by_concepts("hi") == [])


if __name__ == "__main__":
    c1_overlap()
    c2_order_and_cap()
    c3_nothing_active()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
