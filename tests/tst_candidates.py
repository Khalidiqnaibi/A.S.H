"""
tst_candidates.py -- a concept is made the way the brain makes one: from what came back, at sleep, not from one moment.

  K1  an unexplained input is an episode: it grows no concept on the spot, it is only noted as a candidate
  K2  a moment made of specifics (a duration, a clock time, a path, a link, a run of numbers) is not even a candidate
  K3  something that comes back often enough, on separate occasions, becomes a concept at sleep, its prototype the gist
  K4  something that comes back a lot but in one sitting does not; something that never comes back is forgotten
  K5  candidates survive a restart; and a network not asked for deferred growth still grows at once (what the old tests rely on)
  K6  only so many new concepts a night, and none for something a concept now explains
  K7  a concept is not dragged toward an input it matches only weakly (nothing grown now takes it), but learns from a real match

Everything runs on a small made-up network in a temp folder.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_candidates.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import numpy as np  # noqa: E402

import tst_wondering as tw  # noqa: E402
from src.concepts.candidates import is_episode_like  # noqa: E402
from src.concepts.network import ConceptNetwork  # noqa: E402

check = tw.check
DIM = 32


def rnd(seed, base=None, noise=0.0):
    r = np.random.default_rng(seed)
    v = (base if base is not None else r.standard_normal(DIM)) + noise * r.standard_normal(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def deferred(**kw):
    return ConceptNetwork(dim=DIM, persist=False, deferred_growth=True, **kw)


def k1_episode_first():
    print("\n=== K1. One unexplained input is an episode ===")
    n = deferred()
    a = n.perceive(rnd(1), text="my cousin is visiting from Amman")
    check("it grows no concept", a.grown is None and len(n.concepts) == 0)
    check("it is noted as a candidate, and still counts as novel", len(n.candidates.items) == 1 and a.novelty > 0.5)
    check("the same input again is the same candidate, not a second one", (n.perceive(rnd(1), text="my cousin is visiting") and len(n.candidates.items) == 1))


def k2_specifics():
    print("\n=== K2. A moment of specifics is not a candidate ===")
    for t in ("switched to Windows PowerShell (was on Claude for 9s)", "at 14:35 the build failed", "see https://example.com/x",
              r"saved to C:\Users\me\file.txt", "order 4521 total 88 paid 12", "it was 2026-11-02"):
        check(f"episode-like: {t[:40]}", is_episode_like(t))
    for t in ("my cousin is visiting from Amman", "I like jazz in the evening", "what is the capital of France", "my favorite number is 4242", ""):
        check(f"not episode-like: {t[:40]!r}", not is_episode_like(t))
    n = deferred()
    n.perceive(rnd(2), text="screen changed while in Code for 12s")
    check("such an input leaves no candidate", len(n.candidates.items) == 0)


def k3_recurrence_becomes_concept():
    print("\n=== K3. What comes back becomes a concept at sleep ===")
    n = deferred()
    base = rnd(3)
    t0 = 1_000_000.0
    for i in range(4):
        v = rnd(10 + i, base=base, noise=0.05)
        c = n.candidates.note(v, f"jazz sentence {i}", t0 + i * 3600)     # four arrivals, an hour apart: four occasions
    check("it is ripe", len(n.candidates.ripe()) == 1 and n.candidates.ripe()[0].occasions >= 2)
    r = n.promote_candidates(now=t0 + 5 * 3600)
    check("one concept is made, and nothing is left waiting", r["promoted"] == 1 and len(n.concepts) == 1 and r["waiting"] == 0)
    c = next(iter(n.concepts.values()))
    check("its prototype is the gist: close to every arrival, closer to the middle than to any one", float(c.prototype @ base) > 0.9)
    check("it is named by what was actually said", c.title() and "jazz sentence" in c.title())


def k4_one_sitting_and_forgotten():
    print("\n=== K4. One sitting is not enough; what never returns fades ===")
    n = deferred()
    base = rnd(4)
    for i in range(10):
        n.candidates.note(rnd(30 + i, base=base, noise=0.05), "same minute", 5000.0 + i)     # ten times in ten seconds
    check("ten times in one sitting is still one occasion, so not ripe", n.candidates.ripe() == [])
    n.candidates.note(rnd(99), "a one-off", 5000.0)
    r = n.promote_candidates(now=5000.0 + 15 * 86400)
    check("a fortnight later with no return both are forgotten", r["promoted"] == 0 and r["waiting"] == 0)


def k5_restart_and_legacy():
    print("\n=== K5. Candidates survive a restart; the plain network still grows at once ===")
    a = ConceptNetwork(dim=DIM, deferred_growth=True)
    for i in range(2):
        a.candidates.note(rnd(5 + i, base=rnd(5), noise=0.05), f"line {i}", 100.0 + i * 5000)
    folder = tempfile.mkdtemp()
    a.save(folder)
    b = ConceptNetwork(dim=DIM, deferred_growth=True)
    check("it loads", b.load(folder))
    cand = b.candidates.items[0]
    check("the candidate, its count, occasions and sentences are back", len(b.candidates.items) == 1 and cand.n == 2 and cand.occasions == 2
          and cand.texts == ["line 0", "line 1"])
    legacy = ConceptNetwork(dim=DIM, persist=False)
    g = legacy.perceive(rnd(7), text="anything at all")
    check("a network not asked for deferred growth grows the concept on the spot", g.grown is not None and legacy.candidates is None)
    old = ConceptNetwork(dim=DIM)
    old.add_concept(rnd(8))
    f2 = tempfile.mkdtemp()
    old.save(f2)
    c = ConceptNetwork(dim=DIM, deferred_growth=True)
    check("a graph saved before candidates existed loads into a deferred one with none", c.load(f2) and c.candidates.items == [])


def k6_per_night_and_known():
    print("\n=== K6. Few a night, none for what is already explained ===")
    n = deferred()
    for k in range(12):
        base = rnd(200 + k)
        for i in range(3):
            n.candidates.note(rnd(300 + k * 10 + i, base=base, noise=0.05), f"topic {k}", 1000.0 + i * 5000)
    r = n.promote_candidates(now=40000.0)
    check("only eight are made in one night, the rest wait", r["promoted"] == 8 and r["waiting"] == 4)
    m = deferred()
    base = rnd(500)
    m.add_concept(base)
    for i in range(3):
        m.candidates.note(rnd(510 + i, base=base, noise=0.05), "known", 1000.0 + i * 5000)
    check("a candidate a concept now explains makes no second concept", m.promote_candidates(now=30000.0)["promoted"] == 0 and len(m.concepts) == 1)


def k7_weak_match_teaches_nothing():
    print("\n=== K7. Only a real match teaches ===")
    n = deferred()
    base = rnd(600)
    c = n.add_concept(base)
    p0 = c.prototype.copy()
    r = np.random.default_rng(601)
    far = r.standard_normal(DIM).astype(np.float32)
    far -= (far @ base) * base                     # orthogonal to it, then pushed back to a weak match
    far = far / np.linalg.norm(far)
    weak = (0.35 * base + 0.94 * far).astype(np.float32)
    weak /= np.linalg.norm(weak)
    for _ in range(20):
        n.perceive(weak, text="something only a little like it")
    check("twenty weak matches leave the prototype where it was", float(c.prototype @ p0) > 0.9999)
    near = (0.9 * base + 0.43 * far).astype(np.float32)
    near /= np.linalg.norm(near)
    for _ in range(20):
        n.perceive(near, text="clearly it")
    check("real matches still move it", float(c.prototype @ p0) < 0.9999)


if __name__ == "__main__":
    for fn in (k1_episode_first, k2_specifics, k3_recurrence_becomes_concept, k4_one_sitting_and_forgotten, k5_restart_and_legacy,
               k6_per_night_and_known, k7_weak_match_teaches_nothing):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
