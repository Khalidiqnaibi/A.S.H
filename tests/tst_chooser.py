"""
tst_chooser.py -- "what kind of thing was that?", decided by meaning and learned from corrections.

  H1  a sentence none of the examples contained is still labelled by what it means
  H2  it answers only when clear: ordinary sentences stay unlabelled ("none"), a close call says nothing
  H3  a correction is kept for good (across a restart), moves a sentence from the wrong label, and teaches paraphrases
  H4  no encoder, a failing encoder, an empty sentence: it says nothing and never raises
  H5  in the runtime: made once per name when an encoder is configured, kept in the folder; none without one

The encoder here is a test stand-in (words that mean the same share a dimension); calibration against the real one is in tst_chooser_real.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_chooser.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import semantic_fake as sf  # noqa: E402
import tst_wondering as tw  # noqa: E402
from src.daemon.chooser import Chooser, clean  # noqa: E402

check = tw.check

SEEDS = {
    "thanks": ["thanks", "thank you very much", "cheers mate", "appreciate it"],
    "bye": ["later", "goodbye", "see you tomorrow", "gotta go bye"],
    "none": ["what is the weather today", "play some music", "who is ada lovelace", "tell me a joke", "open the browser", "what time is it"],
}


def make(path=None, **kw):
    return Chooser("test", sf.embed, SEEDS, path=path, accept=0.5, margin=0.05, **kw)


def h1_meaning():
    print("\n=== H1. By meaning ===")
    c = make()
    for said, want in [("thanks a lot", "thanks"), ("much obliged", "thanks"), ("ty so much", "thanks"), ("Ash, cheers", "thanks"),
                       ("ok bye", "bye"), ("heading off now cya", "bye"), ("catch you later", "bye"), ("leaving now peace", "bye")]:
        v = c.classify(said)
        check(f"{said!r} -> {want}", v.label == want, f"{v.label} {v.score:.2f} {v.scores}")


def h2_only_when_clear():
    print("\n=== H2. Only when clear ===")
    c = make()
    for said in ("what is the weather like", "play something nice", "who is Omar", "open my notes", "how tall is everest", "banana"):
        v = c.classify(said)
        check(f"{said!r} is not labelled", v.label is None, f"{v.label} {v.score:.2f}")
    check("a verdict is falsy when there is no label", not c.classify("banana"))
    tight = Chooser("t", sf.embed, {"a": ["thanks"], "b": ["thank"], "none": ["zzz"]}, accept=0.5, margin=0.2)
    check("two labels that are equally near: no answer (a close call)", tight.classify("thanks").label is None)


def h3_learning():
    print("\n=== H3. Corrections ===")
    p = os.path.join(tempfile.mkdtemp(), "c.json")
    c = make(p)
    v = c.classify("bedtime for me")
    check("before: not known", v.label is None, f"{v.label} {v.score:.2f}")
    check("learn() takes it", c.learn("bedtime for me", "bye"))
    check("after: it is a goodbye", c.classify("bedtime for me").label == "bye")
    check("and a paraphrase of what it learned", c.classify("bedtime now sleep").label == "bye", str(c.classify("bedtime now sleep").scores))
    again = make(p)
    check("it is kept across a restart", again.classify("bedtime for me").label == "bye")
    check("the same sentence twice is not stored twice", not c.learn("Bedtime for me!", "bye"))
    c.learn("bedtime for me", "thanks")
    check("told it was something else, it moves", "bedtime for me" not in [clean(x) for x in c.learned.get("bye", [])]
          and c.classify("bedtime for me").label == "thanks")
    for i in range(300):
        c.learn(f"zq{i} wxy", "thanks")
    check("what it keeps is bounded", len(c.learned["thanks"]) <= c.cap)
    check("too long or empty is refused", not c.learn("", "bye") and not c.learn("x " * 100, "bye"))


def h4_never_raises():
    print("\n=== H4. Never raises ===")
    none = Chooser("n", None, SEEDS)
    check("no encoder: says nothing", not none.ready and none.classify("thanks").label is None)
    boom = Chooser("b", lambda texts: 1 / 0, SEEDS)
    check("a failing encoder: says nothing", boom.classify("thanks").label is None)
    odd = Chooser("o", lambda texts: [[1.0]], SEEDS)
    check("an encoder that returns the wrong shape: says nothing", odd.classify("thanks").label is None)
    c = make()
    check("empty and odd sentences", c.classify("").label is None and c.classify("!!!").label is None and c.classify("a" * 5000).label is None)
    check("status", c.status()["asked"] >= 3)


def h5_runtime():
    print("\n=== H5. In the runtime ===")
    import tst_effects as te
    d = tempfile.mkdtemp()
    rt = te.rig(chooser_embed=sf.embed, chooser_dir=d)
    a = rt.chooser("small", SEEDS, accept=0.5)
    check("made once per name", a is not None and rt.chooser("small", SEEDS) is a)
    a.learn("bedtime for me", "bye")
    check("what it learns is kept in the folder", os.path.exists(os.path.join(d, "small.json")))
    check("a bare runtime has none, so the rules stand", te.rig().chooser("small", SEEDS) is None)


if __name__ == "__main__":
    for fn in (h1_meaning, h2_only_when_clear, h3_learning, h4_never_raises, h5_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
