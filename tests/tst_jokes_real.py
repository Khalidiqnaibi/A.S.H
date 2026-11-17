"""
tst_jokes_real.py -- "you're repeating yourself", by the REAL sentence encoder.

Checked on wordings the patterns do not list, and on sentences about jokes that are not a complaint (asking for another, laughing, asking
about one). If the encoder cannot be loaded it says so and passes.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_jokes_real.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

POS = ["bro that's the same one again", "ugh not this joke again", "i swear you told me this yesterday", "you're repeating yourself",
       "didn't you just say that", "come on, new material please", "this joke is getting old", "you've got to stop with this one"]
NEG = ["tell me a different joke", "that was funny", "i like that joke", "what joke was that", "haha good one", "make me laugh",
       "who wrote that joke", "tell me another one", "i told that joke to my sister", "what time is it", "thanks", "sing me a song",
       "i've heard of that song", "you already know my name", "that's the third alarm today", "again please", "say it again, i didn't catch it"]


def encoder():
    try:
        logging.disable(logging.CRITICAL)
        from tools.shared_model import load
        m = load(os.environ.get("ASH_EMBED_MODEL", "all-MiniLM-L6-v2"))
        return lambda texts: m.encode(list(texts), show_progress_bar=False)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (the sentence encoder is not available here: {type(e).__name__}; nothing to judge)")
        return None


def main():
    emb = encoder()
    if emb is None:
        check("no encoder here: skipped", True)
        return
    from src.daemon.chooser import Chooser
    from src.daemon.fun import SEEDS
    c = Chooser("j", emb, SEEDS, accept=0.6, margin=0.1)
    print("\n=== Complaints the patterns do not list ===")
    hit = 0
    for s in POS:
        v = c.classify(s)
        hit += v.label == "repeat"
        print(f"  {'ok ' if v.label == 'repeat' else 'miss'} {s!r} -> {v.label} ({v.score:.2f}, ahead {v.margin:.2f})")
    check(f"most are understood ({hit}/{len(POS)})", hit >= int(0.6 * len(POS)), f"{hit}/{len(POS)}")
    print("\n=== What must be left alone ===")
    false = [(s, c.classify(s).label) for s in NEG if c.classify(s).label is not None]
    for s, k in false:
        print(f"  FALSE {s!r} -> {k}")
    check("asking for another, laughing, and everything else is not a complaint", not false, str(false))


if __name__ == "__main__":
    main()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
