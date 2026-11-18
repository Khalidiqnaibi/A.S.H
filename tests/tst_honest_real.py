"""
tst_honest_real.py -- "what can you do?" by the REAL sentence encoder.

Today "What can you do then?" matched no pattern (one trailing word broke it), went to the model, and was answered with a made-up "I know you
asked about Khalid, but I can't pull his contacts...". The pattern now allows the usual trailing words; this checks the chooser behind it on
wordings the pattern does not list, and on sentences that look similar but ask something else. If the encoder cannot be loaded it says so and
passes.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_honest_real.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

POS = ["so what are you actually capable of", "what's in your toolbox", "how are you useful to me", "what jobs can you take on",
       "what kind of things can you handle", "tell me about your abilities", "what do you bring to the table", "what could i get you to do",
       "what are you able to help with"]
NEG = ["can you help me with this email", "what can i do about the noise", "what do you think of my plan", "what do you remember about me",
       "who built you", "how do you feel today", "what time is it", "what did you do yesterday", "what does this button do", "what is my cpu usage",
       "what can you see", "play something", "what do you do when i'm away", "what's the weather going to do", "can you do it again",
       "how does a transformer work", "tell me a joke", "what should i do tonight"]


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
    from src.daemon.honest import SEEDS
    c = Chooser("h", emb, SEEDS, accept=0.62, margin=0.1)
    print("\n=== Capability questions the pattern does not list ===")
    hit = 0
    for s in POS:
        v = c.classify(s)
        hit += v.label == "capabilities"
        print(f"  {'ok ' if v.label == 'capabilities' else 'miss'} {s!r} -> {v.label} ({v.score:.2f}, ahead {v.margin:.2f})")
    check(f"most are understood ({hit}/{len(POS)})", hit >= int(0.55 * len(POS)), f"{hit}/{len(POS)}")
    print("\n=== What must be left alone ===")
    false = [(s, c.classify(s).label) for s in NEG if c.classify(s).label is not None]
    for s, k in false:
        print(f"  FALSE {s!r} -> {k}")
    check("help requests, questions about the person, the weather and everything else are not capability questions", not false, str(false))


if __name__ == "__main__":
    main()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
