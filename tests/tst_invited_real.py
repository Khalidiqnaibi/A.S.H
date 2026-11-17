"""
tst_invited_real.py -- "I'm free, bring something up", by the REAL sentence encoder.

The runtime recognised an invitation only when two patterns both matched ("free moment" and "ask me"). Checked here on wordings the patterns
do not list, and on sentences that mention being free or a question without inviting ASH to bring something up. If the encoder cannot be
loaded it says so and passes.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_invited_real.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

POS = ["i'm at a loose end, got any questions for me", "nothing on right now, what's on your mind", "i can chat for a bit, go ahead",
       "take your turn, ask me something", "if you've got something to ask, now's your chance", "i'm all ears, what did you want to know",
       "i've got a few minutes, bring something up", "talk to me, i'm bored and free"]
NEG = ["i have a free moment tomorrow at ten for the dentist", "i'm free on friday", "what is a free moment", "i'm busy right now", "don't bother me",
       "free wifi is available at the cafe", "i'm not free until five", "i can ask you a question", "do you have time for a question",
       "what time is it", "tell me a joke", "thanks", "who is omar", "remind me when i'm free", "i had a free period at school",
       "can i ask you something", "play some music", "i'm tired", "what do you want for dinner"]


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
    from src.daemon.freemoment import INVITE_SEEDS
    c = Chooser("i", emb, INVITE_SEEDS, accept=0.62, margin=0.1)
    print("\n=== Invitations the patterns do not list ===")
    hit = 0
    for s in POS:
        v = c.classify(s)
        hit += v.label == "invite"
        print(f"  {'ok ' if v.label == 'invite' else 'miss'} {s!r} -> {v.label} ({v.score:.2f}, ahead {v.margin:.2f})")
    check(f"most are understood ({hit}/{len(POS)})", hit >= int(0.5 * len(POS)), f"{hit}/{len(POS)}")
    print("\n=== What must be left alone ===")
    false = [(s, c.classify(s).label) for s in NEG if c.classify(s).label is not None]
    for s, k in false:
        print(f"  FALSE {s!r} -> {k}")
    check("being free later, asking for ASH's time, and everything else is not an invitation", not false, str(false))


if __name__ == "__main__":
    main()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
