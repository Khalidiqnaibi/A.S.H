"""
tst_play_real.py -- "is that a request to play something?" by the REAL sentence encoder.

Today "whats todays trivia" matched no pattern and went to the model, which said it had no access to it. Checked here on phrasings the
patterns do not list, and on sentences that mention games, music or trivia without asking to play one (those must be left alone). If the
encoder cannot be loaded it says so and passes.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_play_real.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

POS = {
    "trivia": ["what's the trivia for today", "do you have a trivia question for me", "hit me with some trivia", "daily trivia please",
               "quiz me on something", "i'd like the trivia now"],
    "riddle": ["got any riddles", "i'd like a riddle", "ask me something tricky like a riddle", "give me a brain teaser riddle"],
    "game": ["wanna play a game", "let's play something together", "can we have a game", "i feel like playing a game"],
}
NEG = ["play some music", "play the next song", "turn the music up", "what is a riddle", "who invented trivia", "tell me a joke", "who won the game last night",
       "the game starts at eight", "i played chess yesterday", "what time is it", "thanks", "open spotify", "i'm bored", "what's the answer to that",
       "play something upbeat", "what should we do tonight", "pause the video", "i love this game", "remind me about the trivia night on friday"]


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
    from src.daemon.play import SEEDS
    c = Chooser("p", emb, SEEDS, accept=0.62, margin=0.1)
    hit = total = 0
    print("\n=== Requests the patterns do not list ===")
    for want, items in POS.items():
        for s in items:
            total += 1
            v = c.classify(s)
            hit += v.label == want
            print(f"  {'ok ' if v.label == want else 'miss'} {s!r} -> {v.label} ({v.score:.2f}, ahead {v.margin:.2f})")
    check(f"most are understood ({hit}/{total})", hit >= int(0.65 * total), f"{hit}/{total}")
    wrong = [(s, w, c.classify(s).label) for w, items in POS.items() for s in items if c.classify(s).label not in (None, w)]
    check("and when it answers it is the right kind of game", not wrong, str(wrong))
    print("\n=== What must be left alone ===")
    false = [(s, c.classify(s).label) for s in NEG if c.classify(s).label is not None]
    for s, k in false:
        print(f"  FALSE {s!r} -> {k}")
    check("music, talk about games, and everything else is not a request to play", not false, str(false))


if __name__ == "__main__":
    main()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
