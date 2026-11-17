"""
tst_courtesy_real.py -- the courtesy chooser against the REAL sentence encoder (the one ASH classifies with).

The other tests use a stand-in encoder to test the mechanics. This one checks the thing that matters: that phrasings the rules never
listed are recognised by meaning, and that ordinary sentences with courtesy words in them are not. If the encoder cannot be loaded (no
model, no network on first use) it says so and passes: it cannot judge.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_courtesy_real.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

# phrasings the rules do not list
POS = {
    "thanks": ["thanks so much", "much obliged", "i really appreciate it", "that's great, thanks a bunch", "many thanks", "thx a lot ash", "cheers mate"],
    "bye": ["catch you tomorrow", "i'm out", "gotta run, bye", "see ya", "talk soon", "heading off now", "ok i'm gone"],
    "night": ["i'm off to bed", "good night ash", "sleep tight", "nighty night ash"],
    "brb": ["back in five", "hold on a sec", "i'll be right back", "give me one minute"],
    "ack": ["okay got it", "alright cool", "oh i see", "right, makes sense", "sounds good to me", "all good"],
}
# sentences that contain courtesy words but are something else, and plain requests
NEG = ["what is my cpu usage", "tell me about the roman empire", "can you play something upbeat", "i want to go home",
       "thanks but that's not what i asked", "later today remind me to call dad", "what do you remember about me", "bye the way what time is it",
       "good night's sleep tips", "remember to buy milk", "how are you feeling", "yes please", "no thanks", "ok", "sure", "who is lieth",
       "my name is khalid", "goodbye cruel world is a song right", "i need a minute to think about what to ask", "what's the plan for tonight",
       "whats todays trivia", "what is the capital of france", "open spotify", "i love this song"]


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
    from src.daemon.courtesy import SEEDS, Courtesy
    c = Courtesy(chooser=Chooser("courtesy", emb, SEEDS, accept=0.6, margin=0.12))
    print("\n=== Real encoder: what the rules never listed ===")
    total = hit = 0
    for want, items in POS.items():
        for said in items:
            total += 1
            got = c.kind(said)
            hit += got == want
            print(f"  {'ok ' if got == want else 'miss'} {said!r} -> {got}")
    check(f"most unlisted phrasings are understood ({hit}/{total})", hit >= int(0.75 * total), f"{hit}/{total}")
    wrong = [(s, c.kind(s)) for items in POS.values() for s in items if c.kind(s) is not None and c.kind(s) not in POS]
    print("\n=== Real encoder: what must stay a sentence ===")
    false = [(s, c.kind(s)) for s in NEG if c.kind(s) is not None]
    for s, k in false:
        print(f"  FALSE {s!r} -> {k}")
    check("nothing that is a question, a request or a sentence is taken for a courtesy", not false, str(false))
    wrong_kind = [(s, w, c.kind(s)) for w, items in POS.items() for s in items if c.kind(s) not in (None, w)]
    check("and when it answers it never answers with the wrong kind", not wrong_kind, str(wrong_kind))


if __name__ == "__main__":
    main()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
