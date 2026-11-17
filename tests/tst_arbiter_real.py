"""
tst_arbiter_real.py -- the arbiter's choosers against the REAL sentence encoder.

Two decisions that were regexes: which PLACE a correction names ("i mean from your memory"), and whether a sentence is a "who is X"
question at all. Checked here on phrasings the rules do not list, and on sentences that must be left alone. If the encoder cannot be
loaded it says so and passes: it cannot judge.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_arbiter_real.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

SOURCES_POS = {
    "memory": ["no i meant what we talked about before", "use what you remember", "think about our earlier chats", "i told you about this before"],
    "notes": ["use the notes i gave you", "check what i asked you to keep", "it's in my notes"],
    "wikipedia": ["search the internet for it", "i mean the famous one", "look him up online", "the one on wikipedia"],
    "people": ["one of the people i told you about", "my family", "somebody in my contacts"],
    "concepts": ["what ideas do you connect to it", "from what you've learned"],
}
SOURCES_NEG = ["what time is it", "play something", "that's not right", "tell me more about it", "go on", "yes", "no", "thanks", "never mind",
               "who is omar", "open the notes app", "how is the weather", "remind me tomorrow", "can you repeat that", "what do you mean"]
WHO_POS = ["who exactly is Linda", "what do we know about Khalid", "anything you can tell me about Sara", "so who's this Omar guy",
           "I forgot who Lieth is", "explain who Ada Lovelace was", "tell me something about Omar", "is Omar someone you know"]
WHO_SUBJECTS = {"who exactly is Linda": "linda", "what do we know about Khalid": "khalid", "anything you can tell me about Sara": "sara",
                "so who's this Omar guy": "omar", "I forgot who Lieth is": "lieth", "explain who Ada Lovelace was": "ada lovelace",
                "tell me something about Omar": "omar", "is Omar someone you know": "omar"}
WHO_NEG = ["what time is it", "play something upbeat", "how are you", "set a timer", "what is my cpu usage", "write an email to my boss",
           "open spotify", "what should i cook", "who are you", "what can you do", "thanks a lot", "see you later", "how do i fix this error",
           "turn the volume down", "remember to buy milk", "what is the capital of france"]


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
    from src.daemon.arbiter import SOURCE_SEEDS, WHO_SEEDS, loose_subject
    from src.daemon.chooser import Chooser
    src = Chooser("s", emb, SOURCE_SEEDS, accept=0.55, margin=0.1)
    who = Chooser("w", emb, WHO_SEEDS, accept=0.6, margin=0.1)

    print("\n=== Which place a sentence names ===")
    hit = total = 0
    for want, items in SOURCES_POS.items():
        for s in items:
            total += 1
            v = src.classify(s)
            hit += v.label == want
            print(f"  {'ok ' if v.label == want else 'miss'} {s!r} -> {v.label} ({v.score:.2f}, ahead {v.margin:.2f})")
    check(f"most unlisted ways of naming a place are understood ({hit}/{total})", hit >= int(0.7 * total), f"{hit}/{total}")
    false = [(s, src.classify(s).label) for s in SOURCES_NEG if src.classify(s).label is not None]
    check("ordinary sentences are not taken for naming a place", not false, str(false))
    wrong = [(s, w, src.classify(s).label) for w, items in SOURCES_POS.items() for s in items if src.classify(s).label not in (None, w)]
    check("and when it answers it names the right place", not wrong, str(wrong))

    print("\n=== Is it a who-question ===")
    hit = 0
    for s in WHO_POS:
        v = who.classify(s)
        sub = loose_subject(s)
        ok = v.label == "who_question" and sub is not None and sub.lower() == WHO_SUBJECTS[s]
        hit += ok
        print(f"  {'ok ' if ok else 'miss'} {s!r} -> {v.label} ({v.score:.2f}) subject {sub!r}")
    check(f"most unlisted who-questions are understood, with their subject ({hit}/{len(WHO_POS)})", hit >= int(0.75 * len(WHO_POS)), f"{hit}")
    false = [(s, who.classify(s).label) for s in WHO_NEG if who.classify(s).label is not None]
    check("requests and other questions are not taken for one", not false, str(false))


if __name__ == "__main__":
    main()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
