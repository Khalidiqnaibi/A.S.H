"""
tst_feelings_real.py -- "how are you feeling" versus "what's your status", by the REAL sentence encoder.

Today "whats your emotional status" was answered with the CPU and battery. One rule matched the word "status"; the one for feelings did not
know "emotional". Checked here on phrasings neither rule lists, and on questions about the USER's feelings or about other things, which must be
left alone. If the encoder cannot be loaded it says so and passes.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_feelings_real.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

FEELINGS = ["whats your emotional status", "what kind of mood are you in right now", "everything okay with you", "how have you been feeling lately",
            "you holding up alright", "how's your emotional health", "are you in good spirits", "how's your heart today", "what's your current mood"]
STATUS = ["how heavy is the cpu load right now", "tell me the memory situation", "is the battery low", "how much ram is left", "give me the system report",
          "how's the machine doing", "how full is the disk"]
NEITHER = ["what time is it", "how am i feeling", "do you think i'm stressed", "how do i feel about this", "how are you going to fix this", "how's the weather",
           "how's my schedule", "who is omar", "play something", "thanks", "tell me a joke", "what do you remember", "what's the plan for today",
           "how long will it take", "what's my emotional state", "open the browser", "i'm sad", "i feel tired today"]


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
    from src.daemon.feelings import SEEDS
    c = Chooser("f", emb, SEEDS, accept=0.6, margin=0.1)
    for label, items in (("feelings", FEELINGS), ("status", STATUS)):
        print(f"\n=== {label}: phrasings neither rule lists ===")
        hit = 0
        for s in items:
            v = c.classify(s)
            hit += v.label == label
            print(f"  {'ok ' if v.label == label else 'miss'} {s!r} -> {v.label} ({v.score:.2f}, ahead {v.margin:.2f})")
        check(f"most are understood as {label} ({hit}/{len(items)})", hit >= int(0.7 * len(items)), f"{hit}/{len(items)}")
        other = "status" if label == "feelings" else "feelings"
        wrong = [s for s in items if c.classify(s).label == other]
        check(f"and none is taken for {other}", not wrong, str(wrong))
    print("\n=== neither: must be left alone ===")
    false = [(s, c.classify(s).label) for s in NEITHER if c.classify(s).label is not None]
    for s, k in false:
        print(f"  FALSE {s!r} -> {k}")
    check("questions about the user, or about anything else, are not taken for either", not false, str(false))


if __name__ == "__main__":
    main()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
