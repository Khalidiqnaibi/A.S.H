"""tst_probe.py -- natural rewordings of what ASH already does must never reach the model. Each family is a few wordings that are NOT the ones the rule was written from;
a model that counts its calls fails the line it is handed. Lines that still reach it are in GAPS (known, taken one at a time); when one is fixed this says so and it moves up.
Rules only (no encoder), no network, no real devices. Run:  PYTHONIOENCODING=utf-8 python tests/tst_probe.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402

check = tw.check

FAMILIES = {
    "self: model": ["which model are you using", "what llm is answering me", "what are you thinking with", "tell me which model you run on"],
    "self: see": ["can you see my screen", "are you looking at me", "what are you able to see right now"],
    "self: hear": ["are you listening to me", "can you hear me right now", "is your mic on"],
    "self: state": ["how are you set up", "what's turned on right now"],
    "memory: check": ["check your memory", "is your memory clean", "audit my memory", "is there junk in your memory"],
    "memory: clean": ["tidy up your memory", "clean your memory", "clean up my memory"],
    "learn: status": ["what have you learned", "what are you learning", "what did you learn from google"],
    "learn: method": ["how do you learn from google", "how do you learn things"],
    "learn: do": ["learn how to make a thumbnail in photoshop", "go and learn how to rotate a pdf", "teach yourself how to use git"],
}
# Known gaps: still reach the model. {line: why}
GAPS = {}


def main():
    n = 0
    for fam, lines in FAMILIES.items():
        print(f"\n=== {fam} ===")
        for text in lines:
            rt = tm.rig()
            used, said = rt.say(text)
            if text in GAPS:
                check(f"(known gap) {text!r}", used, "answered now: move it out of GAPS")
            else:
                check(f"{text!r}", not used and bool(said), f"model={used} said={said[-1:]}")
            n += used
    print(f"\n  {n} lines reach the model")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
