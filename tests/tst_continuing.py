"""tst_continuing.py -- 28 Nov: twelve short lines said in the middle of an exchange ("No, not like this.", "Try multiple of them.", "Click image.", "And that was wrong.") scored
0.07-0.28 and were never answered. In a warm exchange a short line is now at least "probably for ASH"; nothing else changes. Offline.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_continuing.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.senses import address  # noqa: E402

check = tw.check


def warm(seconds_ago):
    c = address.Conversation()
    c.note_ash_spoke(1000.0, asked=False)
    return c, 1000.0 + seconds_ago


def main():
    print("\n=== A1. A warm exchange ===")
    for line in ("No, not like this.", "Try multiple of them.", "Click image.", "And that was wrong.", "Try multiple of them now.", "Do the other one."):
        c, now = warm(12.0)
        v = address.score(line, now, convo=c)
        check(f"{line!r} -> at least probably for ASH ({v.confidence:.2f})", v.confidence >= address.LIKELY_AT and not v.addressed or v.addressed, ", ".join(v.reasons))
    print("\n=== A2. Nothing else is stretched ===")
    c, now = warm(12.0)
    check("a long line (a story told to someone) is not lifted", address.score("so then I went to the shop and she told me the whole thing was cancelled again", now, convo=c).confidence < address.LIKELY_AT)
    check("first person only is not lifted ('I'm exhausted')", address.score("I'm so exhausted today", now, convo=c).confidence < address.LIKELY_AT)
    address.KNOWN_OTHERS.add("linda")
    check("said to someone ASH knows is not lifted", address.score("Linda, pass me that one", now, convo=c).confidence < address.LIKELY_AT)
    address.KNOWN_OTHERS.discard("linda")
    c, now = warm(120.0)
    check("a cold exchange (two minutes on) is not lifted", address.score("Try multiple of them.", now, convo=c).confidence < address.LIKELY_AT)
    c = address.Conversation()
    check("no exchange at all: nothing lifted", address.score("Click image.", 1000.0, convo=c).confidence < address.LIKELY_AT)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
