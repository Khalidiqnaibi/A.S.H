"""
tst_usermood.py -- questions about the PERSON's mood are about them, and a dict of readings is never read out.

Live (18 Nov): "What's my emotional status?" -> ASH's own ("Pretty happy -- you've thanked me 3 times today"); "What's the user's
emotional state?" -> a failed Wikipedia lookup; "What is the user feeling right now?" -> a timeout; and, when the model gave nothing,
"sentiment: POSITIVE, confidence: 0.9911842942237854" was spoken.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_usermood.py
"""

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import feelings  # noqa: E402

check = tw.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def u1_pattern():
    print("\n=== U1. Whose mood ===")
    for t in ["What's my emotional status?", "What's the user's emotional state?", "What is the user feeling right now?", "How am I feeling, Ash?",
              "What do you think I'm feeling?", "how do I feel", "what's my mood"]:
        check(f"the person's: {t!r}", feelings.wants_user_mood(t))
    for t in ["What's your emotional state?", "How are you feeling, Ash?", "how are you", "what's your mood", "what is the weather"]:
        check(f"not the person's: {t!r}", not feelings.wants_user_mood(t))


def u2_say():
    print("\n=== U2. What it says ===")
    check("nothing known: it says so and asks", "I can't tell" in feelings.user_mood_say(None) and feelings.user_mood_say(None).endswith("How are you?"))
    check("upbeat, said as a reading of the words", "upbeat" in feelings.user_mood_say(0.6) and "your words" in feelings.user_mood_say(0.6))
    check("low", "low" in feelings.user_mood_say(-0.6))
    check("level", "level" in feelings.user_mood_say(0.0))


def u3_runtime():
    print("\n=== U3. On the runtime ===")
    import tst_effects as te
    rt = te.rig(wonder=True, moments=True)
    n = len(rt.ash.calls)
    rt.say("What's my emotional status?", source="console")
    check("not ASH's mood, and not the model", "your words" in rt.out[-1] or "I can't tell" in rt.out[-1], rt.out[-1])
    check("...the model was not asked", len(rt.ash.calls) == n)
    rt.moments.set_mood(-0.7, rt.clock())
    rt.say("How am I feeling, Ash?", source="console")
    check("with a reading it says what it is", "low" in rt.out[-1], rt.out[-1])
    rt.say("What's your emotional state?", source="console")
    check("its own mood is still its own", "your words" not in rt.out[-1] and "I can't tell" not in rt.out[-1], rt.out[-1])


def u4_raw():
    print("\n=== U4. Readings are not spoken ===")
    src = open(os.path.join(ROOT, "src", "brain", "brain.py"), encoding="utf-8").read()
    check("the empty-narration fallback says raw facts only where the tool has words for them",
          re.search(r"readable = isinstance\(tool_out, dict\)", src) is not None and "nothing readable to say" in src)
    from src.brain.brain import _fmt_tool_output
    check("the fast path's template is unchanged", _fmt_tool_output({"ok": True, "sentiment": "POSITIVE", "confidence": 0.99}) == "sentiment: POSITIVE, confidence: 0.99")


if __name__ == "__main__":
    u1_pattern()
    u2_say()
    u3_runtime()
    u4_raw()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
