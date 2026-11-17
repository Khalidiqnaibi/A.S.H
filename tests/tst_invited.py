"""
tst_invited.py -- "What do you have in store for me for a free moment? This is a free moment."

The user opens the moment themselves. What ASH keeps for one (a question, a remark) is offered at once, though a conversation is
open and the gap rules would say wait. With nothing saved it says so; it never sends this to the model.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_invited.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import tst_freemoment as tf  # noqa: E402

check = tw.check
ASK = "What do you have in store for me for a free moment? This is a free moment."


def i1_something():
    print("\n=== I1. Something saved: it is offered now ===")
    h = tf.rt_rig()
    h.say("hello there")
    h.step(2, 0.0)
    n = len(h.ash.calls)
    h.say(ASK)
    h.step(1, 3.0)
    said = h.said()
    check("it brings something up though a conversation is open", len(said) >= 1 and "store for me" not in said[-1], str(said))
    check("the model was not asked", len(h.ash.calls) == n, str(h.ash.calls))


def i2_question():
    print("\n=== I2. An open question comes first ===")
    h = tf.rt_rig(enabled=False)
    h.say("I told Omar something about work")
    h.step(2, 0.0)
    h.say(ASK)
    h.step(1, 3.0)
    check("the saved question is asked", "Who is Omar to you?" in h.said(), str(h.said()))


def i3_nothing():
    print("\n=== I3. Nothing saved: it asks something ===")
    h = tf.rt_rig(disabled=["mood_check", "gap_closure", "system_assessment", "joke"])
    n = len(h.ash.calls)
    h.say(ASK)
    said = h.said()
    check("it asks the person something instead of 'nothing saved up' (that was the opposite of the invitation), and not the model",
          said and said[-1].startswith("Okay, free moment. ") and said[-1].endswith("?") and "Nothing saved up" not in said[-1] and len(h.ash.calls) == n,
          str(said))
    h.say(ASK)
    again = h.said()
    check("asked again, it asks a different thing", len(again) >= 2 and again[-1] != again[-2], str(again[-2:]))


def i5_by_meaning():
    print("\n=== I5. Worded unlike the patterns ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import semantic_fake as sf
    from src.daemon.chooser import Chooser
    from src.daemon.freemoment import INVITE_SEEDS
    ch = Chooser("invite", sf.embed, INVITE_SEEDS, accept=0.62, margin=0.1)
    for said in ("now is a good time ask away", "free right now what did you want to ask me"):
        check(f"{said!r} is an invitation by meaning", ch.classify(said).label == "invite", str(ch.classify(said).scores))
    for said in ("i have a free moment tomorrow at ten for the dentist", "free wifi is available here", "i'm not free until five", "what time is it"):
        check(f"{said!r} is not", ch.classify(said).label is None, str(ch.classify(said).scores))
    h = tf.rt_rig(disabled=["mood_check", "gap_closure", "system_assessment", "joke"])
    h.rt.cfg.chooser_embed = sf.embed
    n = len(h.ash.calls)
    h.say("now is a good time ask away")
    said = h.said()
    check("in the runtime it opens the moment, with no model", said and said[-1].startswith("Okay, free moment. ") and len(h.ash.calls) == n, str(said[-1:]))
    off = tf.rt_rig(disabled=["mood_check", "gap_closure", "system_assessment", "joke"])
    n = len(off.ash.calls)
    off.say("now is a good time ask away")
    check("without an encoder the patterns alone decide: it is left to the model", len(off.ash.calls) > n or not any("free moment" in s for s in off.said()))


def i4_other_lines():
    print("\n=== I4. Other lines are left alone ===")
    h = tf.rt_rig()
    h.say("I have a free moment tomorrow at ten for the dentist")
    check("a plain statement about a free moment is not a request", not any("Nothing saved" in s for s in h.said()), str(h.said()))


if __name__ == "__main__":
    i1_something()
    i2_question()
    i3_nothing()
    i4_other_lines()
    i5_by_meaning()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
