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
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
