"""
tst_moodanswer.py -- a question or greeting is not the answer to "how are you holding up today?".

Live: "Who is Linda" was answered "Ah, that sounds rough. I'll go easy."; "hello Ash" was answered "Good. Hold onto that."

Run:  PYTHONIOENCODING=utf-8 python tests/tst_moodanswer.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import tst_freemoment as tf  # noqa: E402
from src.daemon.freemoment import FreeMoments  # noqa: E402

check = tw.check


def m1_rule():
    print("\n=== M1. What is not an answer ===")
    f = FreeMoments.is_not_an_answer
    for t in ["Who is Linda", "hello Ash", "What time is it?", "what do you think of rainy days", "remind me to call mom",
              "Can you check my calendar for tomorrow?", "Tell me about Samsung Notes", "good morning"]:
        check(f"not an answer: {t!r}", f(t))
    for t in ["pretty tired honestly", "fine thanks", "not great", "I'm good", "bit stressed about the exam", "how about you?",
              "how are you?", "tired, you?", "yeah"]:
        check(f"an answer (or turning it round): {t!r}", not f(t))


def m2_runtime():
    print("\n=== M2. In the runtime ===")
    h = tf.rt_rig()
    h.free()
    first = h.said()[0]
    check("the mood question was asked", first.startswith("How are you"), first)
    h.step(5, 5.0)
    n = len(h.ash.calls)
    h.say("Who is Linda", source="microphone", addressed=True)
    check("a question after it goes on to be answered, not read as a mood", len(h.ash.calls) == n + 1 and h.rt.moments.mood(h.clock[0]) is None,
          str((h.ash.calls, h.said())))
    h.say("pretty tired honestly", source="microphone", addressed=False)
    check("the answer is still taken afterwards", "rough" in h.said()[-1], str(h.said()))


def m3_statement():
    print("\n=== M3. A statement about something else is not a mood ===")
    from src.daemon.freemoment import MoodCheck
    fm = FreeMoments.__new__(FreeMoments)
    fm.read = lambda t: 0.9                       # the sentiment model finds nearly everything 'positive'
    fm.set_mood = lambda v, now: None
    mc = MoodCheck(fm)
    check("'My dad's name is Afif' is not an answer", mc.hear("My dad's name is Afif", 0.0) is None)
    check("'I have an exam on Sunday at ten' is not", mc.hear("I have an exam on Sunday at ten", 0.0) is None)
    check("'pretty tired honestly' is", mc.hear("pretty tired honestly", 0.0) is not None)
    check("'fine thanks' is", mc.hear("fine thanks", 0.0) is not None)
    check("'good' is", mc.hear("good", 0.0) is not None)


if __name__ == "__main__":
    m1_rule()
    m2_runtime()
    m3_statement()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
