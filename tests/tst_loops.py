"""
tst_loops.py -- a recogniser or a model stuck repeating itself is not taken as speech or as an answer.

Live: whisper wrote "I'm not sure if I'm going to work with Ash, I'm not sure if ... Ash, ..." (three times) from noise, it was
answered as a request, and the small model then said the same sentence three times back.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_loops.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.senses.mic import is_loop, trusted_text  # noqa: E402
from src.daemon.repeat import RepeatGuard  # noqa: E402

check = tw.check


def l1_mic():
    print("\n=== L1. The microphone ===")
    live = "I'm not sure if I'm going to work with Ash, I'm not sure if I'm going to work with Ash, I'm not sure if I'm going to work with Ash."
    check("the live loop is one", is_loop(live))
    check("two repeats are not (people do say things twice)", not is_loop("I'm not sure if I'm going to work with Ash, I'm not sure if I'm going to work with Ash."))
    check("a short repeat is not", not is_loop("no no no no"))
    check("ordinary speech is not", not is_loop("Can you check my calendar for tomorrow and then remind me about the dentist at ten?"))
    check("a request with a repeated word is not", not is_loop("please please please remind me to call mom at five o'clock today"))
    segs = [NS(text=live, avg_logprob=-0.4, no_speech_prob=0.0)]
    check("a transcript that is a loop is dropped", trusted_text(segs) == "")
    segs = [NS(text="What time is it?", avg_logprob=-0.3, no_speech_prob=0.0)]
    check("a normal one is kept", trusted_text(segs) == "What time is it?")


def l2_answer():
    print("\n=== L2. An answer saying one sentence three times ===")
    g = RepeatGuard()
    s = "I'm not sure if I'm going to work with Ash. I'm not sure if I'm going to work with Ash. I'm not sure if I'm going to work with Ash. How about you feel?"
    out = g.check(s)
    check("said once", out.count("going to work with Ash") == 1 and "How about you feel?" in out, out)
    g2 = RepeatGuard()
    check("a normal answer is untouched", g2.check("It is six. The weather is fine.") == "It is six. The weather is fine.")


if __name__ == "__main__":
    l1_mic()
    l2_answer()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
