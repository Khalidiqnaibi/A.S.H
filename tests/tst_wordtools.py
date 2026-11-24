"""
tst_wordtools.py -- "spell Segway": a letter at a time, by rule.

It went to the model: 28 seconds, and the answer was 'To spell Segway, it would be spelled as "Segway."'.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_wordtools.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.wordtools import WordTools, letters  # noqa: E402

check = tw.check


def w1_spell():
    print("\n=== W1. Spelling ===")
    w = WordTools()
    for t, want in [("spell Segway", "Segway: S, E, G, W, A, Y."), ("How do you spell segue?", "segue: S, E, G, U, E."),
                    ("how is Khalid spelled", "Khalid: K, H, A, L, I, D."), ("can you spell necessary please", "necessary: N, E, C, E, S, S, A, R, Y."),
                    ("what's the spelling of rhythm", "rhythm: R, H, Y, T, H, M."), ('spell "Opera"', "Opera: O, P, E, R, A."),
                    ("spell New York", "New York: N, E, W, space, Y, O, R, K.")]:
        check(f"{t!r} -> {want}", w.spell(t) == want, str(w.spell(t)))
    check("'spell it again' is the last word again", w.spell("spell it again") == "New York: N, E, W, space, Y, O, R, K.")
    check("'spell that out loud' too", w.spell("spell that out loud").startswith("New York"))
    check("symbols are named", letters("a-b'c.d") == "A, hyphen, B, apostrophe, C, dot, D")
    fresh = WordTools()
    check("'spell it again' with nothing spelled says so", "haven't spelled anything yet" in fresh.spell("spell it again"))
    for t in ("what is a spell", "spell", "cast a spell on me", "how do you feel", "spelling bee", "", None, "spell " + "x" * 90):
        check(f"{str(t)[:30]!r} is left alone", WordTools().spell(t) is None)


def w2_runtime():
    print("\n=== W2. In the runtime ===")
    import tst_modellast as tm
    rt = tm.rig()
    used, said = rt.say("Spell Segway")
    check("by rule, no model", not used and said and said[-1] == "Segway: S, E, G, W, A, Y.", f"{used} {said}")
    used, said = rt.say("spell it again")
    check("'spell it again' works across lines", not used and said[-1] == "Segway: S, E, G, W, A, Y.", f"{used} {said}")
    used, said = rt.say("what is a spell checker")
    check("a question about something else still goes where it went", used or "S, E" not in said[-1], str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    w1_spell()
    w2_runtime()
