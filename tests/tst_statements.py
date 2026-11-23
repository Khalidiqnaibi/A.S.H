"""
tst_statements.py -- "I'm just building my custom hardware chip.": heard, acknowledged in a line, and kept -- with no model.

  S1  what is a statement (and what is left to the feelings, the requests, the questions and the model)
  S2  in the runtime: no model call, the line is written to memory, the rest still goes where it went

Run:  PYTHONIOENCODING=utf-8 python tests/tst_statements.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.statements import Statements  # noqa: E402

check = tw.check


def s1_what():
    print("\n=== S1. What counts ===")
    s = Statements()
    for text, obj in [("I'm just building my custom hardware chip.", "custom hardware chip"),
                      ("Ash, I've bought a 3D printer and I worked on it for a little bit. It's pretty good actually, working very well", "3D printer"),
                      ("I finished the report this morning", "report"), ("we're setting up a new server tonight", "new server"),
                      ("I started a new project today", "new project")]:
        got = s.hear(text)
        check(f"{text[:48]!r} is heard, and the thing is said back", got is not None and obj in got, str(got))
    for text in ("I'm feeling a bit tired today", "I'm so bored", "I need you to open chrome", "I'm building a chip, what do you think?",
                 "I'm not sure what you mean by that", "that's not what I had in mind", "I love pizza", "can you help me build a chip",
                 "I think you are wrong", "what are you building", "remind me that I bought a printer", "yes I did that",
                 "I want you to look at my code", "Maybe in the future I will hook you up for the printer"):
        check(f"{text[:48]!r} is left alone", s.hear(text) is None)
    check("not the same line twice in a row", len({Statements().hear("I'm building a chip") for _ in range(1)}) == 1)


def s2_runtime():
    print("\n=== S2. In the runtime ===")
    import tst_modellast as tm

    class Mem:
        def __init__(self):
            self.turns = []

        def commit_turn(self, user_text, ash_text, user, name, importance=0.5, **k):
            self.turns.append((user_text, ash_text, importance))
    rt = tm.rig()
    rt.brain.memory = Mem()
    used, said = rt.say("I'm just building my custom hardware chip ash.")
    check("no model call, a line back", not used and said and "custom hardware chip" in said[-1], f"{used} {said}")
    check("kept as what was said", rt.brain.memory.turns and "custom hardware chip" in rt.brain.memory.turns[0][0], str(rt.brain.memory.turns))
    for text in ("I'm feeling a bit tired today", "what do you think about building a chip", "Draw me a diagram of a chip"):
        rt2 = tm.rig()
        used, said = rt2.say(text)
        check(f"{text!r} still goes where it went (the model)", used, str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    s1_what()
    s2_runtime()
