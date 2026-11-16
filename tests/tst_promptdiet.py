"""
tst_promptdiet.py -- what goes into the model's prompt, and what no longer does.

Live, with a 3B model on a 4 GB card, an answer took 16-40 s: the prompt was ~2,400 tokens (prefill is the cost), and held the
retrieved memory three times, four paragraphs of the manual that shared one word with the question, and a note telling the model
to ask a clarifying question about ordinary conversation.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_promptdiet.py
"""

import os
import re
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain.memory_engine import MemoryEngine, WorkingSlot  # noqa: E402
from src.memory.memory_router import MemoryRouter  # noqa: E402

check = tw.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def d1_overlap():
    print("\n=== D1. One shared word is not relevance ===")
    r = MemoryRouter.__new__(MemoryRouter)
    q = r._tokenize("name three fruits that are red")
    manual = ("ASH manual, routines. ASH notices when you open the same apps in the same order on several days and asks once if it should "
              "remember that as a routine; yes saves it, and run my name opens them in order. Say the name of a routine you kept and it runs. Routines can be listed, renamed, forgotten or kept for weekdays only, and each one " "opens apps in a fixed order with a short pause between windows so that nothing is missed while the machine is busy.")
    check("a long manual entry sharing only 'name' is out", not r._enough_overlap(q, manual))
    check("with two words in common it is in", r._enough_overlap(r._tokenize("name my routine for apps"), manual))
    check("a short memory is judged as before", r._enough_overlap(q, "Linda is the user's sister and loves red fruits"))
    check("a short question is judged as before", r._enough_overlap(r._tokenize("what is a routine"), manual))


def d2_working():
    print("\n=== D2. Retrieved memory is not repeated in the recent conversation ===")
    me = NS(wm=[WorkingSlot("user", "hello there", time.time()), WorkingSlot("retrieved", "[core] [CORE MEMORY] lots of text", time.time()),
                WorkingSlot("ash", "hi", time.time())])
    block = MemoryEngine.working_block(me)
    check("the user's and ASH's lines are there", "user: hello there" in block and "ash: hi" in block, block)
    check("what memory returned is not", "CORE MEMORY" not in block and "retrieved" not in block, block)


def d3_note():
    print("\n=== D3. 'Ask one clarifying question' only when actions tie ===")
    src = open(os.path.join(ROOT, "src", "brain", "brain.py"), encoding="utf-8").read()
    check("the deadlock note needs an intent", re.search(r"elif verdict2\.deadlocked and intent:\s*\n?\s*(?:#[^\n]*\n\s*)*note = \"Multiple interpretations", src) is not None)


if __name__ == "__main__":
    d1_overlap()
    d2_working()
    d3_note()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
