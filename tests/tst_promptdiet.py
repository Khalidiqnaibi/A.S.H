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


def d4_context():
    print("\n=== D4. Window flicking is not context ===")
    from src.daemon.journal import Journal
    from src.senses.base import Modality, SensorEvent
    j = Journal(tempfile.mkdtemp())
    now = time.time()
    for i, (src, text) in enumerate([("active_window", "switched to Stremio (was on Claude for 3s)"),
                                     ("active_window", "switched to Claude (was on Stremio for 27s)"),
                                     ("active_window", "switched to Stremio (was on Claude for 4s)"),
                                     ("console", "write me a haiku about rain")]):
        j.append(SensorEvent(source=src, modality=Modality.TEXT, text=text, ts=now - 30 + i, salience=0.5))
    j.flush()
    block = j.context_block(seconds=600)
    check("the 3s and 4s flicks are gone", "for 3s" not in block and "for 4s" not in block, block)
    check("a real stretch (27s) and the user's line stay", "for 27s" in block and "haiku" in block, block)


def d5_episodes():
    print("\n=== D5. Episodes that repeat the conversation, or each other ===")
    me = NS(wm=[WorkingSlot("user", "Tell me about Samsung Notes in two sentences", time.time())])
    me.working_block = lambda max_chars=4000: MemoryEngine.working_block(me, max_chars)
    block = ("[RELEVANT EPISODES]" + chr(10) + "- [2026-11-14 19:56] Immortal: explain recursion simply" + chr(10)
             + "- [2026-11-14 19:57] Immortal: explain recursion simply" + chr(10)
             + "- [2026-11-16 19:01] Immortal: Tell me about Samsung Notes in two sentences [mentions: Samsung Notes]" + chr(10)
             + "- [2026-11-10 10:00] Immortal: my dad runs a clinic in Hebron")
    out = MemoryEngine._without_what_is_already_said(me, block)
    check("an exact repeat is said once", out.count("explain recursion simply") == 1, out)
    check("an episode that is a line of the conversation is left out", "Samsung Notes" not in out, out)
    check("a real earlier one stays", "clinic in Hebron" in out, out)
    only = MemoryEngine._without_what_is_already_said(me, "[RELEVANT EPISODES]" + chr(10) + "- [2026-11-16 19:01] Immortal: Tell me about Samsung Notes in two sentences")
    check("nothing left: no empty block", only == "", repr(only))


def d6_context_sources():
    print("\n=== D6. The prompt's activity block skips what the user said ===")
    from src.daemon.journal import Journal
    from src.senses.base import Modality, SensorEvent
    j = Journal(tempfile.mkdtemp())
    now = time.time()
    for i, (src, text) in enumerate([("console", "write me a haiku"), ("microphone", "what time is it"), ("idle", "user went away from the machine")]):
        j.append(SensorEvent(source=src, modality=Modality.TEXT, text=text, ts=now - 20 + i, salience=0.5))
    j.flush()
    block = j.context_block(seconds=600, skip_sources=("console", "microphone"))
    check("what was said is not repeated there, what happened is", "haiku" not in block and "time is it" not in block and "went away" in block, block)
    check("by default it is all there", "haiku" in j.context_block(seconds=600))


if __name__ == "__main__":
    d1_overlap()
    d2_working()
    d3_note()
    d4_context()
    d5_episodes()
    d6_context_sources()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
