"""
tst_aboutme.py -- "what about my relationships?" and "what do you know about me?" are answered from what ASH was told.

Live (18 Nov): "What about my relationships?" was asked three times and timed out with the model, though ASH knows Linda (sister) and
Afif (father); "What do you know about me?" read out a list of raw notes, with no name and no people.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_aboutme.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import remember  # noqa: E402

check = tw.check


def a1_pattern():
    print("\n=== A1. What counts as asking ===")
    for t in ["What about my relationships?", "What are my relationships that you remember?", "what about my relationship, Ash?",
              "who are the people in my life", "who is my family", "tell me about my relatives", "list my friends and family"]:
        check(f"asks about the people: {t!r}", remember.wants_relationships(t))
    for t in ["What do you know about me?", "what about my dog", "What about my relationships with food", "who is Linda", "my sister is Linda"]:
        check(f"not that: {t!r}", not remember.wants_relationships(t))


def a2_reply():
    print("\n=== A2. The reply ===")
    r = tw.rig()
    r.kb.store_relation("Linda Iqnaibi", role="sister")
    r.kb.store_relation("Afif", role="father")
    say = remember.relationships(r.kb)
    check("the people and how they are related", "Linda Iqnaibi is your sister" in say and "Afif is your father" in say, say)
    check("nobody on record: it says so and how to tell it", "don't have anyone" in remember.relationships(tw.rig().kb))


def a3_runtime():
    print("\n=== A3. On the runtime ===")
    import tst_effects as te
    rt = te.rig(wonder=True, notes=True, notes_path=os.path.join(tempfile.mkdtemp(), "notes.json"))
    rt.wonder.kb.store_relation("Linda Iqnaibi", role="sister")
    n = len(rt.ash.calls)
    rt.say("What about my relationships?", source="console")
    check("answered by rule, the model not asked", "Linda Iqnaibi is your sister" in rt.out[-1] and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("remember that the plumber comes on Friday", source="console")
    rt.say("What do you know about me?", source="console")
    check("'what do you know about me' has the people AND the notes", "Linda Iqnaibi (sister)" in rt.out[-1] and "plumber" in rt.out[-1], rt.out[-1])
    check("...and it is not the model", len(rt.ash.calls) == n)


if __name__ == "__main__":
    a1_pattern()
    a2_reply()
    a3_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
