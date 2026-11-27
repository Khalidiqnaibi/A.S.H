"""tst_relations.py -- 27 Nov: "what architecture are you using" / "what's System 1" / "what relationships do you know" / "draw me a tree" / "second degree
relationships related to Khalid" / "What project?" / "why is that?" (after the emotions) -- 42-87 s each on the model, with made-up answers. By rule now.
Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_relations.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import relations, selfmodel  # noqa: E402
from src.daemon.activity import Topic  # noqa: E402

check = tw.check


def graph(rt):
    kb = rt.wonder.kb
    kb.people["omar"] = {"name": "Omar", "relation": "cousin", "between": {"sara": "friends"}}
    kb.people["sara"] = {"name": "Sara", "relation": "colleague", "between": {"dana": "sisters"}}
    kb.people["dana"] = {"name": "Dana", "relation": "", "between": {}}
    return kb


def main():
    print("\n=== R1. How it is built ===")
    for t, k in (("what architecture are you using", "how:architecture"), ("what's your brain structure like", "how:architecture"),
                 ("what's system 1 are you using and how does it work", "how:system1"), ("what is system two", "how:system2")):
        check(f"{t!r} -> {k}", selfmodel.parse(t) == k, str(selfmodel.parse(t)))
    rt = tm.rig()
    used, said = rt.say("what architecture are you using")
    check("answered from what is true, not 'I don't have a specific architecture'", not used and said and said[-1].startswith("Two speeds."), f"{used} {said}")

    print("\n=== R2. The people graph ===")
    for t, k in (("what relationships do you know", "list"), ("draw me a tree of relationships you know", "tree"),
                 ("what are the second degree relationships you can see related to khalid", "second")):
        check(f"{t!r} -> {k}", (relations.parse(t) or ("",))[0] == k, str(relations.parse(t)))
    rt = tm.rig()
    kb = graph(rt)
    s = relations.answer(kb, "Khalid", "list")
    check("the list: who each is to you, and the ties", s == "You: Dana, Omar (cousin), Sara (colleague). Between them: Omar and Sara are friends; Sara and Dana are sisters.", s)
    s = relations.answer(kb, "Khalid", "tree")
    check("the tree: you at the root", s.splitlines()[0] == "Khalid" and "  - Omar (cousin)" in s and "      - Sara (friends)" in s, s)
    s = relations.answer(kb, "Khalid", "second", "Omar")
    check("two steps from Omar: through Sara to Dana", "Omar → Sara (friends) → Dana (sisters)" in s, s)
    s = relations.answer(kb, "Khalid", "second", "Khalid")
    check("two steps from you", "Omar →" not in s or "Two steps from you" in s, s)
    used, said = rt.say("what relationships do you know")
    check("in the runtime, no model", not used and said and said[-1].startswith("You: "), f"{used} {said}")
    check("nobody known: says how to teach it", relations.answer(types.SimpleNamespace(people={}), "x", "list").startswith("I don't know anyone yet"))

    print("\n=== R3. 'What project?' and 'why is that?' follow what was just said ===")
    rt = tm.rig()
    rt.activity._cur = types.SimpleNamespace(subject="ambient.py", project="Branch", start=time.time() - 120)
    rt.wonder.kb.store_term("Branch", "im working on my first novel")
    used, said = rt.say("what am I working on")
    used, said = rt.say("what project?")
    check("the project of the window in front, with what it is", not used and said and said[-1] == "Branch -- your first novel.", f"{used} {said}")
    rt = tm.rig()
    rt.say("what's your emotional status in detail")
    rt._last_out, rt._last_out_at = "A little short-tempered. Emotions (out of 10): boredom 10.0, frustration 10.0.", rt.clock()
    used, said = rt.say("why is that?")
    check("'why is that?' after the emotions: from the engine, no model", not used and said and "push frustration and boredom up" in said[-1], f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
