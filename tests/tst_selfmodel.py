"""
tst_selfmodel.py -- ASH answers questions about itself from the live runtime.

25 Nov: "What model are you using" -> "You're not using any specific model"; "What hardware are you using" -> "That's what I just said";
"What can you tell me about this machine" -> the CPU load; "What's new in the update" -> a stale answer about the user's frustration;
"What do you know about yourself" -> "You haven't told me anything about yourself Ash".

Run:  PYTHONIOENCODING=utf-8 python tests/tst_selfmodel.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import selfmodel  # noqa: E402

check = tw.check


def s1_parse():
    print("\n=== S1. What is asked ===")
    want = {"What model are you using, Ash?": "model", "which model are you running": "model", "what's your model": "model",
            "What model are you?": "model", "which model is this": "model", "what model are you now": "model", "what model runs you": "model",
            "what model are you building": None,
            "What hardware are you using, Ash?": "hardware", "What can you tell me about this machine, Ash?": "hardware",
            "what are your specs": "hardware", "what gpu do you have": None, "what's your hardware": "hardware",
            "What's new with you in the new update?": "whatsnew", "what's new": "whatsnew", "what changed in the update": "whatsnew",
            "What's recently added to your memory.": "whatsnew",
            "What was the last update?": "whatsnew", "what's the latest update": "whatsnew", "what was added last": "whatsnew",
            "what was the most recent change": "whatsnew", "when were you last updated": "whatsnew",
            "What do you know about yourself Ash?": "self", "describe yourself": "self", "what do you know about you": "self",
            "How much do you remember?": "size", "what did you learn today": "size"}
    for t, k in want.items():
        if k is None:
            continue
        check(f"{t!r} -> {k}", selfmodel.parse(t) == k, str(selfmodel.parse(t)))
    for said, k in {"how do you use your emotions": "how:emotions", "Why do you use your emotions?": "how:emotions", "how do you remember things": None,
                    "how does your memory work": "how:memory", "what happens when you sleep": "how:sleep", "do you sleep": "how:sleep",
                    "what is a concept": "how:concepts", "what is repair mode": "how:repair", "how do you work with my data": "how:privacy",
                    "how are my privacy handled": None}.items():
        if k:
            check(f"{said!r} -> {k}", selfmodel.parse(said) == k, str(selfmodel.parse(said)))
    for said, k in {"How to use your concepts, Ash.": "how:concepts", "How can you learn concepts, Ash?": "how:teach", "How to teach you new concepts.": "how:teach",
                    "What can your concept memory do?": "how:concepts", "Show me your concepts tree.": "concepttree",
                    "what concepts do you have": "concepttree", "list your concepts": "concepttree", "how does your concept memory work": "how:concepts"}.items():
        check(f"{said!r} -> {k}", selfmodel.parse(said) == k, str(selfmodel.parse(said)))
    class _C:
        def __init__(s, i, a):
            s.id, s.activations = i, a

    class _Net:
        concepts = {1: _C(1, 9), 2: _C(2, 5)}

        def describe(s, i):
            return {1: "NoSQL", 2: "documents"}[i]

        def neighbours(s, i, n):
            return [(2 if i == 1 else 1, 0.5)]
    said = selfmodel.say("concepttree", NS(brain=NS(concepts=NS(net=_Net()))))
    check("the concept tree is read from the real graph: count, the busiest, and what they link to", said.startswith("I hold 2 concepts. The ones that come up most: NoSQL (linked to documents)"), said)
    check("how to teach it says the real ways", "from Google, learn how to" in selfmodel.say("how:teach", NS()))
    for t in ["how do you open chrome", "how do you spell necessary", "what is a hash map", "how do i use my memory better"]:
        check(f"not a how-do-you-work question: {t!r}", selfmodel.parse(t) is None, str(selfmodel.parse(t)))
    for t in ["What's the temperature today?", "what is a hash map", "what's the news", "who is Linda", "what do you know about me", "what can you do",
              "who are you", "tell me about yourself", "what do you know about Linda", "what's new on YouTube", "how are you", "tell me about Samsung Notes"]:
        check(f"not about itself: {t!r}", selfmodel.parse(t) is None, str(selfmodel.parse(t)))


def fake(model="qwen2.5:3b", light="", notes=None):
    llm = NS(_ollama_model=model, _light_model=light, _mode="ollama")
    ash = NS(llm=llm, episodic_memory=NS(store=NS(episodes={i: 1 for i in range(12)})), entity_memory=NS(store=NS(entities={i: 1 for i in range(5)})))
    brain = NS(concepts=NS(net=NS(concepts={i: 1 for i in range(30)})))
    wonder = NS(kb=NS(people={"a": {}, "b": {}}))
    return NS(ash=ash, brain=brain, wonder=wonder, _started_at=__import__("time").time() - 3 * 3600)


def s2_answers():
    print("\n=== S2. The answers ===")
    rt = fake(light="qwen2.5:1.5b")
    m = selfmodel.say("model", rt)
    check("the model it really uses, and that rules do the rest", "qwen2.5:3b" in m and "qwen2.5:1.5b" in m and "rules and tools" in m, m)
    check("no model: it says so", "no language model" in selfmodel.say("model", fake(model="")))
    h = selfmodel.say("hardware", rt)
    check("the machine, not the load", h.startswith("This machine has") and "threads" in h and "memory" in h, h)
    check("not the CPU percentage line", "CPU at" not in h, h)
    z = selfmodel.say("size", rt)
    check("what it holds, from the stores", "12 remembered moments" in z and "5 things" in z and "2 people" in z and "30 concepts" in z and "3 hours" in z, z)
    w = selfmodel.say("whatsnew", rt)
    check("what is new: from the real history (or says it cannot see it)", w.startswith("Lately:") or "can't see" in w, w)
    me = selfmodel.say("self", rt)
    check("about itself: what it is, the model, what it holds", me.startswith("I'm A.S.H") and "qwen2.5:3b" in me and "12 remembered moments" in me, me[:200])


def s3_runtime():
    print("\n=== S3. On the runtime, before the notes and the facts ===")
    import tst_effects as te
    rt = te.rig(wonder=True, notes=True, notes_path=os.path.join(tempfile.mkdtemp(), "notes.json"))
    rt.ash.llm = NS(_ollama_model="qwen2.5:3b", _light_model="", _mode="ollama")
    rt.say("remember that my favorite number is 4242", source="console")
    n = len(rt.ash.calls)
    rt.say("What model are you using, Ash?", source="console")
    check("answered by rule with the real model", "qwen2.5:3b" in rt.out[-1] and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("What do you know about yourself Ash?", source="console")
    check("'about yourself' is not the notes' 'what do you know about X'", rt.out[-1].startswith("I'm A.S.H") and "told me anything" not in rt.out[-1], rt.out[-1][:120])
    rt.say("what do you know about me", source="console")
    check("'about me' is still about the person", "I'm A.S.H" not in rt.out[-1], rt.out[-1][:100])
    rt.say("tell me about Samsung Notes", source="console")
    check("and other subjects are untouched", "I'm A.S.H" not in rt.out[-1])


def s4_prompt():
    print("\n=== S4. The model is told what is true about ASH ===")
    from src.brain.system2 import System2Deliberator
    rt = fake()
    blk = selfmodel.block(rt)
    check("a short block, with the real model", blk.startswith("[ABOUT YOU]") and "qwen2.5:3b" in blk and len(blk) < 900, blk)
    s2 = System2Deliberator.__new__(System2Deliberator)
    seen = []
    s2._call_llm = lambda human, system="": (seen.append(human), "ok")[1]
    s2.last_narration = {}
    from src.brain.signals import Drives
    s2.narrate("A.S.H", "hello", {}, {"core": "", "episodic": "", "entity": ""}, "", Drives(), "", about=blk)
    check("it is in the prompt when given", seen and "[ABOUT YOU]" in seen[0] and "qwen2.5:3b" in seen[0], str(seen)[:200])
    seen.clear()
    s2.narrate("A.S.H", "hello", {}, {"core": "", "episodic": "", "entity": ""}, "", Drives(), "")
    check("and not when it is not", seen and "[ABOUT YOU]" not in seen[0])


if __name__ == "__main__":
    s1_parse()
    s2_answers()
    s3_runtime()
    s4_prompt()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
