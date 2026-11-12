"""
tst_remember.py -- "remember X is my friend", "who am I", "where did you get that".

From the Nov 8 session: "remember lieth is khalid's best friend" went to the model,
which said "I'm aware of this from other contexts" and stored nothing; "who am i"
got a line from the activity log; "where did u get this info" was answered from
the window title. These are rules now, on the real runtime and the real store.

  R1  Parsing: friend, "best friend" is a friend, owner = the user or "my", two people
  R2  Stored: the person and the relation are really in the knowledge store and entity memory
  R3  Said as stored; the model is not called; an unreadable "remember ..." still goes on
  R4  Undo takes it back
  R5  A family tie is not overwritten by "friend"
  R6  "Who am I" is answered from the name and the people on record
  R7  "Where did you get that" says whether it was a tool, the model, or ASH's rules
  R8  Sanitiser: a markdown link / placeholder address is dropped even if memory holds it
  R9  decisions: answered turns are kept apart from observations
  R10 Brain.sleep skips a second night inside the gap; a forced one runs
  R11 The same observation again is not re-processed; a different one is
  R12 The Ollama request asks to keep the model loaded

Run:  PYTHONIOENCODING=utf-8 python tests/tst_remember.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check


def rig():
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, knowledge_path=os.path.join(d, "knowledge.json"),
                wonder_path=os.path.join(d, "wonders.json"))
    rt.brain.user = "Khalid"
    return rt


def r1_parse():
    print("\n=== R1. Parsing ===")
    from src.daemon.remember import parse
    cases = [
        ("remember lieth is khalid's best friend", "Khalid", {"name": "Lieth", "role": "friend", "of": ""}),
        ("Remember that Lieth is my best friend", "Khalid", {"name": "Lieth", "role": "friend", "of": ""}),
        ("remember lieth is my friend", "", {"name": "Lieth", "role": "friend", "of": ""}),
        ("remember my best friend is Lieth", "Khalid", {"name": "Lieth", "role": "friend", "of": ""}),
        ("remember Omar is my cousin.", "Khalid", {"name": "Omar", "role": "cousin", "of": ""}),
        ("remember lieth is a good friend of mine", "Khalid", {"name": "Lieth", "role": "friend", "of": ""}),
        ("remember Omar is Sara's cousin", "Khalid", {"name": "Omar", "role": "cousin", "of": "Sara"}),
        ("remember to buy milk", "Khalid", None),
        ("remember it is my friend", "Khalid", None),
        ("remember Khalid is my friend", "Khalid", None),
        ("remember ash is my friend", "Khalid", None),
        ("remember the meeting is tomorrow", "Khalid", None),
    ]
    for text, user, want in cases:
        got = parse(text, user)
        check(f"{text!r} -> {want}", got == want, str(got))


def r2_r3_stored():
    print("\n=== R2/R3. Stored, and said as stored ===")
    rt = rig()
    try:
        before = len(rt.ash.calls)
        rt.say("remember lieth is khalid's best friend")
        p = rt.wonder.kb.people.get("lieth")
        check("the person is in the knowledge store as a friend (not best friend)",
              p is not None and p["relation"] == "friend", str(p))
        check("the answer says friend, once, and not best", rt.out and rt.out[-1] == "Got it -- Lieth is your friend.",
              str(rt.out))
        check("the model was not asked", len(rt.ash.calls) == before, str(rt.ash.calls[before:]))
        ents = rt.wonder.kb.entities
        found = [e for e in ents.store.all() if e.primary_identifiers.get("name") == "lieth"] if ents else []
        check("entity memory has Lieth with relation friend",
              (not ents) or (found and found[0].attributes.get("relation_to_user") == "friend"),
              str([e.attributes for e in found]))
        rt.say("remember lieth is khalid's best friend")
        check("said twice: no change, and says so", "already" in rt.out[-1].lower(), rt.out[-1])
        n = len(rt.ash.calls)
        rt.say("remember to buy milk tomorrow")
        check("a 'remember' it can't read still goes on to the model", len(rt.ash.calls) == n + 1, str(rt.ash.calls[n:]))
        rt.say("remember Omar is Sara's cousin")
        check("two people: the tie is stored", rt.wonder.kb.between_known("Omar", "Sara"), str(rt.out[-1]))
    finally:
        te.stop_all(rt)
    return rt


def r4_undo():
    print("\n=== R4. Undo ===")
    rt = rig()
    try:
        rt.say("remember lieth is my friend")
        check("stored", "lieth" in rt.wonder.kb.people)
        rt.say("undo that")
        check("undone: the person is gone", "lieth" not in rt.wonder.kb.people, str(rt.out[-1]))
    finally:
        te.stop_all(rt)


def r5_family():
    print("\n=== R5. Family is not overwritten ===")
    rt = rig()
    try:
        rt.wonder.kb.store_relation("Omar", role="cousin")
        rt.say("remember Omar is my brother")
        check("a different family tie is refused, honestly",
              rt.wonder.kb.relation_of("Omar") == "cousin" and "haven't changed" in rt.out[-1], rt.out[-1])
        rt.say("remember Omar is my cousin")
        check("the same tie again is not rewritten", "already" in rt.out[-1], rt.out[-1])
        rt.say("remember Dana is my friend")
        rt.say("remember Dana is my colleague")
        check("a non-family tie can be added over a friend",
              rt.wonder.kb.relation_of("Dana") == "colleague", rt.out[-1])
    finally:
        te.stop_all(rt)


def r6_identity():
    print("\n=== R6. Who am I ===")
    rt = rig()
    try:
        n = len(rt.ash.calls)
        rt.say("who am i")
        check("by rule: the name, no model", "Khalid" in rt.out[-1] and len(rt.ash.calls) == n, rt.out[-1])
        check("not an introduction of ASH", "A.S.H" not in rt.out[-1], rt.out[-1])
        rt.say("remember lieth is my friend")
        rt.say("who's the user?")
        check("names the people on record", "Lieth (friend)" in rt.out[-1], rt.out[-1])
        rt.brain.user = ""
        rt.say("what do you know about me")
        check("without a name it says so", "don't have your name" in rt.out[-1], rt.out[-1])
        n = len(rt.ash.calls)
        rt.say("who is messi")
        check("other 'who is' questions are not taken", len(rt.ash.calls) == n + 1)
    finally:
        te.stop_all(rt)


def r7_source():
    print("\n=== R7. Where did you get that ===")
    from src.daemon.explain import asks_source, source_reply
    for q in ("where did u get this info", "where did you get that", "how do you know that", "what's your source",
              "where is that from", "who told you that"):
        check(f"{q!r} is a source question", asks_source(q))
    for q in ("where did you put the file", "where is the nearest cafe", "how do you do that"):
        check(f"{q!r} is not", not asks_source(q))
    model = source_reply({"kind": "answer", "query": "who is messi", "intent": "conversation", "tool": None,
                          "pathway": "slow"})
    check("the model's own training is named as such", "trained" in model and "window" not in model.lower(), model)
    tool = source_reply({"kind": "answer", "intent": "get_time", "tool": "date_time", "pathway": "fast"})
    check("a tool is named", "date time tool" in tool, tool)
    check("ASH's own rules", "own rules" in source_reply({"kind": "asked", "query": "x"}))
    check("Wikipedia", "Wikipedia" in source_reply({"kind": "facts"}))
    rt = rig()
    try:
        rt._last_why = {"kind": "answer", "query": "who is messi", "intent": "conversation", "tool": None,
                        "pathway": "slow"}
        n = len(rt.ash.calls)
        rt.say("where did u get this info")
        check("said by rule, not by the model", "trained" in rt.out[-1] and len(rt.ash.calls) == n, rt.out[-1])
    finally:
        te.stop_all(rt)


def r8_sanitiser():
    print("\n=== R8. Sanitiser ===")
    from src.daemon.honest import sanitize
    known = "Khalid's friends: Lieth. contact: khalid.iqnaibi@email.com"
    out = sanitize("Your friend is [Khalid Afif Iqnaibi](mailto:khalid.iqnaibi@email.com) and Lieth.", known)
    check("a made-up address in a markdown link is gone, even when memory holds it", "@" not in out, out)
    out = sanitize("Write to [Ana](mailto:ana@real.org) tonight.", "ana@real.org")
    check("a link is flattened to its text", "mailto" not in out and "Ana" in out, out)
    out = sanitize("Email him at john@example.com.", "john@example.com")
    check("a placeholder domain is never real", "@" not in out, out)
    out = sanitize("Mail ana@real.org", "ana@real.org")
    check("a real address the user gave is kept", "ana@real.org" in out, out)


def r9_history():
    print("\n=== R9. decisions ===")
    import tst_brain as tb
    brain = tb.build_brain(tempfile.mkdtemp())
    for i in range(60):
        brain.observe(text=f"switched to window {i}", source="screen")
    brain.think("hello there")
    check("the observes did not push out the answer",
          len(brain.answer_history) >= 1 and all(t.pathway.value != "observe" for t in brain.answer_history),
          str([t.pathway.value for t in brain.answer_history]))
    check("the full ring still has them", len(brain.trace_history) >= 60)
    from types import SimpleNamespace
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "py", "ash.py"), encoding="utf-8").read()
    ns = {}
    start = src.index("    def explain_history")
    end = src.index("\n\nllm = build_llm()")
    import textwrap
    exec("from typing import List, Dict, Any\n" + textwrap.dedent(src[start:end]), ns)
    mine = SimpleNamespace(brain=brain)
    ans = ns["explain_history"](mine, 50)
    allh = ns["explain_history"](mine, 500, observed=True)
    check("explain_history gives the answered turns by default", 1 <= len(ans) < 10, str(len(ans)))
    check("...and everything on request", len(allh) > 60, str(len(allh)))


def r10_sleep():
    print("\n=== R10. One night at a time ===")
    import tst_brain as tb
    brain = tb.build_brain(tempfile.mkdtemp())
    a = brain.sleep(min_gap_s=3600)
    b = brain.sleep(min_gap_s=3600)
    check("the first runs", "skipped" not in a, str(a)[:80])
    check("the second, soon after, is skipped and says why", "skipped" in b and "ago" in b["skipped"], str(b))
    c = brain.sleep()
    check("with no gap asked (a manual sleep) it runs", "skipped" not in c)


def r11_observe():
    print("\n=== R11. A repeat observation ===")
    rt = rig()
    try:
        seen = []
        orig = rt.brain.observe
        rt.brain.observe = lambda **kw: (seen.append(kw.get("text")), orig(**kw))[1]
        from src.senses.base import Modality, SensorEvent

        def obs(text):
            rt.t[0] += 2
            rt.submit(SensorEvent(source="window", modality=Modality.TEXT, text=text, addressed=False,
                                  salience=0.1, urgency=0.0))
            rt._tick()
        for _ in range(3):
            obs("Switched to Windows PowerShell")
        obs("Switched to Opera")
        check("the same thing three times is processed once", seen.count("Switched to Windows PowerShell") == 1, str(seen))
        check("a different one is processed", seen.count("Switched to Opera") == 1, str(seen))
        rt.t[0] += 60
        obs("Switched to Windows PowerShell")
        check("the same one a minute later is processed again", seen.count("Switched to Windows PowerShell") == 2, str(seen))
    finally:
        te.stop_all(rt)


def r12_keepalive():
    print("\n=== R12. Keep the model loaded ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ash_llm_under_test", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    sent = []

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"message": {"content": "hi"}}

        def iter_lines(self, *a, **k):
            return iter(())
    orig = L.requests.post
    L.requests.post = lambda url, json=None, timeout=None, **k: (sent.append(json), R())[1]
    try:
        llm = L.LLM(mode="ollama", ollama_model="m", ollama_url="http://x/api/chat")
        os.environ.pop("ASH_OLLAMA_KEEP_ALIVE", None)
        llm._call_ollama([{"role": "user", "content": "hi"}])
        check("the request carries a keep_alive of hours", sent and sent[-1].get("keep_alive") == "6h", str(sent[-1:]))
        os.environ["ASH_OLLAMA_KEEP_ALIVE"] = "-1"
        llm._call_ollama([{"role": "user", "content": "hi"}])
        check("it can be overridden", sent[-1].get("keep_alive") == "-1")
    finally:
        L.requests.post = orig
        os.environ.pop("ASH_OLLAMA_KEEP_ALIVE", None)


def r13_reasoning():
    print("\n=== R13. A model's thinking is never the answer ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ash_llm_under_test2", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    spoken = [
        "Okay, so the user is asking, \"who is messi.\" From what I know, messi is a Spanish football player.",
        "Alright, so the user mentioned \"wew\" and I'm trying to figure out what they're asking about.",
        "Okay, so I'm trying to figure out what to do here. The user mentioned wew.",
        "Let me break this down step by step.",
        "Hmm, let me think about this.",
        "The user is asking me to create a circle in ASCII art.",
    ]
    for s in spoken:
        check(f"reasoning caught: {s[:46]!r}", L.looks_like_reasoning(s))
    fine = ["Lionel Messi is an Argentine footballer.", "Well, I think that's Inter Miami.", "So the answer is 42.",
            "Okay. It's 1:35 PM.", "I'm trying my best, but I don't know that one.", "Sure, the user guide is in docs."]
    for s in fine:
        check(f"an answer is left alone: {s[:46]!r}", not L.looks_like_reasoning(s))
    check("a closed think block goes", L._strip_think("<think>hmm</think>It is 5.").strip() == "It is 5.")
    check("an unclosed think block goes to the end", L._strip_think("Hi. <think>so the user").strip() == "Hi.")
    check("a missing opening tag goes", L._strip_think("let me see...</think>It is 5.").strip() == "It is 5.")

    sent = []

    class R:
        status_code = 200

        def __init__(self, body, code=200):
            self.text, self.status_code = body, code

    import json as _j
    plan = [R(_j.dumps({"message": {"content": "", "thinking": "Okay, so the user is asking who is messi"}}))]
    orig = L.requests.post
    L.requests.post = lambda url, json=None, timeout=None, **k: (sent.append(dict(json)), plan[0])[1]
    try:
        llm = L.LLM(mode="ollama", ollama_model="r1", ollama_url="http://x/api/chat")
        out = llm._call_ollama([{"role": "user", "content": "who is messi"}])
        check("only thinking came back: the answer is empty, not the thinking", out == "", repr(out))
        check("the request asked the model not to think", sent[-1].get("think") is False, str(sent[-1].keys()))
        plan[0] = R(_j.dumps({"message": {"content": "Okay, so the user is asking about Messi. Let me recall."}}))
        check("reasoning written as the answer is dropped", llm._call_ollama([{"role": "user", "content": "x"}]) == "")
        plan[0] = R(_j.dumps({"message": {"content": "Messi is an Argentine footballer."}}))
        check("a real answer passes", llm._call_ollama([{"role": "user", "content": "x"}]).startswith("Messi"))
        # a model with no thinking option: 400 mentioning it -> sent again without
        seq = [R("{\"error\": \"\\\"mistral\\\" does not support think\"}", 400),
               R(_j.dumps({"message": {"content": "Hello."}}))]
        sent.clear()
        L.requests.post = lambda url, json=None, timeout=None, **k: (sent.append(dict(json)), seq.pop(0))[1]
        out = llm._call_ollama([{"role": "user", "content": "x"}])
        check("a model without 'think' is asked again without it", out == "Hello." and len(sent) == 2
              and "think" not in sent[1], f"{out!r} {[('think' in s) for s in sent]}")
    finally:
        L.requests.post = orig


if __name__ == "__main__":
    r1_parse()
    r2_r3_stored()
    r4_undo()
    r5_family()
    r6_identity()
    r7_source()
    r8_sanitiser()
    r9_history()
    r10_sleep()
    r11_observe()
    r12_keepalive()
    r13_reasoning()
    print("\nALL PASS" if tw.OK else "\nSOME FAILED")
    sys.exit(0 if tw.OK else 1)
