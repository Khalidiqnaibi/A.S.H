"""
tst_repair.py -- frustration changes how ASH behaves.

25 Nov: "Bruh." x5, "Never mind." x6, "That's wrong, you do have that information", "You are really frustrating me today Ash." -- and ASH went on as before:
a filler, 20 s, a confident wrong answer, "How else might I assist?".

Run:  PYTHONIOENCODING=utf-8 python tests/tst_repair.py
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import repair  # noqa: E402

check = tw.check


def r1_signals():
    print("\n=== R1. What counts ===")
    r = repair.Repair(clock=lambda: 0.0)
    for t in ["Bruh.", "never mind", "You are really frustrating me today Ash.", "That's wrong, Ash, you do have that information.",
              "Again, that's unrelated to what I ask.", "stop making stuff up", "you're not listening"]:
        check(f"a signal: {t!r}", r.score(t)[0] > 0, str(r.score(t)))
    for t in ["what time is it", "thanks", "remind me to call mom", "that's wrong, I meant Tuesday"[:0] or "open Chrome", "play some music", "who is Linda"]:
        check(f"not one: {t!r}", r.score(t)[0] == 0.0)
    check("the same question twice is one", r.score("what is the weather like today", "what is the weather like today")[0] == 1.0)


def r1b_shorten():
    print("\n=== R1b. The reply is cut to one sentence ===")
    check("feelings talk and the offer go, the answer stays",
          repair.shorten("You seem frustrated, let's try a different approach. The model is qwen2.5:3b. How about we look up something fun together?") == "The model is qwen2.5:3b.")
    check("an answer that is only a question back is 'I don't know'", repair.shorten("Could you tell me more? How can I help?") == repair.NOTHING)
    check("only the first answer sentence is kept", repair.shorten("It is ten past two. It was ten past one an hour ago.") == "It is ten past two.")
    check("empty is 'I don't know'", repair.shorten("") == repair.NOTHING)


def r2_threshold():
    print("\n=== R2. Enough of them, in time ===")
    t = [0.0]
    r = repair.Repair(clock=lambda: t[0])
    check("one 'bruh' is nothing", r.hear("Bruh.") is None and not r.active())
    t[0] += 10
    check("a second 'never mind' is not yet", r.hear("never mind") is None and not r.active())
    t[0] += 10
    a = r.hear("Bruh.")
    check("the third crosses it: one apology, mode on", a == repair.APOLOGY and r.active())
    t[0] += 5
    check("and only one", r.hear("you are really frustrating me") is None and r.active())
    check("the model is told", "frustrated" in r.note() and "I don't know" in r.note())
    t[0] += 30
    check("a thanks ends it", r.hear("thanks") is None and not r.active() and r.note() == "")
    t[0] += 1000
    r2 = repair.Repair(clock=lambda: t[0])
    r2.hear("Bruh.")
    t[0] += 400
    r2.hear("never mind")
    t[0] += 400
    r2.hear("Bruh.")
    check("signals older than five minutes do not add up", not r2.active())
    r3 = repair.Repair(clock=lambda: t[0])
    check("a single strong one ('frustrating me') is enough with one more", r3.hear("You are really frustrating me today Ash.") == repair.APOLOGY)
    t[0] += 700
    check("ten quiet minutes end it", not r3.active())


def r3_log():
    print("\n=== R3. What went wrong is kept ===")
    p = os.path.join(tempfile.mkdtemp(), "failures.jsonl")
    t = [1000.0]
    r = repair.Repair(clock=lambda: t[0], path=p)
    r.hear("what is the model")
    r.hear("That's wrong, you do have that information.")
    r.hear("You are really frustrating me today.")
    rows = [json.loads(x) for x in open(p, encoding="utf-8")]
    check("one row, with the lines that led to it", len(rows) == 1 and any("frustrating" in x for x in rows[0]["lines"]) and "what is the model" in rows[0]["said"], str(rows))


def r4_runtime():
    print("\n=== R4. On the runtime ===")
    import tst_effects as te
    rt = te.rig(think_cue_s=0.05)
    check("on by default", rt.repair is not None)
    n = len(rt.out)
    rt.say("Bruh.", source="console")
    rt.say("never mind", source="console")
    rt.say("You are really frustrating me today Ash", source="console")
    sorry = [o for o in rt.out[n:] if o == repair.APOLOGY]
    check("one apology was said", len(sorry) == 1, str(rt.out[n:]))
    check("repair mode is on and the brain carries the note", rt.repair.active(rt.clock()) and "frustrated" in str(getattr(rt.brain, "repair_note", "")))
    m0 = len(rt.ash.calls) if hasattr(rt, "ash") and hasattr(rt.ash, "calls") else 0
    o0 = len(rt.out)
    rt.say("Ugh.", source="console")
    rt.say("hmm", source="console")
    check("a backchannel sound gets no answer and no model turn", len(rt.out) == o0 and (not hasattr(rt, "ash") or not hasattr(rt.ash, "calls") or len(rt.ash.calls) == m0), str(rt.out[o0:]))
    seen = []
    rt.face = NS(on_feeling=lambda a: seen.append(a))
    rt._repair_shown = False
    rt._repair_face(True)
    rt._repair_face(True)
    check("the face looks sorry while it is on, once", seen == ["sad"], str(seen))
    rt._repair_face(False)
    check("and goes back when it ends", seen == ["sad", ""], str(seen))
    rt.face = None
    check("no thinking filler while it is on", rt._thinking_cue(True) is None)
    rt.say("thanks", source="console")
    check("a thanks ends it and clears the note", not rt.repair.active(rt.clock()) and getattr(rt.brain, "repair_note", "x") == "")
    t = rt._thinking_cue(True)
    check("fillers are back", t is not None)
    if t is not None:
        t.cancel()
    # a turn that gave nothing, while it is on
    rt.say("bruh", source="console")
    rt.say("you are really frustrating me", source="console")
    rt._last_no_reply_notice = -1e12
    out0 = len(rt.out)
    rt._model_unavailable(NS(addressed=True, source="console", text="x", meta={}), NS(likely=False, as_dict=lambda: {}))
    check("an empty turn says 'I don't know that one' (not 'My brain's not answering')", any(o.startswith("I don't know that one.") for o in rt.out[out0:]), str(rt.out[out0:]))


def r5_prompt():
    print("\n=== R5. The model is told ===")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "brain", "brain.py"), encoding="utf-8").read()
    check("the slow path adds the repair note to the trace note", 'rn = str(getattr(self, "repair_note", "") or "")' in src and "note = (note + \" \" + rn).strip()" in src)


if __name__ == "__main__":
    r1_signals()
    r1b_shorten()
    r2_threshold()
    r3_log()
    r4_runtime()
    r5_prompt()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
