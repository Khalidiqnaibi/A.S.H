"""
tst_unrouted.py -- which lines reached the language model, and why.

  U1  the ledger: a line per model turn, scrubbed, capped, never a guest's; the share of turns the model answered
  U2  common(): the lines that keep reaching the model, grouped by meaning
  U3  the runtime: a rule answer is not a model turn; a line that no rule takes is, with its reason

Run:  PYTHONIOENCODING=utf-8 python tests/tst_unrouted.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.unrouted import Unrouted, reason_of  # noqa: E402

check = tw.check


def u1_ledger():
    print("\n=== U1. The ledger ===")
    p = os.path.join(tempfile.mkdtemp(), "u.jsonl")
    u = Unrouted(p, clock=lambda: 1000.0, scrub=lambda s: s.replace("4111111111111111", "[card]"))
    u.model_turn("what do you think of my code", "no_tool_matched", took_s=63.0)
    u.model_turn("my card is 4111111111111111", "conversation")
    u.model_turn("what is the time", "tool_narrated", tool=True)
    u.model_turn("a guest said this", "conversation", guest=True)
    u.model_turn("   ", "conversation")
    u.answered_elsewhere()
    u.answered_elsewhere()
    check("three lines kept (not the guest's, not an empty one)", len(u.rows) == 3, str(len(u.rows)))
    check("scrubbed like the journal", all("4111" not in r["text"] for r in u.rows) and "[card]" in u.rows[1]["text"])
    check("the share the model answered: 5 turns of 7... counts guests and empties too", abs(u.rate() - 5 / 7) < 1e-9, str(u.rate()))
    again = Unrouted(p)
    check("kept across a restart", len(again.rows) == 3 and again.rows[0]["why"] == "no_tool_matched")
    small = Unrouted(None, max_lines=5)
    for i in range(12):
        small.model_turn(f"line {i}", "conversation")
    check("capped", len(small.rows) == 5 and small.rows[-1]["text"] == "line 11")

    class R:
        tool_used = None

        class trace:
            notes = ["Brain: SLOW path -- no proposal cleared minimum score (0.03 < 0.1)"]
    check("reason: no tool matched", reason_of(R) == "no_tool_matched")

    class T:
        tool_used = "calculate"
        trace = None
    check("reason: a tool ran and the model said it", reason_of(T) == "tool_narrated")
    check("reason: nothing to go on", reason_of(object()) == "conversation")


def u2_common():
    print("\n=== U2. What keeps reaching the model ===")
    u = Unrouted(None)
    for t in ("what do you think of my code", "what do you think of my code", "explain the code to me", "who is khalid in your memory",
              "who is khalid in your memory", "who is khalid in your memory", "draw me a diagram"):
        u.model_turn(t, "no_tool_matched")
    g = u.common()
    check("by the same words: the repeated lines, most first", g and g[0]["example"] == "who is khalid in your memory" and g[0]["n"] == 3
          and g[1]["n"] == 2, str([(x["example"], x["n"]) for x in g]))

    def embed(texts):                       # two meanings: about code, about people
        return np.array([[1.0, 0.0] if ("code" in t) else [0.0, 1.0] for t in texts])
    g = u.common(embed)
    check("by meaning (the encoder): the two code questions are one group", g and any(x["n"] == 3 and "code" in x["example"] for x in g), str([(x["example"], x["n"]) for x in g]))
    check("a one-off is not a group", all(x["n"] >= 2 for x in g))
    check("the report says it", "x3" in u.report(embed) and "answered" in u.report(embed))


def u2b_suggestions():
    print("\n=== U2b. Written at sleep ===")
    p = os.path.join(tempfile.mkdtemp(), "u.jsonl")
    u = Unrouted(p, clock=lambda: 50.0)
    for t in ["what is the plan", "what is the plan", "what is the plan", "once only", "twice", "twice"]:
        u.model_turn(t, "conversation")
    n = u.write_suggestions()
    sp = p + ".suggest.json"
    got = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    check("only the groups that came up three times or more are suggested", n == 1 and [g["example"] for g in got.get("groups", [])] == ["what is the plan"], str(got))
    check("with how many turns the model had of all", got.get("model") == 6 and got.get("turns") == 6)
    check("none: an empty list, no error", Unrouted(None).write_suggestions() == 0)
    import tst_honest as th
    rt = th.rig("Sure.")
    rt.unrouted = Unrouted(os.path.join(tempfile.mkdtemp(), "u.jsonl"), clock=lambda: 1.0)
    for _ in range(3):
        rt.unrouted.model_turn("tell me about octopuses", "conversation")
    rep = rt._sleep_phase(force=True)
    check("the sleep phase writes them", rep.get("unrouted_groups") == 1 and os.path.exists(rt.unrouted.path + ".suggest.json"), str(rep))


def u2c_usage():
    print("\n=== U2c. Asked how much it uses the model ===")
    for t in ("how often do you use the model", "how many times did you use the LLM today", "how much do you use the language model",
              "your model usage today", "how many of my questions went to the model"):
        check(f"{t!r} asks", Unrouted.asks_usage(t))
    for t in ("what model are you", "use the model for this", "how often do I use spotify", "what is the language model"):
        check(f"{t!r} does not", not Unrouted.asks_usage(t))
    u = Unrouted(None)
    check("nothing yet: says so", "nothing went to the language model" in u.usage_reply())
    for _ in range(3):
        u.answered_elsewhere()
    u.model_turn("tell me about octopuses", "conversation", took_s=63.0)
    u.model_turn("tell me about octopuses", "conversation", took_s=5.0)
    s = u.usage_reply()
    check("how many, the share, the slowest and what keeps going to it", "answered 2 of 5 turns (40%)" in s and "slowest took 63 s" in s and "octopuses" in s, s)
    import tst_modellast as tm
    rt = tm.rig()
    used, said = rt.say("how often do you use the model")
    check("in the runtime: by rule, from its own numbers", not used and said and ("language model" in said[-1]), f"{used} {said}")


def u3_runtime():
    print("\n=== U3. In the runtime ===")
    import tst_honest as th
    rt = th.rig("Sure, here is an answer.")
    path = os.path.join(tempfile.mkdtemp(), "u.jsonl")
    rt.unrouted = Unrouted(path, clock=lambda: 5.0)
    rt.say("what can you do for now, Ash?")                    # a rule answers
    n0 = len(rt.ash_.calls)
    check("a rule answer: no model call, not written", n0 == 0 and not rt.unrouted.rows and rt.unrouted.stats["model"] == 0, str(rt.unrouted.status()))
    rt.say("tell me something strange about octopuses")        # nothing takes this: the model does
    check("a line no rule takes reaches the model, and is recorded with its words", len(rt.ash_.calls) == 1 and rt.unrouted.rows
          and "octopuses" in rt.unrouted.rows[-1]["text"], str(rt.unrouted.rows))
    check("...with a reason", rt.unrouted.rows[-1]["why"] in ("conversation", "no_tool_matched", "ambiguous", "escalated", "tool_narrated"),
          str(rt.unrouted.rows[-1]))
    check("the share is counted, and shown in the status", 0 < rt.status()["unrouted"]["rate"] <= 1.0, str(rt.status()["unrouted"]))
    check("and written to disk", os.path.exists(path) and "octopuses" in open(path, encoding="utf-8").read())


if __name__ == "__main__":
    u1_ledger()
    u2_common()
    u2b_suggestions()
    u2c_usage()
    u3_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)
