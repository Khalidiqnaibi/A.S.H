"""
tst_digest.py -- the day's failures as a file, and "what went wrong today" from the same numbers.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_digest.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import digest  # noqa: E402

check = tw.check
NOW = 1795700000.0


def files():
    d = tempfile.mkdtemp()
    f, u = os.path.join(d, "failures.jsonl"), os.path.join(d, "unrouted.jsonl")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": NOW - 3600, "when": "2026-11-25 14:05", "lines": ["Bruh.", "That's wrong."], "said": ["What time is it?"]}) + "\n")
        fh.write(json.dumps({"t": NOW - 1800, "when": "2026-11-25 14:35", "what": "fact corrected", "heard": "my brother is not Omar", "said": "Okay -- I've taken out Omar."}) + "\n")
        fh.write(json.dumps({"t": NOW - 90000, "when": "old", "lines": ["x"], "said": []}) + "\n")
    with open(u, "w", encoding="utf-8") as fh:
        for i, (txt, why, tool, took) in enumerate([("Spell Segway", "no_tool_matched", False, 27.9), ("Spell Segway", "no_tool_matched", False, 12.0),
                                                    ("What do you see?", "tool_narrated", True, 5.6), ("tell me a story about space", "no_tool_matched", False, 40.0)]):
            row = {"t": NOW - 600 * (i + 1), "text": txt, "why": why, "tool": tool, "took_s": took}
            if took == 40.0:
                row["note"] = "the model had to be loaded first; memory was at 91%"
            fh.write(json.dumps(row) + "\n")
    return f, u, d


def d1_summary():
    print("\n=== D1. What the day held ===")
    f, u, d = files()
    s = digest.summary(f, u, NOW)
    check("two failures in the last day (the old one is out)", len(s["failures"]) == 2, str(len(s["failures"])))
    check("three lines went to the model (the tool one did not)", len(s["model_lines"]) == 3)
    check("two were slow", len(s["slow"]) == 2 and s["slow"][0]["took_s"] == 40.0)
    check("one came up twice", len(s["repeated"]) == 1 and s["repeated"][0][0] == "spell segway")
    path = digest.write(os.path.join(d, "digest"), f, u, NOW)
    text = open(path, encoding="utf-8").read()
    check("a file, named for the day", path.endswith(".md") and "digest-2026-11-" in path, path)
    check("the file names the failures, the slow ones and the next rule", "repair mode started" in text and "fact corrected" in text and "40 s" in text and "2 x “Spell Segway”" in text, text)
    check("a slow line carries why it was slow", "40 s -- \u201ctell me a story about space\u201d -- the model had to be loaded first; memory was at 91%" in text, text)
    from src.daemon.unrouted import Unrouted
    from src.daemon.ambient import AmbientRuntime
    ur = Unrouted(os.path.join(d, "x.jsonl"))
    ur.model_turn("a slow one", "no_tool_matched", 50.0, note="the machine was squeezed (memory is at 90%)")
    ur.model_turn("a quick one", "no_tool_matched", 1.0)
    check("the ledger keeps the note, and only when there is one", ur.rows[0].get("note") == "the machine was squeezed (memory is at 90%)" and "note" not in ur.rows[1])
    import types
    rt = types.SimpleNamespace(governor=types.SimpleNamespace(status=lambda: {"squeezed": False, "memory": 91.0, "cpu": 95.0, "why": ""}))
    check("the runtime names the reasons it can know", AmbientRuntime._slow_note(rt, True) == "the model had to be loaded first; memory was at 91%; CPU was at 95%", AmbientRuntime._slow_note(rt, True))
    rt.governor.status = lambda: {"squeezed": True, "memory": 91.0, "cpu": 10.0, "why": "memory is at 91%"}
    check("squeezed: said, with why", AmbientRuntime._slow_note(rt, False) == "the machine was squeezed (memory is at 91%)")
    check("calm and warm: nothing to say", AmbientRuntime._slow_note(types.SimpleNamespace(governor=None), False) == "")
    check("nothing kept: said so", "Nothing went wrong" in digest.render(digest.summary(None, None, NOW), "2026-11-25"))


def d2_say():
    print("\n=== D2. 'What went wrong today' ===")
    for t in ("what went wrong today", "What went wrong?", "show me the failures", "how many times did you fail", "list your mistakes"):
        check(f"{t!r}", digest.asks_digest(t))
    for t in ("what time is it", "what went wrong with my code", "why did that go wrong"):
        check(f"not: {t!r}", not digest.asks_digest(t))
    f, u, _ = files()
    say = digest.say(f, u, NOW)
    check("counts, and the line that kept coming back", "2 times" in say and "3 lines" in say and "2 of them slow" in say and "Spell Segway" in say, say)


if __name__ == "__main__":
    d1_summary()
    d2_say()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
