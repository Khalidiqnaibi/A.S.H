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
            fh.write(json.dumps({"t": NOW - 600 * (i + 1), "text": txt, "why": why, "tool": tool, "took_s": took}) + "\n")
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
