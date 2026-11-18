"""
tst_smallfixes.py -- three small things from the 18 Nov log.

  S1  A routine's name is cut between words, not in the middle of one ("something I do when I watch mo")
  S2  "Remember it as ..." said again is not kept as a note ("Noted: it as something you do when you watch movies")
  S3  A company is not a person: ASH asked "Is Samsung a classmate?"

Run:  PYTHONIOENCODING=utf-8 python tests/tst_smallfixes.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.wondering import Wonder, NOT_PERSON  # noqa: E402
from src.daemon.notes import Notes  # noqa: E402

check = tw.check


def s1_routine_name():
    print("\n=== S1. A routine's name ===")
    w = tw.rig()
    q = Wonder(id="r1", kind="routine", subject="routine", question="?", key="routine:x", data={"steps": ["Stremio-shell-ng", "Edge"]})
    got = w._parse_answer(q, "Remember it as something I do when I watch movies.")
    check("the whole name, not 'something I do when I watch mo'", got is not None and got["name"] == "something I do when I watch movies", str(got))
    long = w._parse_answer(q, "call it the thing I do every single morning before I start any work at all today")
    check("a very long one is cut at a word", long is not None and len(long["name"]) <= 40 and not long["name"].endswith("a") or long["name"].count(" ") >= 3, str(long))
    got = w._parse_answer(q, "yes, call it movie time")
    check("a short one is as before", got is not None and got["name"] == "movie time", str(got))


def s2_not_a_note():
    print("\n=== S2. 'It as ...' is not a note ===")
    for t in ["Remember it as something I do when I watch movies.", "remember that as my movie routine", "keep in mind this as a rule"]:
        check(f"not a note: {t!r}", Notes.wants_note(t) is None, str(Notes.wants_note(t)))
    check("a real note is still one", Notes.wants_note("remember that the plumber comes on Friday") == "the plumber comes on Friday")


def s3_brands():
    print("\n=== S3. A company is not a person ===")
    for b in ("samsung", "sony", "stremio", "nosql", "nvidia", "playstation", "mongodb"):
        check(f"{b} is not a name", b in NOT_PERSON)
    w = tw.rig()
    for _ in range(3):
        w.observe("I take notes in Samsung Notes during my classes, Samsung is great for the lecture")
    check("no 'is Samsung a classmate?' question comes of it", not any("Samsung" in r["subject"] for r in w.pending() if r["kind"] == "person"),
          str([r["question"] for r in w.pending()]))


if __name__ == "__main__":
    s1_routine_name()
    s2_not_a_note()
    s3_brands()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
