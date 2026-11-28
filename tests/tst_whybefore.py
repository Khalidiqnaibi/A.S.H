"""
tst_whybefore.py -- "why did you say the one before it?" explains the answer before the last, not the last.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_whybefore.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import tst_effects as te  # noqa: E402
from src.daemon import explain  # noqa: E402

check = tw.check


def b1_parse():
    print("\n=== B1. What is asked ===")
    for t in ("why did you say the one before it", "Why did you say the one before that?", "why did you say what you said before that", "why did you say the answer before"):
        check(f"{t!r}", explain.asks_before(t))
    for t in ("why did you say that", "why", "what did you say before that", "why did you open Chrome"):
        check(f"not: {t!r}", not explain.asks_before(t))


def b2_runtime():
    print("\n=== B2. On the runtime ===")
    h = te.rig()
    rt = getattr(h, "rt", h)
    rt._last_why = {"kind": "asked", "query": "your timetable"}
    rt.response_sink("Linda is your cousin.", {"addressed": True, "source": "console", "pathway": "slow"})
    rt._last_why = {"kind": "correction"}
    rt.response_sink("Paris is the capital of France.", {"addressed": True, "source": "console", "pathway": "fast"})
    h.say("why did you say the one before it", source="console")
    from src.daemon.explain import explain as ex
    check("it explains the earlier answer (the timetable), not the last (a correction)", h.out[-1] == ex({"kind": "asked", "query": "your timetable"}) and h.out[-1] != ex({"kind": "correction"}), h.out[-1])
    h.say("why did you say that", source="console")
    check("'why did you say that' is still the last one", h.out[-1] == ex({"kind": "correction"}), h.out[-1])
    e = te.rig()
    e.say("why did you say the one before it", source="console")
    check("nothing before: said plainly", "haven't said anything before" in e.out[-1], e.out[-1])


def b3_mismatch():
    print("\n=== B3. 'Why are you saying something and writing something else?' ===")
    check("asked", explain.asks_mismatch("Why are you saying something and writing something else?"))
    check("not other whys", not explain.asks_mismatch("why did you say that") and not explain.asks_mismatch("what time is it"))
    h = te.rig()
    rt = getattr(h, "rt", h)
    n = len(rt.ash.calls)
    h.say("Why are you saying something and writing something else?", source="console")
    check("answered from the real cause, not the camera, no model", "brief mode" in h.out[-1] and "camera" not in h.out[-1] and len(rt.ash.calls) == n, h.out[-1])
    rt.speech.brief = True
    h.say("Why are you saying something and writing something else?", source="console")
    check("brief mode on: says so and how to turn it off", "full answers" in h.out[-1], h.out[-1])


if __name__ == "__main__":
    b1_parse()
    b2_runtime()
    b3_mismatch()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
