"""
tst_declined.py -- a tool that says "I'm switched off / not on this machine" did its job.

"Look at my screen" with screen capture off (or no sensor) returned ok=False, so
the brain counted a failure: the critic was punished, the tool's track record
fell, and the fast path escalated to the model to say what the tool had already
said. A tool may now return {"ok": False, "declined": True, "error": "..."}:

  D1  Fast path: the decline is said as it is, the critic gets no punishment,
      the outcome model records nothing, and no model call is made
  D2  Slow path: the same -- tool_success is None, not False
  D3  A real failure (no "declined") still counts as one
  D4  The screen tool marks "no sensor" and "sensor unavailable" as declines

Run:  PYTHONIOENCODING=utf-8 python tests/tst_declined.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_brain as tb  # noqa: E402
from src.brain.signals import Pathway  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


STATE = {"mode": "ok"}


def tool(q):
    if STATE["mode"] == "declined":
        return {"ok": False, "declined": True, "error": "Screen capture is off, so I can't look."}
    if STATE["mode"] == "failed":
        return {"ok": False, "error": "it crashed"}
    return "2026-08-24T10:00:00"


def warm_brain():
    brain = tb.build_brain(tempfile.mkdtemp())
    brain.system1.registry.add("date_time", tool, ["get_time"])
    for q in ("what time is it please", "what time is it now", "what is the time", "what time is it currently",
              "tell me what time is it", "what time is it right now", "so what time is it again",
              "what time is it at this moment", "what's the time right now"):
        brain.think(q)
    return brain


def d1_fast():
    print("\n=== D1. Fast path ===")
    brain = warm_brain()
    seen = []
    orig = brain.predictive.record_outcome
    brain.predictive.record_outcome = lambda intent, ok: (seen.append((intent, ok)), orig(intent, ok))[1]
    STATE["mode"] = "declined"
    tb.LLM_CALLS["n"] = 0
    r = None
    for q in ("what time is it, tell me", "time now please", "what's the time exactly", "the time please"):
        tb.LLM_CALLS["n"] = 0
        r = brain.think(q)
        if r.pathway == Pathway.FAST and r.tool_output and r.tool_output.get("declined"):
            break
    check("(a fast turn reached the declining tool)", r.pathway == Pathway.FAST and r.tool_output
          and r.tool_output.get("declined"), f"{r.pathway.value}")
    check("it says what the tool said", r.text == "Screen capture is off, so I can't look.", r.text)
    check("no model was asked", tb.LLM_CALLS["n"] == 0, str(tb.LLM_CALLS["n"]))
    check("not a success and not a failure", r.tool_success is None)
    check("the outcome model recorded nothing for it", not [x for x in seen if x[1] is False] and not seen, str(seen))
    pend = brain.system1._pending
    check("the critic is not told it failed", pend is not None and pend["tool_success"] is None
          and pend["escalated"] is False, str(pend and (pend["tool_success"], pend["escalated"])))
    STATE["mode"] = "ok"


def d2_slow():
    print("\n=== D2. Slow path ===")
    brain = tb.build_brain(tempfile.mkdtemp())
    brain.system1.registry.add("date_time", tool, ["get_time"])
    STATE["mode"] = "declined"
    seen = []
    orig = brain.predictive.record_outcome
    brain.predictive.record_outcome = lambda intent, ok: (seen.append((intent, ok)), orig(intent, ok))[1]
    r = brain.think("what time is it")          # cold: deliberates
    check("(slow path ran the tool)", r.pathway in (Pathway.SLOW, Pathway.ESCALATED) and r.tool_output
          and r.tool_output.get("declined"), r.pathway.value)
    check("tool_success is None, not False", r.tool_success is None)
    check("nothing recorded against the tool", not seen, str(seen))
    STATE["mode"] = "ok"


def d3_real_failure():
    print("\n=== D3. A real failure still counts ===")
    brain = tb.build_brain(tempfile.mkdtemp())
    brain.system1.registry.add("date_time", tool, ["get_time"])
    STATE["mode"] = "failed"
    seen = []
    orig = brain.predictive.record_outcome
    brain.predictive.record_outcome = lambda intent, ok: (seen.append((intent, ok)), orig(intent, ok))[1]
    r = brain.think("what time is it")
    check("tool_success False", r.tool_success is False, str(r.tool_success))
    check("recorded as a failure", ("get_time", False) in seen, str(seen))
    STATE["mode"] = "ok"


def d4_screen_tool():
    print("\n=== D4. The screen tool ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "screen_tool_under_test", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "screen_tool.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    st.set_looker(None)
    out = st.screen_view_tool("what's on my screen")
    check("no sensor: a decline", out.get("ok") is False and out.get("declined") is True, str(out))
    st.set_looker(lambda: {"ok": False, "declined": True, "error": "I can't look at the screen on this machine: x."})
    out = st.screen_view_tool("what's on my screen")
    check("sensor unavailable: a decline, with the reason", out.get("declined") is True and "x." in out["error"], str(out))
    st.set_looker(lambda: {"ok": False, "error": "grab failed"})
    out = st.screen_view_tool("what's on my screen")
    check("a grab that really failed is not", out.get("declined") is False, str(out))
    st.set_looker(None)


if __name__ == "__main__":
    for fn in (d1_fast, d2_slow, d3_real_failure, d4_screen_tool):
        fn()
    print("\n" + ("ALL PASS" if OK else "FAILURES"))
    sys.exit(0 if OK else 1)
