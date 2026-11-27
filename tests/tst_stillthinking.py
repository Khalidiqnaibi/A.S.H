"""tst_stillthinking.py -- a slow answer: the cue, then once at half a minute 'Still working on it.' with why; an answer cancels both.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_stillthinking.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402

check = tw.check


def rig():
    rt = tm.rig()
    said = []
    rt.response_sink = lambda t, m: said.append(t)
    rt.cfg.think_cue_s = 0.05
    rt.governor = types.SimpleNamespace(status=lambda: {"squeezed": False, "memory": 92.0, "cpu": 5.0, "why": ""})
    return rt, said


def main():
    old = AmbientRuntime.STILL_THINKING_S
    AmbientRuntime.STILL_THINKING_S = 0.2
    try:
        rt, said = rig()
        cue = rt._thinking_cue(True)
        time.sleep(0.5)
        check("the cue, then 'still working on it' with why", len(said) == 2 and said[1] == "Still working on it. Memory is at 92%, which is why it is slow. Say “ask Gemini” and I'll hand it over.", str(said))
        cue.cancel()
        rt, said = rig()
        cue = rt._thinking_cue(True)
        time.sleep(0.1)
        cue.cancel()
        time.sleep(0.4)
        check("an answer after the cue stops the second line", len(said) == 1, str(said))
        rt, said = rig()
        cue = rt._thinking_cue(True)
        cue.cancel()
        time.sleep(0.4)
        check("an answer before the cue: nothing at all", said == [], str(said))
        rt, said = rig()
        rt.governor = None
        cue = rt._thinking_cue(True)
        time.sleep(0.5)
        cue.cancel()
        check("no reason known: just 'Still working on it.'", said[-1:] == ["Still working on it."], str(said))
        rt, said = rig()
        check("not to ASH: no cue", rt._thinking_cue(False) is None)
    finally:
        AmbientRuntime.STILL_THINKING_S = old
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
