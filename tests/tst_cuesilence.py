"""
tst_cuesilence.py -- "On it." / "Give me a second." is never followed by nothing.

Live (18 Nov): 23 of 66 answers were a thinking cue, and seven of those were followed by silence ("No reply to give") because the notice that
the model gave nothing is limited to once in ten minutes -- the person heard "Working on it." and then nothing, again and again.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_cuesilence.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import tst_effects as te  # noqa: E402

check = tw.check


def c1_cue_then_nothing():
    print("\n=== C1. A cue, then nothing ===")
    rt = te.rig()
    ev = NS(addressed=True, source="microphone", text="what about it", meta={})
    dec = NS(likely=False, as_dict=lambda: {})
    rt._last_no_reply_notice = rt.clock()                   # told a minute ago: the limit would keep it quiet
    n = len(rt.out)
    rt._model_unavailable(ev, dec)
    check("with no cue said, the limit holds: silence, as designed", len(rt.out) == n, str(rt.out[n:]))
    rt._cue_fired = True                                    # "Working on it." was said
    rt._model_unavailable(ev, dec)
    check("after a cue it is always said that no answer came", len(rt.out) == n + 1 and "brain" in rt.out[-1], str(rt.out[n:]))
    check("and the flag is spent: the next empty one is limited again", rt._cue_fired is False)
    rt._last_no_reply_notice = rt.clock()
    n = len(rt.out)
    rt._model_unavailable(ev, dec)
    check("so it does not repeat on its own", len(rt.out) == n)


def c2_cue_sets_flag():
    print("\n=== C2. The cue sets the flag when it fires ===")
    rt = te.rig(think_cue_s=0.05)
    timer = rt._thinking_cue(True)
    check("a timer was made, flag clear", timer is not None and rt._cue_fired is False)
    import time
    time.sleep(0.3)
    check("when it fired the flag was set", rt._cue_fired is True)
    timer.cancel()


if __name__ == "__main__":
    c1_cue_then_nothing()
    c2_cue_sets_flag()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
