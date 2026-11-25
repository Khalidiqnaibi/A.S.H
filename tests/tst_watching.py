"""
tst_watching.py -- watching something is not being away.

25 Nov: twenty minutes of a video with no key and no mouse counted as "away" (the welcome-back came, the time was not counted as theirs).
Something playing on the computer caps the idle time that presence is judged from.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_watching.py
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


def w1_idle():
    print("\n=== W1. Idle time while something is playing ===")
    h = te.rig()
    rt = getattr(h, "rt", h)
    rt.periphery.sensors["idle"] = NS(seconds_idle=lambda: 1200.0)
    rt._playing = None
    check("nothing playing: twenty minutes idle is twenty minutes", rt._idle_seconds() == 1200.0, str(rt._idle_seconds()))
    rt._playing = {"item": "video", "title": "NoSQL lecture", "artist": ""}
    check("something playing: counted as present", rt._idle_seconds() == rt.WATCHING_IDLE_S, str(rt._idle_seconds()))
    rt.periphery.sensors["idle"] = NS(seconds_idle=lambda: 12.0)
    check("a short idle is left as it is", rt._idle_seconds() == 12.0, str(rt._idle_seconds()))


if __name__ == "__main__":
    w1_idle()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
