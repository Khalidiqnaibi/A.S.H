"""
tst_restart.py -- "restart yourself" asks first, then restarts (not the computer, and not a shutdown).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_restart.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import tst_effects as te  # noqa: E402
from src.daemon import voicectl  # noqa: E402

check = tw.check


def r1_parse():
    print("\n=== R1. What is asked ===")
    for t in ("restart yourself", "Restart Ash", "reboot yourself", "relaunch the assistant", "start yourself again"):
        c = voicectl.parse(t)
        check(f"{t!r} -> restart", c is not None and c.name == "restart", str(c))
    for t in ("restart the computer", "restart my pc", "restart the video", "shut down", "restart"):
        c = voicectl.parse(t)
        check(f"not a restart of ASH: {t!r}", c is None or c.name != "restart", str(c))


def r2_runtime():
    print("\n=== R2. Ask first, then restart ===")
    rt = te.rig()
    shut, restart = [], []
    rt.request_shutdown = lambda: shut.append(1)
    rt.request_restart = lambda: restart.append(1)
    rt.say("restart yourself")
    check("it asks first and does nothing yet", "Restart ASH?" in rt.out[-1] and not restart and not shut, rt.out[-1])
    rt.say("yes")
    check("a yes restarts (and does not just shut down)", restart == [1] and not shut, f"{restart} {shut}")
    check("and says so", "Restarting" in rt.out[-1], rt.out[-1])
    rt.say("shut down")
    rt.say("yes")
    check("a shutdown after it is still a shutdown", shut == [1] and restart == [1], f"{restart} {shut}")
    rt2 = te.rig()
    rt2.request_restart = None
    rt2.say("restart yourself")
    rt2.say("yes")
    check("with no way to restart, it says so", "can't restart myself" in rt2.out[-1], rt2.out[-1])


if __name__ == "__main__":
    r1_parse()
    r2_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
