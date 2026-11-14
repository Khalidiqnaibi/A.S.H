"""
tst_tickguard.py -- one failing part of the tick must not stop the rest.

_social_step ran about twenty parts of the loop inside one try. Eight of them were not guarded on their own, so a part that
kept raising -- a reminders save on a full disk, say -- stopped the calendar, the crash watch, the free moments and the
wondering every second, and was announced to the user as "internal errors". Each part now runs on its own, says so once a
minute, and counts as one error per ten minutes.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_tickguard.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check


def t1_one_part_fails():
    print("\n=== One part fails, the rest run ===")
    rt = te.rig()
    ran = []
    rt._watch_step = lambda now: ran.append("watch")
    rt._reminder_step = lambda now: (_ for _ in ()).throw(OSError(28, "No space left on device"))
    rt._presence_step = lambda now: ran.append("presence")
    rt._situation_step = lambda now, idle: ran.append("situation")
    rt._guard_step = lambda now: ran.append("guard")
    rt._feelings_step = lambda now: ran.append("feelings")
    rt.calendar = type("C", (), {"refresh": lambda self, now: ran.append("calendar")})()
    rt._calendar_at = -1e9

    logs = []

    class H(logging.Handler):
        def emit(self, rec):
            logs.append(rec.getMessage())
    h = H()
    logging.getLogger("ash.daemon.ambient").addHandler(h)
    try:
        for _ in range(30):
            rt.t[0] += 1
            rt._tick()
    finally:
        logging.getLogger("ash.daemon.ambient").removeHandler(h)
    check("the parts after the failing one still run, every tick", ran.count("presence") >= 25 and ran.count("guard") >= 25
          and ran.count("feelings") >= 25, str({k: ran.count(k) for k in set(ran)}))
    check("...including the calendar further down", "calendar" in ran, str(set(ran)))
    check("the failure is logged once, not thirty times", sum("Reminders failed" in m for m in logs) == 1,
          str([m for m in logs if "failed" in m]))
    check("...and is one error, not thirty", rt.stats.get("errors", 0) == 1, str(rt.stats.get("errors")))
    check("so the user is not told about 'internal errors' every second", rt._system_note(None, rt.t[0]) is not None
          and rt._system_note(None, rt.t[0]) is None)

    rt.t[0] += 700
    rt._tick()
    check("ten minutes on, a part that is still failing is counted once more", rt.stats.get("errors", 0) == 2,
          str(rt.stats.get("errors")))


def t2_helper():
    print("\n=== The helper ===")
    rt = te.rig()
    check("it returns what the part returned", rt._step("x", lambda a: a + 1, 1) == 2)
    check("a part that raises gives None, not an exception", rt._step("y", lambda: 1 / 0) is None)


if __name__ == "__main__":
    t1_one_part_fails()
    t2_helper()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
