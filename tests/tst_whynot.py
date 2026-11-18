"""
tst_whynot.py -- when ASH cannot ask the saved question it says the REAL reason.

Live (18 Nov): "this is a free moment, what do you have to ask me?" -> "1 question saved up, but I'm not allowed to ask them yet (muted, quiet
hours, or told to stop)" -- the user was not muted, it was not quiet hours, nobody had said stop. A guess, worded as a fact.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_whynot.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.freemoment import Environment  # noqa: E402

check = tw.check


def env(**kw):
    base = dict(idle_s=30.0, muted=False, quiet_hours=False, paused=False, conversation_open=False, plan_active=False, free_moment=True)
    base.update(kw)
    return Environment(**base)


def w1_reasons():
    print("\n=== W1. The real reason ===")
    w = tw.rig()
    check("nothing waiting: no reason", w.why_not(env(), w.t[0]) == "")
    w.observe("I told Omar something about work")
    now = w.t[0] + 5000
    check("a question that could be asked: no reason (it is asked)", w.why_not(env(), now) == "", w.why_not(env(), now))
    check("muted is said as muted", w.why_not(env(muted=True), now) == "I'm muted")
    check("quiet hours", w.why_not(env(quiet_hours=True), now) == "it's quiet hours")
    check("privacy mode", w.why_not(env(paused=True), now) == "privacy mode is on")
    w.muted_subjects["omar"] = now
    why = w.why_not(env(), now)
    check("a subject the user said to stop asking about is named", "stop asking about" in why and "Omar" in why, why)
    w.muted_subjects.clear()
    w.subject_last["omar"] = now - 1800
    why = w.why_not(env(), now)
    check("a subject asked about a few minutes ago says so", "I asked about Omar" in why and "minutes ago" in why, why)
    w.subject_last["omar"] = now - 3 * 3600
    why = w.why_not(env(), now)
    check("...or how many hours", "3 hours ago" in why, why)
    w.off = True
    check("asking switched off", w.why_not(env(), now) == "asking questions is switched off" or "off" in w.why_not(env(), now), w.why_not(env(), now))


def w2_runtime():
    print("\n=== W2. In what it says ===")
    import tst_freemoment as tf
    h = tf.rt_rig(enabled=False)
    h.say("I told Omar something about work")
    h.rt.wonder.muted_subjects["omar"] = h.clock[0]
    h.say("What do you have in store for me for a free moment? This is a free moment.")
    said = h.said()[-1]
    check("it says what it has and the real reason", "about Omar" in said and "stop asking about Omar" in said, said)
    check("and not the old guess", "muted, quiet hours, or told to stop" not in said, said)


if __name__ == "__main__":
    w1_reasons()
    w2_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
