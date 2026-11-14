"""
tst_ctlsay.py -- `ashctl say` goes through the loop, as a person typing would.

It used to call the brain from the control thread: at the same moment as the loop, past the attention gate, the lane, every
rule ("never mind", "turn off the mic", the taste and game readers) and with a brain that is not thread-safe -- which is also
why trying a rule-based feature with `say` in a live test showed the model's answer instead of the rule's.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_ctlsay.py
"""

import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check


def run_with_loop(rt, fn, ticks=400):
    """Call fn() on a thread, tick the loop here until it returns."""
    out = []
    th = threading.Thread(target=lambda: out.append(fn()), daemon=True)
    th.start()
    for _ in range(ticks):
        rt.t[0] += 1
        rt._tick()
        th.join(0.02)
        if not th.is_alive():
            break
    return out[0] if out else None, th


def t1_ask():
    print("\n=== ask() ===")
    rt = te.rig()
    who = []
    orig = rt.ash.think

    def think(query, image=None, context=""):
        who.append(threading.get_ident())
        return orig(query, image=image, context=context)
    rt.ash.think = think
    caller = []
    got, th = run_with_loop(rt, lambda: (caller.append(threading.get_ident()),
                                         rt.ask("tell me something about the sea", timeout=20))[1])
    check("a question comes back answered", got is not None and got["text"] == "Sure.", str(got))
    check("...with its pathway", got and got["pathway"] == "slow", str(got))
    check("the brain was asked by the loop, not from the caller's thread", bool(who) and bool(caller)
          and all(w != caller[0] for w in who), f"{who} vs caller {caller}")
    check("the event went through the loop (it counts as an event)", rt.stats["events"] >= 1, str(rt.stats))
    check("no waiter is left behind", rt._waiters == [])

    n = len(who)
    try:
        got, th = run_with_loop(rt, lambda: rt.ask("turn on the mic", timeout=20))
        check("a rule answers a rule's request, with no model: the reason to go through the loop",
              got is not None and got["text"] == "Microphone on." and len(who) == n, str(got))
    finally:
        te.stop_all(rt)

    rt2 = te.rig()
    out = []
    got = rt2.ask("anything", timeout=0.3)
    check("nothing answering in time is None, not a hang", got is None)
    check("...and the waiter is removed", rt2._waiters == [])


def t2_control():
    print("\n=== the control command ===")
    from src.daemon.service import ControlServer
    rt = te.rig()
    cs = ControlServer.__new__(ControlServer)
    cs.runtime = rt
    got = cs.dispatch("say hello there")
    check("with the loop not running, it still answers directly (as before)", got.get("ok") and got.get("text") == "Sure.", str(got))
    rt.alive = lambda: True
    got, th = run_with_loop(rt, lambda: cs.dispatch("say tell me about the sea"))
    check("with the loop running, it goes through it", got and got.get("ok") and got.get("text") == "Sure.", str(got))
    rt.ask = lambda text, timeout=240: None
    got = cs.dispatch("say anything")
    check("no answer in time is an error the caller can read", got.get("ok") is False and "try again" in got.get("error", ""), str(got))


if __name__ == "__main__":
    t1_ask()
    t2_control()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
