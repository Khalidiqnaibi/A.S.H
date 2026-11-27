"""tst_keptturns.py -- a request a rule answered still goes through ASH's memory: the user's own line is kept once (not 'yes'/'ok', not a guest, not twice, not system text), the way a
model-answered one always was. Fakes only. Offline.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_keptturns.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402

check = tw.check


class Mem:
    def __init__(self):
        self.turns = []

    def commit_turn(self, user_text, ash_text, user_actor, ash_actor, importance=0.5, **kw):
        self.turns.append((user_text, importance))
        return {}


def main():
    print("\n=== K1. Rule-answered requests are kept ===")
    rt = tm.rig()
    mem = Mem()
    rt.brain.memory = mem
    used, said = rt.say("what can you do with google docs")
    check("a rule-answered request is kept once, lightly", not used and mem.turns == [("what can you do with google docs", 0.35)], f"{used} {mem.turns}")
    n = len(mem.turns)
    rt.say("what can you do with google docs")
    check("the same line twice in a row is not kept twice", len(mem.turns) == n)
    mem.turns.clear()
    rt.say("yes")
    rt.say("kk")
    check("'yes' and 'ok' are not kept", not mem.turns, str(mem.turns))
    rt.say("what is the capital of peru")
    check("a model-answered turn is left to the model path (not double-kept here)", all(t[0] != "what is the capital of peru" for t in mem.turns), str(mem.turns))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
