"""tst_savedqueue.py -- "what are you saving for me?" lists the free-moment queue; "clear your queue" drops it. By rule, no model.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_savedqueue.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402

check = tw.check


def main():
    print("\n=== Q1. What is asked ===")
    L, C = AmbientRuntime._SAVED_LIST, AmbientRuntime._SAVED_CLEAR
    for t in ("What are you saving for me?", "what's in your queue", "what questions do you have saved", "what are you saving up for a free moment",
              "how many questions do you have for me", "list the saved questions", "Ash, what are you holding?"):
        check(f"{t!r} lists", bool(L.match(t)))
    for t in ("clear your queue", "drop the saved questions", "forget those questions", "please empty the queue", "delete your saved questions"):
        check(f"{t!r} clears", bool(C.match(t)))
    for t in ("what are you doing", "what is a queue", "what's saving the file", "clear the screen", "forget the milk", "what questions are on the exam"):
        check(f"{t!r} is neither", not L.match(t) and not C.match(t))

    print("\n=== Q2. In the runtime ===")
    rt = tm.rig()
    w = rt.wonder
    used, said = rt.say("What are you saving for me?")
    check("nothing saved: said so, no model", not used and said and "Nothing saved up" in said[-1], f"{used} {said}")
    w.raise_wonder("person", "omar", "Who is Omar?", "k1")
    w.raise_wonder("project", "ash", "What are you building in ASH?", "k2")
    used, said = rt.say("what's in your queue")
    check("two saved: counted and named", not used and said and "2 questions saved up" in said[-1] and "omar" in said[-1], f"{used} {said}")
    used, said = rt.say("clear your queue")
    check("cleared, said how many", not used and said and said[-1] == "Dropped 2 saved questions.", f"{used} {said}")
    check("and really gone", w.pending() == [])
    used, said = rt.say("clear your queue")
    check("again: nothing to drop", said and said[-1] == "There was nothing saved up.", str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
