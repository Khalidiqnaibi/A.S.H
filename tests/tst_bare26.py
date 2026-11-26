"""tst_bare26.py -- 26 Nov lines that waited for the model: a bare verb ("Ash, Switch."), "This is a free moment.", "That's funny." -- by rule now.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_bare26.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402
from src.daemon.courtesy import Courtesy  # noqa: E402

check = tw.check


def main():
    print("\n=== B1. A bare verb ===")
    B = AmbientRuntime._BARE_VERB
    for t in ("Ash, Switch.", "open", "close.", "play", "look up", "remind me", "turn off", "switch to...", "search for"):
        check(f"{t!r} is a bare verb", bool(B.match(t)))
    for t in ("open notepad", "switch to word", "play music", "close it", "remind me to call mom", "start over please now ok"):
        check(f"{t!r} is not", not B.match(t))
    rt = tm.rig()
    used, said = rt.say("Ash, Switch.")
    check("'Ash, Switch.' -> 'Switch to what?', no model", not used and said and said[-1] == "Switch to what?", f"{used} {said}")
    used, said = rt.say("open")
    check("'open' -> 'Open what?'", not used and said and said[-1] == "Open what?", f"{used} {said}")

    print("\n=== B2. 'This is a free moment.' ===")
    F = AmbientRuntime._FREE_BARE
    for t in ("This is a free moment.", "free moment", "I have a free moment", "it's a free moment now", "I'm free now", "Ash, I've got a spare moment"):
        check(f"{t!r} invites", bool(F.match(t)))
    for t in ("is this a free moment?", "I'm free on Tuesday", "the free moment feature is annoying"):
        check(f"{t!r} does not, by itself", not F.match(t))
    rt = tm.rig()
    used, said = rt.say("This is a free moment.")
    rt._tick()
    check("taken without the model", not used, f"{used} {said}")

    print("\n=== B3. A laugh ===")
    c = Courtesy()
    for t in ("That's funny.", "haha", "hahaha", "lol", "lol that's hilarious", "that's so funny", "good one", "you're hilarious"):
        check(f"{t!r} is a laugh", c.kind(t) == "laugh", str(c.kind(t)))
    for t in ("that's funny, what time is it", "is that funny?", "funny how it works out every time like that"):
        check(f"{t!r} is not", c.kind(t) != "laugh", str(c.kind(t)))
    rt = tm.rig()
    used, said = rt.say("That's funny.")
    check("answered with a laugh, no model (it took 41 s)", not used and said and said[-1] in ("Ha.", "Heh -- glad that landed.", "Ha, right?", "I thought so too."), f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
