"""tst_wordinglearn.py -- wording is learned from being asked: a yes to "Did you mean ...?" teaches the meaning chooser that this wording IS that feature, a no teaches it that the wording
is none of them. A stub chooser and a stub route; nothing real is read or run. Run:  PYTHONIOENCODING=utf-8 python tests/tst_wordinglearn.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import meaning  # noqa: E402

check = tw.check


class Chooser:
    def __init__(self):
        self.learned = []

    def learn(self, text, label):
        self.learned.append((text, label))
        return True


def rig():
    rt = tm.rig()
    rt._choosers = {"meaning": Chooser()}
    rt._meaning_route = lambda text: meaning.Route("joke", "tell me a joke", 0.62, 0.02, "smile", "smile", sure=False) if "zorp" in text else None
    return rt


def main():
    print("\n=== W1. A yes teaches the wording ===")
    rt = rig()
    used, said = rt.say("zorp me a funny one")
    check("an unsure reading is asked about", not used and said and said[-1].startswith("Did you mean"), f"{used} {said}")
    used, said = rt.say("yes")
    check("a yes runs it and the wording is kept under that feature", rt._choosers["meaning"].learned == [("zorp me a funny one", "joke")], str(rt._choosers["meaning"].learned))
    check("and it is counted", rt.stats.get("meaning_learned") == 1)

    print("\n=== W2. A no teaches that it is none of them ===")
    rt = rig()
    rt.say("zorp me a funny one")
    used, said = rt.say("no")
    check("a no keeps the wording as 'none'", rt._choosers["meaning"].learned == [("zorp me a funny one", "none")] and said[-1] == "Okay, never mind.", f"{rt._choosers['meaning'].learned} {said}")

    print("\n=== W3. Anything else is a new line, and nothing is learned ===")
    rt = rig()
    rt.say("zorp me a funny one")
    used, said = rt.say("what's the capital of Peru")
    check("nothing learned", rt._choosers["meaning"].learned == [])
    rt = rig()
    rt._choosers = {}
    rt.say("zorp me a funny one")
    used, said = rt.say("yes")
    check("with no chooser (no encoder) a yes still just runs it", said and True)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
