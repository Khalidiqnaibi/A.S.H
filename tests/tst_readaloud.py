"""tst_readaloud.py -- "read it to me" reads the text of the window in front, by rule, with the screen reader's own refusals.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_readaloud.py"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402

check = tw.check


def main():
    print("\n=== R1. What is asked ===")
    R = AmbientRuntime._READ_ALOUD
    for t in ("read it to me", "read this out loud", "Ash, read the screen", "can you read the page to me please", "what does it say",
              "read what's on my screen", "read that aloud", "what does this say"):
        check(f"{t!r} reads", bool(R.match(t)))
    for t in ("read the file notes.txt", "read me a story", "read my email", "what does the fox say to you in the morning", "i read it yesterday",
              "read the news", "read it again tomorrow when I'm back"):
        check(f"{t!r} is not this", not R.match(t))

    print("\n=== R2. In the runtime ===")
    view = {"ok": True, "title": "Notes - Opera", "subject": "Notes", "withheld": None, "reader": True, "text": "Segue means a smooth transition.  Next line."}

    def with_sensor(v):
        rt = tm.rig()
        sc = types.SimpleNamespace(look_now=lambda: dict(v))
        real = rt.periphery.sensors

        class Sensors(dict):                           # asked for by name, never walked by the tick
            def get(self, k, d=None):
                return sc if k == "screenshot" else real.get(k, d)
        rt.periphery.sensors = Sensors(real)
        return rt

    used, said = with_sensor(view).say("read it to me")
    check("the text, with the window's name, no model", not used and said and said[-1] == "Notes: Segue means a smooth transition. Next line.", f"{used} {said}")
    long = dict(view, text=("A sentence that goes on. " * 40))
    used, said = with_sensor(long).say("read this out loud")
    check("a long page is cut at a sentence, not read for a minute", said and len(said[-1]) < 460 and said[-1].endswith("."), str(said and len(said[-1])))
    for v, want in ((dict(view, withheld="private"), "private or blocked"), (dict(view, text=""), "couldn't make out any text"),
                    (dict(view, reader=False, text=""), "screen reader installed"), (dict(view, ok=False), "can't read the screen")):
        used, said = with_sensor(v).say("what does it say")
        check(f"refusal: {want}", not used and said and want in said[-1], f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("read it to me")
    check("no screen sensor: it goes where it did (not claimed)", not (said and "Segue" in said[-1]))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
