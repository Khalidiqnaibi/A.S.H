"""tst_dangling.py -- a line cut off on a word nothing can end on is asked about, not guessed at by the model (it took 47 s).
Run:  PYTHONIOENCODING=utf-8 python tests/tst_dangling.py"""
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
    print("\n=== D1. What is cut off ===")
    D = AmbientRuntime._DANGLING
    for t in ("What's the latest update on", "tell me about the", "can you look at my", "what is the weather in", "how do I get from here to", "show me a picture of",
              "Ash, what's the news about"):
        check(f"{t!r} is cut off", bool(D.match(t)))
    for t in ("What's the latest update", "open notepad", "what time is it", "tell me about yourself", "what's the weather like in Amman", "I am so tired of that", "what is that", "do you remember that", "who are you", "what was that"):
        check(f"{t!r} is whole", not D.match(t))

    print("\n=== D2. In the runtime ===")
    rt = tm.rig()
    used, said = rt.say("What's the latest update on")
    check("asked what the rest was, no model", not used and said and said[-1] == "On what? I lost the end of that.", f"{used} {said}")
    used, said = rt.say("What's the latest update on")
    check("the same cut-off line again goes through (it is what they mean)", used or not (said and said[-1].startswith("On what")), f"{used} {said}")
    used, said = rt.say("tell me about the")
    check("another one: a plain 'say the rest'", not used and said and said[-1] == "I only caught part of that -- say the rest?", f"{used} {said}")
    used, said = rt.say("so what port am I on")
    check("a whole question that ends on a preposition is not cut off", used or not (said and "lost the end" in said[-1]), f"{used} {said}")
    used, said = rt.say("what is the capital of France")
    check("a whole question is not touched", used or (said and "On what" not in said[-1]), f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
