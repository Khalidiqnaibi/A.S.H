"""tst_selfknow.py -- ASH knows what it runs on and does not claim senses it hasn't: 'what model are you running', 'what can you see', 'can you hear me' are answered from the
live state; a model answer that says it sees/hears when the camera or microphone is off is replaced; a changed model is said once. Fakes only, offline.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_selfknow.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.daemon import selfknow as sk  # noqa: E402

check = tw.check
F = {"model": "mistral", "light_model": "qwen3.5:1.5b", "face": True, "camera": False, "screen": True, "muted": False, "paused": False, "episodes": 1400, "entities": 380}


def main():
    print("\n=== S1. Asking ===")
    for t, want in [("what model are you running", "model"), ("which model is answering", "model"), ("what can you see right now", "see"), ("can you see me", "see"),
                    ("can you hear me", "hear"), ("are you listening right now", "hear"), ("Ash, what are you running on", "model"), ("what is the capital of peru", None), ("can you see the problem here", None)]:
        check(f"{t!r} -> {want}", sk.asks(t) == want, str(sk.asks(t)))
    print("\n=== S2. Answers come from the state ===")
    check("the model, and that it is mostly rules", "mistral" in sk.answer("model", F) and "rules" in sk.answer("model", F))
    check("camera off is said", "camera is off" in sk.answer("see", F), sk.answer("see", F))
    check("privacy on means nothing is seen", sk.answer("see", dict(F, paused=True)).startswith("No"))
    check("a muted microphone is said", "microphone is off" in sk.answer("hear", dict(F, muted=True)))
    print("\n=== S3. Claims that are not so ===")
    check("'I can see you look tired' with the camera off is replaced", sk.ungrounded("I can see you look tired today.", F) and "can't see you" in sk.ungrounded("I can see you look tired today.", F), str(sk.ungrounded("I can see you look tired today.", F)))
    check("with the camera on it stands", sk.ungrounded("I can see you look tired today.", dict(F, camera=True)) is None)
    check("'I heard you' with the microphone muted is replaced", sk.ungrounded("I heard you say that.", dict(F, muted=True)) is not None)
    check("an ordinary sentence is left alone", sk.ungrounded("The match starts at eight and I see why you like it.", F) is None)
    print("\n=== S4. A changed model ===")
    check("said when it changed", sk.model_change("mistral", "qwen3.5:4b") == "I'm thinking with qwen3.5:4b now; last time it was mistral.")
    check("not when it didn't, or the first time", sk.model_change("mistral", "mistral") == "" and sk.model_change("", "mistral") == "")
    print("\n=== S5. In the runtime ===")
    import tst_modellast as tm
    rt = tm.rig()
    used, said = rt.say("what can you see right now")
    check("answered by rule, no model", not used and said and "camera" in said[-1], f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("are you listening right now")
    check("hearing answered by rule", not used and said and True, f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
