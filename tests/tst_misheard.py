"""
tst_misheard.py -- "I said Opera, not Obra": learned once, put right in every later transcript.

Speech recognition mishears the same words the same way ("Open Obra Ash" for Opera). The user already says what they meant, and nothing was
listening for it. Now a correction is kept (state/misheard.json), applied to later transcripts (whole words), and the word meant goes into the
recogniser's hint.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_misheard.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.misheard import Misheard  # noqa: E402

check = tw.check


def m1_parse():
    print("\n=== M1. What is a correction ===")
    m = Misheard()
    for t, want in [("I said Opera, not Obra", ("Obra", "Opera")), ("no, I said Khalid not Khaled", ("Khaled", "Khalid")),
                    ("I didn't say Obra, I said Opera", ("Obra", "Opera")), ("not Obra, I said Opera", ("Obra", "Opera")),
                    ("I meant Spotify, not spot a fly", ("spot a fly", "Spotify")), ('I said "Opera" not "Obra"', ("Obra", "Opera"))]:
        check(f"{t!r} -> {want}", m.parse(t) == want, str(m.parse(t)))
    for t in ("it's fine, not great", "I said yes not no", "I said it was fine not great", "I said that is fine not great", "I said hello",
              "not great, it's fine", "I said the quick brown fox jumps over the lazy dog not something else entirely", "what did I say"):
        check(f"{t!r} is not a correction of the recogniser", m.parse(t) is None, str(m.parse(t)))


def m2_apply():
    print("\n=== M2. Applying it ===")
    p = os.path.join(tempfile.mkdtemp(), "m.json")
    m = Misheard(p)
    check("said back", m.hear("I said Opera, not Obra") == "Got it -- when I hear \u201cObra\u201d I'll take it as \u201cOpera\u201d.")
    check("a later transcript is put right, whole words, the case kept", m.fix("Open Obra Ash and obra again") == "Open Opera Ash and opera again".replace("opera", "Opera"),
          m.fix("Open Obra Ash and obra again"))
    check("not inside another word", m.fix("the obrazil band") == "the obrazil band")
    check("the word meant is a hint for the recogniser", m.words() == ["Opera"])
    again = Misheard(p)
    check("kept across a restart", again.fix("open obra") == "open Opera", again.fix("open obra"))
    check("an unrelated line is untouched", m.fix("what time is it") == "what time is it" and m.fix("") == "")
    check("'forget that correction' takes the last one back", "I won't change" in m.hear("forget that correction") and m.fix("open obra") == "open obra")
    check("with nothing to take back it says so", "haven't learned" in m.hear("forget that correction"))
    check("a line that is not a correction is not its business", m.hear("what time is it") is None)


def m3_runtime():
    print("\n=== M3. In the runtime ===")
    import tst_modellast as tm
    rt = tm.rig()
    used, said = rt.say("I said Opera, not Obra")
    check("a correction is heard by rule, no model", not used and said and said[-1].startswith("Got it -- when I hear"), f"{used} {said}")
    check("and the microphone is given the fix and the hint", rt.misheard.fix("open obra") == "open Opera" and "Opera" in rt.vocab_hint())

    class Mic:
        def __init__(self):
            self.hint_words, self.corrector = None, None
    from src.senses.mic import MicrophoneSensor
    mic = MicrophoneSensor.__new__(MicrophoneSensor)
    mic.corrector = rt.misheard.fix
    check("the sensor has the hook", hasattr(MicrophoneSensor, "_finish") and mic.corrector("open obra") == "open Opera")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    m1_parse()
    m2_apply()
    m3_runtime()
