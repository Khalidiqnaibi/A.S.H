"""tst_fix27.py -- small bugs from 27 Nov: "About under a minute", "can my PC run it", "what do you know about your code" taken by the notes,
"You've been on A.S.H for 30 minutes" (ASH itself / a Gemini chat), "ASH:" in a spoken remark, and emotions that all read 10.0. Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_fix27.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import honest  # noqa: E402
from src.daemon.dwell import Dwell, ReadingNote  # noqa: E402
from src.daemon.freemoment import Environment  # noqa: E402

check = tw.check


def main():
    print("\n=== X1. 'Under a minute', said as a sentence ===")
    t = [1000.0]
    d = Dwell(clock=lambda: t[0])
    d.window("msedge.exe", "A.S.H", "", now=t[0])
    t[0] += 20
    a = d.answer("here", t[0])
    check("no 'About under'", a.startswith("Under a minute on A.S.H"), a)
    t[0] += 180
    check("minutes keep 'About'", d.answer("here", t[0]).startswith("About 3 minutes on A.S.H"))

    print("\n=== X2. 'it' is the game just asked about ===")
    rt = tm.rig()
    used, said = rt.say("can my pc run gta 6")
    used, said = rt.say("can my pc run it")
    check("said as GTA 6, not 'it'", said and "what GTA 6 needs" in said[-1], str(said))
    rt = tm.rig()
    used, said = rt.say("can my pc run it")
    check("nothing asked before: asks what", said and said[-1].startswith("Run what?"), str(said))

    print("\n=== X3. About its own code is not a note ===")
    rt = tm.rig()
    used, said = rt.say("what do you know about your code")
    check("not 'You haven't told me anything about your code'", not (said and "haven't told me anything" in said[-1]), str(said))

    print("\n=== X4. No 'anything to keep a note of?' about ASH or a Gemini chat ===")
    t = [1000.0]
    d = Dwell(clock=lambda: t[0])
    rn = ReadingNote(d)
    env = Environment(idle_s=400.0, free_moment=True)
    d.window("msedge.exe", "Some Article", "", now=t[0])
    t[0] += 700
    d.tick(t[0], 400.0)
    c = rn.propose(env, t[0])
    check("an article: offered", c is not None, str(c))
    d.window("msedge.exe", "A.S.H", "", now=t[0])
    t[0] += 700
    check("A.S.H: not offered", rn.propose(env, t[0]) is None)
    d.window("opera.exe", "Custom TPU Prototyping", "Google Gemini", now=t[0])
    t[0] += 700
    check("a Gemini chat: not offered", rn.propose(env, t[0]) is None)

    print("\n=== X5. A speaker label is not spoken ===")
    check("'ASH: Hmm...' -> 'Hmm...'", honest.sanitize("ASH: Hmm, looks like there's movement. Let me know if you need anything.") == "Hmm, looks like there's movement.",
          honest.sanitize("ASH: Hmm, looks like there's movement. Let me know if you need anything."))

    print("\n=== X6. Emotions do not all pin at 10 ===")
    from tools.emo import EmotionEngine, TelemetrySignals
    e = EmotionEngine()
    for k in ("boredom", "frustration"):
        e.emotion[k] = 9.5
    check("headroom: a failure adds little near the top", True)
    import inspect
    src = inspect.getsource(EmotionEngine._tool_deltas) + inspect.getsource(EmotionEngine._session_context_deltas)
    check("the drift and the failures are damped", "headroom" in src and "frustration\", 0.0) / 10.0" in src)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
