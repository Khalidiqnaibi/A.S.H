"""tst_secondpass.py -- a phrase the fast model is unsure of is read again by a bigger one already on disk; never a download.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_secondpass.py"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.senses import mic as M

OK = True
def check(n, c, d=""):
    global OK
    print(("  ok   " if c else "  FAIL ") + n + ("" if c else f"  {d}"))
    OK = OK and bool(c)

def seg(t, lp):
    return types.SimpleNamespace(text=t, avg_logprob=lp, no_speech_prob=0.0)

U = M.MicrophoneSensor._unsure
check("a sure phrase is left alone", not U([seg("open notepad", -0.2)]))
check("an unsure phrase gets a second look", U([seg("open nodepad", -0.7)]))
check("the average counts, not one phrase", not U([seg("a", -0.1), seg("b", -0.8)]))
check("no confidence given: no second look", not U([types.SimpleNamespace(text="hi")]) and not U([]))
m = M.MicrophoneSensor.__new__(M.MicrophoneSensor)
m.second_pass_model = "small.en"
m.prompt_hint = lambda: ""
import numpy as np
class Boom:
    def __init__(self, *a, **k):
        Boom.calls.append(k)
        raise OSError("not on disk")
Boom.calls = []
sys.modules["faster_whisper"] = types.SimpleNamespace(WhisperModel=Boom)
a = np.zeros(1600, dtype=np.int16)
check("a model that is not on disk: the first reading is kept", m._second_pass(a) == "")
check("it is asked for with local_files_only (never a download)", bool(Boom.calls) and Boom.calls[0].get("local_files_only") is True, str(Boom.calls))
m._second_pass(a)
check("and given up for good, not tried again", len(Boom.calls) == 1, str(len(Boom.calls)))
print("\n" + ("ALL PASS" if OK else "FAILED"))
sys.exit(0 if OK else 1)
