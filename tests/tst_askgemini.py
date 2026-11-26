"""tst_askgemini.py -- "ask Gemini about X": the Gemini tab brought forward (or Gemini opened) and the question on the clipboard. Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_askgemini.py"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402
from src.senses.browsertabs import BrowserTabSensor  # noqa: E402

check = tw.check


class Clip:
    def __init__(self):
        self.copied = []

    def act(self, x):
        self.copied.append(x)
        return {"ok": True}


def rig(with_tab=True):
    rt = tm.rig()
    rt.desktop = td.drig()
    rt.desktop.ex_.procs.add("opera.exe")
    clip = Clip()
    bt = BrowserTabSensor(enabled=True)
    if with_tab:
        bt.submit({"focused": True, "tabs": [{"id": 7, "url": "https://gemini.google.com/app/x", "title": "Custom TPU Prototyping - Google Gemini",
                                              "active": False, "audible": False, "incognito": False}]})
        bt.read()
    real_s, real_a = rt.periphery.sensors, rt.periphery.actuators

    class S(dict):
        def get(self, k, d=None):
            return bt if k == "browser_tabs" else real_s.get(k, d)

    class A(dict):
        def get(self, k, d=None):
            return clip if k == "clipboard" else real_a.get(k, d)
    rt.periphery.sensors, rt.periphery.actuators = S(real_s), A(real_a)
    return rt, clip, bt


def main():
    print("\n=== G1. What is asked ===")
    G = AmbientRuntime._ASK_GEMINI
    for t, q in (("ask Gemini about the NPU on my laptop", "the NPU on my laptop"), ("Ash, ask Gemini whether my PC can run GTA 6", "my PC can run GTA 6"),
                 ("can you ask gemini what a TPU is", "what a TPU is")):
        m = G.match(t)
        check(f"{t!r} -> {q!r}", m is not None and m.group("q") == q, str(m and m.group("q")))
    check("'what is gemini' is not this", G.match("what is gemini") is None)

    print("\n=== G2. In the runtime ===")
    rt, clip, bt = rig()
    used, said = rt.say("ask Gemini about the NPU on my laptop")
    check("the question on the clipboard", clip.copied == ["The NPU on my laptop?"], str(clip.copied))
    check("the Gemini tab brought forward, said so, no model", not used and said and said[-1] == "Opening your Gemini tab -- the question is on your clipboard, paste it in."
          and bt.take_command() == {"activate": 7} and rt.desktop.ex_.focused == ["opera.exe"], f"{used} {said}")
    rt, clip, bt = rig(with_tab=False)
    used, said = rt.say("ask gemini what a TPU is")
    check("no Gemini tab: Gemini is opened", rt.desktop.ex_.opened == ["https://gemini.google.com/app"] and said[-1].startswith("Opening Gemini"), f"{rt.desktop.ex_.opened} {said}")
    rt, clip, bt = rig()
    rt.context.heard("what bio markers in the blood can help with early cancer detection")
    used, said = rt.say("ask Gemini about that")
    check("'about that' is the question just asked", clip.copied and clip.copied[-1].startswith("What bio markers in the blood"), str(clip.copied))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
