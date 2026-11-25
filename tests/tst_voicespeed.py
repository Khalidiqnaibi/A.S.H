"""
tst_voicespeed.py -- the voice follows the emotion state: a little slower when low (or in repair mode), a little brisker when keen.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_voicespeed.py
"""

import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.senses.actuators import SpeakActuator  # noqa: E402

check = tw.check


class Tts:
    engine = object()

    def __init__(self):
        self.seen = []

    def stream_audio(self, text, voice=None, **kw):
        self.seen.append(kw)
        yield b"RIFF0000WAVE"


def v1_actuator():
    print("\n=== V1. The speed reaches the engine, only when it is not the usual ===")
    tts = Tts()
    sp = SpeakActuator(tts_engine=tts, player=lambda wav, stop: None)
    sp.act("hello there")
    time.sleep(0.3)
    check("the usual speed passes nothing (an engine that does not take it is fine)", tts.seen == [{}], str(tts.seen))
    sp.speed = 0.92
    sp.act("hello again")
    time.sleep(0.3)
    check("a different speed is handed over", tts.seen[-1] == {"speed": 0.92}, str(tts.seen))


def v2_runtime():
    print("\n=== V2. The daemon sets it from the emotion state ===")
    import tst_effects as te
    h = te.rig(with_speaker=True)
    rt = getattr(h, "rt", h)
    sp = rt.periphery.actuators.get("speak")
    check("there is a speaker to set", sp is not None and hasattr(sp, "speed"))
    rt.brain.homeostasis = NS(last_emotion_result=NS(modulation=NS(energy=0.0)), drives=None)
    rt._voice_follows_state()
    check("low energy: slower", abs(sp.speed - 0.9) < 0.011, str(sp.speed))
    rt.brain.homeostasis = NS(last_emotion_result=NS(modulation=NS(energy=1.0)), drives=None)
    rt._voice_follows_state()
    check("keen: brisker, never silly", abs(sp.speed - 1.1) < 0.011, str(sp.speed))
    rt.repair.until = rt.clock() + 100
    rt._voice_follows_state()
    check("repair mode: slower again", sp.speed <= 0.93, str(sp.speed))
    rt.repair.end()
    rt.brain.homeostasis = NS(last_emotion_result=None, drives=None)
    rt._voice_follows_state()
    check("no reading yet: middling", 0.99 <= sp.speed <= 1.01, str(sp.speed))


if __name__ == "__main__":
    v1_actuator()
    v2_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
