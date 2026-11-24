"""tst_noisefloor.py -- the microphone's noise floor follows a room that got louder, not only one that got quieter.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_noisefloor.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.senses.mic import MicrophoneSensor  # noqa: E402

OK = True


def check(n, c, d=""):
    global OK
    print(("  ok   " if c else "  FAIL ") + n + ("" if c else f"  {d}"))
    OK = OK and bool(c)


def mic(floor=0.002):
    m = MicrophoneSensor.__new__(MicrophoneSensor)
    m._noise_floor = floor
    return m


m = mic()
for _ in range(900):
    m._rise_floor(0.02)
check("not before a minute's worth of frames", m._noise_floor == 0.002, str(m._noise_floor))
for _ in range(3000):                                  # a fan came on: 10x the quiet hour's level, all the time
    m._rise_floor(0.02)
check("a room that got louder raises the floor", m._noise_floor > 0.01, str(m._noise_floor))
check("gradually, never past what the room is doing", m._noise_floor <= 0.02 + 1e-9, str(m._noise_floor))

m = mic(0.01)
for _ in range(4000):                                  # someone talking most of the time, with pauses: the pauses set the floor
    m._rise_floor(0.012 if _ % 5 == 0 else 0.15)
check("talk with pauses does not drag it up to the talk", m._noise_floor < 0.03, str(m._noise_floor))

m = mic(0.01)
for _ in range(4000):
    m._rise_floor(0.8)
check("never past 0.05", m._noise_floor <= 0.05, str(m._noise_floor))

m = mic(0.01)
for _ in range(4000):
    m._rise_floor(0.004)
check("a quieter room does not move it here (the old rule lowers it)", m._noise_floor == 0.01, str(m._noise_floor))
print("\n" + ("ALL PASS" if OK else "FAILED"))
sys.exit(0 if OK else 1)
