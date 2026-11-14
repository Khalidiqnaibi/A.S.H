"""
tst_speakleak.py -- cutting ASH off mid-sentence must not leak a thread.

SpeakActuator synthesises the next sentence on a thread while the current one plays, through a queue of two. When the line was
stopped ("stop", "never mind", a new line replacing it) the player left, and the synthesising thread -- blocked in put() on the
full queue, including its final put of the end marker -- never woke: one leaked thread and two or three synthesised sentences per
interrupted long answer, for as long as ASH runs.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_speakleak.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.senses.actuators import SpeakActuator  # noqa: E402

check = tw.check


class Tts:
    engine = True

    def __init__(self, n=12):
        self.n = n
        self.made = 0

    def stream_audio(self, text, voice=None):
        for i in range(self.n):
            self.made += 1
            yield b"RIFF" + bytes(8)


def makers():
    return [t for t in threading.enumerate() if t.name == "ash-tts-make" and t.is_alive()]


def t1_stop_midway():
    print("\n=== Stopped mid-sentence ===")
    tts = Tts()
    played = []
    started = threading.Event()

    def player(wav, stop):
        played.append(1)
        started.set()
        stop.wait(10)                                    # "playing" until it is cut off
    sp = SpeakActuator(tts_engine=tts, player=player, dry_run=False)
    sp.enable()
    sp.act("one. two. three. four. five. six. seven. eight.")
    check("(it started playing)", started.wait(3))
    time.sleep(0.3)
    check("(the synthesiser is waiting with a full queue)", len(makers()) == 1, str(makers()))
    sp.stop()
    deadline = time.time() + 3
    while makers() and time.time() < deadline:
        time.sleep(0.05)
    check("stopped: the synthesising thread ends (it used to wait on the full queue for ever)", not makers(), str(makers()))
    check("...and it stopped making more sentences", tts.made < 12, str(tts.made))

    for _ in range(5):                                   # five interrupted answers in a row
        started.clear()
        sp.act("a. b. c. d. e. f. g. h.")
        started.wait(3)
        time.sleep(0.2)
        sp.stop()
    deadline = time.time() + 4
    while makers() and time.time() < deadline:
        time.sleep(0.05)
    check("five interrupted answers leave no threads behind", not makers(), str(makers()))


def t2_replaced():
    print("\n=== Replaced by a newer line ===")
    tts = Tts()
    gate = threading.Event()

    def player(wav, stop):
        stop.wait(10)
    sp = SpeakActuator(tts_engine=tts, player=player, dry_run=False)
    sp.enable()
    sp.act("one. two. three. four. five. six.")
    time.sleep(0.3)
    sp.act("a new line")
    deadline = time.time() + 4
    while len(makers()) > 1 and time.time() < deadline:
        time.sleep(0.05)
    check("a new line replacing the old one does not leave the old synthesiser behind", len(makers()) <= 1, str(makers()))
    sp.stop()


def t3_whole_line():
    print("\n=== A line that plays out ===")
    tts = Tts(n=5)
    played = []
    sp = SpeakActuator(tts_engine=tts, player=lambda wav, stop: played.append(1), dry_run=False)
    sp.enable()
    sp.act("whole line")
    deadline = time.time() + 3
    while sp.speaking and time.time() < deadline:
        time.sleep(0.05)
    check("every sentence is played when nobody stops it", len(played) == 5, str(len(played)))
    check("...and nothing is left running", not makers() and not sp.speaking)


if __name__ == "__main__":
    t1_stop_midway()
    t2_replaced()
    t3_whole_line()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
