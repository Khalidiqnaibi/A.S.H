"""
tst_sensorrecovery.py -- a sensor that gave up comes back, a microphone that died listens again, and "off" stays off.

Review found: after five failed reads a sensor went to ERROR and Sensor.due() flipped it back to ACTIVE after two minutes -- but
nothing restarted the microphone's own thread, so it showed ACTIVE and heard nothing for good; the restart counter was never
reset, so eight failures in a lifetime ended it; "off" said from another thread while a read was failing was overwritten by ERROR
and revived; "on" straight after "off" found the old capture thread still finishing a phrase, returned, and the thread then
ended; and two retries could start two capture threads.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_sensorrecovery.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import numpy as np  # noqa: E402

import tst_wondering as tw  # noqa: E402
from src.senses.base import Capability, Sensor, SensorEvent, State  # noqa: E402

check = tw.check


class Flaky(Sensor):
    capability = Capability(name="flaky")
    default_interval = 2.0
    fail = True
    retried = 0

    def read(self):
        if self.fail:
            raise OSError("busy")
        return [SensorEvent(source="flaky", text="ok")]

    def on_retry(self):
        self.retried += 1


def t1_base():
    print("\n=== The base sensor ===")
    s = Flaky(enabled=True)
    for _ in range(5):
        s.poll()
    check("five failures: resting", s.state == State.ERROR)
    s.due(s.errored_at + s.RETRY_AFTER_S + 1)
    check("tried again after a while, and the sensor was told (so it can restart its thread)", s.state == State.ACTIVE
          and s.retried == 1)

    s3 = Flaky(enabled=True)
    s3.error_count = 4
    s3.state = State.DISABLED
    s3.poll()
    check("a sensor the user switched off is not turned into ERROR by a failing read, so it is not revived",
          s3.state == State.DISABLED and not s3.due(time.time() + 10_000))


class Mic:
    pass


def make_mic(frames):
    from src.senses.mic import MicrophoneSensor, FRAME_SAMPLES

    class M(MicrophoneSensor):
        def probe(self):
            return True, ""

        def _frames(self):
            for f in frames():
                yield f
    return M(enabled=True)


def t2_mic():
    print("\n=== The microphone ===")
    from src.senses.mic import FRAME_SAMPLES
    quiet = np.zeros(FRAME_SAMPLES, dtype=np.int16)

    # frames flow, then fail once, then flow: the count restarts from zero
    calls = {"n": 0}
    done = threading.Event()

    def frames():
        calls["n"] += 1
        if calls["n"] == 1:
            yield quiet
            raise OSError("headset unplugged")
        for _ in range(5):
            yield quiet
        done.set()
        while True:
            time.sleep(0.01)
            yield quiet
    m = make_mic(frames)
    m.restarts = 3
    m._start_capture()
    time.sleep(0.3)
    check("it died once and is waiting to restart (the count was 3 + 1)", m.restarts >= 1, str(m.restarts))
    m._stop.set()
    m.state = State.ACTIVE
    m._thread = None
    m._stop.clear()
    m._start_capture()
    check("a restart brings frames back", done.wait(3))
    time.sleep(0.1)
    check("...and once frames flow the restart count is back to zero (eight in a lifetime used to end it)", m.restarts == 0,
          str(m.restarts))
    m._stop.set()
    m._thread.join(2)

    # on_retry restarts the capture thread
    started = []
    m2 = make_mic(lambda: iter([]))
    m2._start_capture = lambda: started.append(1)
    m2.restarts = 5
    m2.on_retry()
    check("a sensor given another go restarts its capture thread, and the count", started == [1] and m2.restarts == 0)

    # "on" straight after "off", while the old thread is finishing a phrase
    gate = threading.Event()

    def slow_frames():
        yield quiet
        gate.wait(10)                                   # busy "transcribing" when the stop arrives
        while not m3._stop.is_set():
            time.sleep(0.01)
            yield quiet
    m3 = make_mic(slow_frames)
    m3._start_capture()
    time.sleep(0.1)
    old = m3._thread
    m3._stop.set()                                      # "off"
    threading.Timer(0.3, gate.set).start()              # the phrase finishes a moment later
    m3._start_capture()                                 # "on", straight away
    check("'on' right after 'off' waits for the old thread and then really starts a new one",
          m3._thread is not old and m3._thread.is_alive() and not m3._stop.is_set(), f"{m3._thread} {old}")
    m3._stop.set()
    m3._thread.join(2)

    # two retries do not make two threads
    m4 = make_mic(lambda: iter([quiet] * 3 + [None]))
    alive = threading.Thread(target=lambda: time.sleep(1), daemon=True)
    alive.start()
    m4._thread = alive
    before = m4._thread
    m4._start_capture()
    check("a capture thread that is already running is left alone", m4._thread is before)


if __name__ == "__main__":
    t1_base()
    t2_mic()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
