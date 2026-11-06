"""
tst_echo.py -- ASH not hearing itself.

The lines are from the session of 2026-11-06, where the microphone picked
ASH's own speech up and ASH answered it.

  E1  Its own words, heard while or just after it spoke, are dropped --
      whisper's mishearings included
  E2  The user is still heard: talking over it, a different sentence, the
      same topic in other words, and anything long after
  E3  The speaker tells the guard what it says and when it stops
  E4  The microphone drops the echo before anything else sees it

Run:  PYTHONIOENCODING=utf-8 python tst_echo.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.senses.echo import EchoGuard  # noqa: E402

check = tw.check

SAID1 = ("Ash, what do you understand by the question \"what your step is\"? Are you being asked about the next "
         "action in the integration process, or something else?")
SAID2 = ("My current status is warm and upbeat. The reason for this emotional state is due to the advanced emotional "
         "emotional simulation that Immortal implemented recently. How can I assist you today, Immortal?")


def rig():
    t = [1000.0]
    g = EchoGuard(clock=lambda: t[0])
    g.t = t
    return g


def e1_own_words():
    print("\n=== E1. Its own words ===")
    g = rig()
    g.said(SAID1, start=1000.0, seconds=9.0)
    check("heard while speaking (a mishearing)", g.is_echo("Your step is...", heard_at=1004.0, duration=1.5))
    check("heard just after, transcribed late",
          g.is_echo("Are you to action in the integration process or something else?", heard_at=1012.0, duration=3.9))
    g = rig()
    g.said(SAID2, start=2000.0, seconds=12.0)
    for text, at, dur in (("My current status is warm and upbeat.", 2003.0, 3.0),
                          ("Emotional state is due to the advanced emotional-emotional simulation that immortal "
                           "implemented recently.", 2010.0, 7.6),
                          ("How today? Immortal.", 2013.0, 2.0)):
        check(f"{text[:40]!r}", g.is_echo(text, heard_at=at, duration=dur))
    check("counted", g.dropped == 3)


def e2_user_heard():
    print("\n=== E2. The user is still heard ===")
    g = rig()
    g.said(SAID2, start=2000.0, seconds=12.0)
    for text in ("stop", "Ash, stop.", "no, the other one", "what's your CPU status",
                 "it seems that you are picking up your own speech as my own speech so you have to fix that"):
        check(f"talking over it: {text!r}", not g.is_echo(text, heard_at=2005.0, duration=1.5))
    check("sharing only little words with it is not an echo", not g.is_echo("the state is for you to fix",
                                                                             heard_at=2005.0, duration=1.5))
    check("its words, long after: the user quoting it", not g.is_echo("my current status is warm and upbeat",
                                                                      heard_at=2100.0, duration=3.0))
    check("a near-exact repeat a little late (slow transcription) still is",
          g.is_echo("My current status is warm and upbeat", heard_at=2030.0, duration=3.0))
    check("but a short one that late is not", not g.is_echo("Immortal", heard_at=2030.0, duration=1.0))
    check("nothing said: nothing is an echo", not rig().is_echo("hello there", heard_at=1000.0))
    check("empty: no", not g.is_echo("", heard_at=2005.0))
    check("before ASH began: not an echo", not g.is_echo("my current status is warm", heard_at=1990.0, duration=2.0))


def e3_speaker():
    print("\n=== E3. The speaker tells it ===")
    from src.senses import echo
    from src.senses.actuators import SpeakActuator

    class Tts:
        engine = True

        def stream_audio(self, text, voice=None):
            yield b"RIFF"

    old = echo.guard
    echo.guard = EchoGuard()
    try:
        sp = SpeakActuator(tts_engine=Tts(), player=lambda w, s: s.wait(5), dry_run=False)
        sp.enable()
        sp.act("My current status is warm and upbeat.")
        time.sleep(0.05)
        check("said: the guard knows", echo.guard.is_echo("my current status is warm", heard_at=time.time(), duration=1))
        sp.stop()
        end = list(echo.guard._said)[-1]["end"]
        check("cut off: the end moved to now", abs(end - time.time()) < 1.0)
        check("the user right after is still heard", not echo.guard.is_echo("what time is it", heard_at=time.time()))
    finally:
        echo.guard = old


def e4_mic():
    print("\n=== E4. The microphone ===")
    import numpy as np
    from src.senses import echo
    from src.senses.mic import MicrophoneSensor
    old = echo.guard
    echo.guard = EchoGuard()
    try:
        m = MicrophoneSensor.__new__(MicrophoneSensor)
        m.voiceprint, m.keep_audio, m._speech_seen = False, False, 0
        import threading
        m._lock, m._out = threading.Lock(), []
        heard = ["My current status is warm and upbeat."]
        m.transcribe = lambda audio: heard[0]
        echo.guard.said(SAID2)
        m._finish(np.zeros(16000 * 2, dtype=np.int16))
        check("its own voice: no event", m._out == [])
        heard[0] = "Ash, what's your CPU status?"
        m._finish(np.zeros(16000 * 2, dtype=np.int16))
        check("the user: an event", len(m._out) == 1 and m._out[0].text == "Ash, what's your CPU status?")
    finally:
        echo.guard = old


if __name__ == "__main__":
    for fn in (e1_own_words, e2_user_heard, e3_speaker, e4_mic):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
