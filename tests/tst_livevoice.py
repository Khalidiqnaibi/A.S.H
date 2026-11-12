"""
tst_livevoice.py -- a live-feeling voice: talk over ASH, words ready sooner, first words out sooner, an optional
wake word. No audio device is touched: the microphone loop is fed synthetic frames, the transcriber and the TTS are
fakes, and the "speaker" only records what it was asked to play.

  L1  barge-in: a voice clearly louder than ASH's own echo, for a quarter second, stops it -- not a cough, not the echo
  L2  the wake gate: only what follows the wake word (or an open conversation) is transcribed
  L3  the microphone loop: early transcription is ready when the silence ends (one transcription, not two); talking on
      throws the early guess away; the wake gate; barge-in through the loop
  L4  speech: split where a listener hears a pause; the first chunk plays while the rest is still being made; "stop"
      cuts it off between chunks
  L5  wired up: build_periphery connects the microphone's barge-in to the speaker

Run:  PYTHONIOENCODING=utf-8 python tests/tst_livevoice.py
"""

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.senses.livevoice import BargeIn, WakeGate  # noqa: E402
from src.senses.mic import FRAME_SAMPLES, MicrophoneSensor, SAMPLE_RATE  # noqa: E402

check = tw.check


def l1_barge_in():
    print("\n=== L1. Talking over ASH ===")
    b = BargeIn()
    t = [0.0]

    def feed(rms, n, speaking=True):
        fired = 0
        for _ in range(n):
            t[0] += 0.03
            fired += b.feed(rms, speaking, t[0])
        return fired
    check("ASH's own voice, learned at the start of the line, never fires", feed(0.05, 12) == 0 and feed(0.06, 60) == 0)
    check("its louder moments (1.6x) don't either", feed(0.08, 30) == 0)
    check("a cough (three loud frames) doesn't", feed(0.25, 3) == 0 and feed(0.05, 5) == 0)
    check("a voice clearly louder for a quarter second does, once", feed(0.25, 12) == 1)
    check("not again straight away", feed(0.25, 12) == 0)
    check("nothing while ASH is quiet", feed(0.5, 30, speaking=False) == 0)
    b2 = BargeIn()
    t2 = [0.0]
    fired = 0
    for i in range(40):
        t2[0] += 0.03
        fired += b2.feed(0.3, True, t2[0])
    check("loud from the very start of the line is learned as the echo, not taken as the user", fired == 0)


def l2_wake_gate():
    print("\n=== L2. The wake gate ===")
    clock = [100.0]
    open_conv = [False]
    scores = [0.1]
    g = WakeGate(lambda f: scores[0], threshold=0.5, hold_s=4.0, conversation_open=lambda: open_conv[0], clock=lambda: clock[0])
    check("nothing heard: not transcribed", g.allows(1.5) is False and g.stats["skipped"] == 1)
    scores[0] = 0.9
    g.feed(None)
    clock[0] += 3
    check("the wake word just before: transcribed", g.allows(1.5))
    clock[0] += 10
    check("too long ago: not", g.allows(1.5) is False)
    open_conv[0] = True
    check("a conversation already open: transcribed without it", g.allows(1.5))
    scores[0] = None
    check("a detector that returns nothing is not a wake", g.feed(None) is False)


def tone(n_frames, amp=6000, hz=300):
    out = []
    for i in range(n_frames):
        t = (np.arange(FRAME_SAMPLES) + i * FRAME_SAMPLES) / SAMPLE_RATE
        out.append((amp * np.sin(2 * np.pi * hz * t)).astype(np.int16))
    return out


def quiet(n_frames):
    return [np.zeros(FRAME_SAMPLES, np.int16) for _ in range(n_frames)]


class Mic(MicrophoneSensor):
    """The real loop, fed prepared frames at real-time pace."""

    def __init__(self, frames, delay=0.0, words="what time is it ash", pace=0.03, **kw):
        super().__init__(enabled=False, transcriber=self._t, **kw)
        self._prepared, self.delay, self.words, self.pace = frames, delay, words, pace
        self.calls, self.last_speech_at, self.queued_at = [], None, None
        self._vad = None

    def _t(self, audio):
        self.calls.append(len(audio))
        time.sleep(self.delay)
        return self.words

    def _frames(self):
        for f in self._prepared:
            if np.abs(f).max() > 0:
                self.last_speech_at = time.time()
            time.sleep(self.pace)
            yield f

    def _finish(self, audio, text=None):
        super()._finish(audio, text=text)
        self.queued_at = time.time()


def l3_loop():
    print("\n=== L3. The microphone loop ===")
    import contextlib
    import io
    frames = quiet(5) + tone(30) + quiet(30)
    m = Mic(frames, delay=0.35)
    with contextlib.redirect_stderr(io.StringIO()):
        m._loop()
    lag_early = m.queued_at - m.last_speech_at
    check("one utterance queued, transcribed once (the early transcription was used)", len(m._out) == 1 and len(m.calls) == 1
          and m.live_stats["early_hits"] == 1, f"calls={len(m.calls)} {m.live_stats}")
    m2 = Mic(frames, delay=0.35, speculative=False)
    m2._loop()
    lag_late = m2.queued_at - m2.last_speech_at
    check(f"the words are ready sooner after you stop ({lag_early:.2f}s vs {lag_late:.2f}s without)",
          lag_early < lag_late - 0.2, f"{lag_early:.2f} {lag_late:.2f}")
    frames = quiet(5) + tone(20) + quiet(12) + tone(20) + quiet(30)
    m3 = Mic(frames, delay=0.1)
    m3._loop()
    check("talking on after a short pause throws the early guess away; the whole sentence is transcribed",
          len(m3._out) == 1 and m3.live_stats["early_wasted"] == 1 and m3.calls[-1] > m3.calls[0], f"{m3.calls} {m3.live_stats}")
    frames = quiet(5) + tone(30) + quiet(30)
    heard = [0.0]
    gate = WakeGate(lambda f: heard[0], threshold=0.5)
    m4 = Mic(frames, wake=gate)
    m4._loop()
    check("with a wake word and none heard: nothing transcribed or queued", m4.calls == [] and len(m4._out) == 0, f"{m4.calls}")
    heard[0] = 0.9
    m5 = Mic(frames, wake=WakeGate(lambda f: heard[0], threshold=0.5))
    m5._loop()
    check("the wake word heard: transcribed", len(m5._out) == 1)
    stopped = []
    m6 = Mic(quiet(5) + tone(14, amp=1500) + tone(20, amp=20000) + quiet(30), barge=BargeIn())
    m6.speaking_fn = lambda: True
    m6.on_barge_in = lambda: stopped.append(1)
    m6._loop()
    check("talking over ASH stops it, through the loop (before any transcription)", stopped == [1] and m6.live_stats["barge_ins"] == 1,
          f"{stopped} {m6.live_stats}")
    check("status says what is on", m6.status()["barge_in"] is True and m6.status()["wake_word"] is False)


def l4_speech():
    print("\n=== L4. Speaking sentence by sentence ===")
    from tools.tts import sentence_chunks
    check("short sentences stay together", sentence_chunks("Sure. It's 3 PM.") == ["Sure. It's 3 PM."])
    parts = sentence_chunks("First, open the file. Then, find the second paragraph and read it carefully. "
                            "After that, close it again and tell me what you found there. Done!")
    check("a longer reply splits where a pause is heard", len(parts) == 2 and parts[1].endswith("Done!"), str(parts))
    check("a very long sentence splits at a comma", all(len(p) <= 240 for p in sentence_chunks("word, " * 80)))
    from src.senses.actuators import SpeakActuator
    made, played = [], []

    class TTS:
        engine = True

        def stream_audio(self, text, voice=None):
            for i, part in enumerate(sentence_chunks(text)):
                time.sleep(0.3)                          # synthesis takes a while per chunk
                made.append((i, time.time()))
                yield f"wav{i}".encode()

    def player(wav, stop):
        played.append((wav, time.time()))
        stop.wait(0.4)                                   # playing takes a while too
    sp = SpeakActuator(tts_engine=TTS(), player=player, dry_run=False)
    sp.enable()
    text = ("This is the first sentence of a longer answer, long enough on its own. And here is a second one that "
            "keeps going for a while longer. And a third one, to be sure that the rest is still being made.")
    t0 = time.time()
    sp.act(text)
    time.sleep(0.45)
    first_play = played[0][1] - t0 if played else None
    check("the first chunk plays after one chunk is made, not all three", first_play is not None and first_play < 0.5
          and len(made) < 3, f"first play at {first_play}, made {len(made)}")
    deadline = time.time() + 5
    while sp.speaking and time.time() < deadline:
        time.sleep(0.05)
    check("all of it is played, in order", [w for w, _ in played] == [b"wav0", b"wav1", b"wav2"], str(played))
    played.clear()
    made.clear()
    sp.act(text)
    time.sleep(0.4)
    sp.stop()
    time.sleep(1.2)
    check("'stop' cuts it off: no further chunk plays", len(played) <= 1 and not sp.speaking, str(played))


def l5_wired():
    print("\n=== L5. Wired up ===")
    from src.daemon.service import DEFAULT_CONFIG
    from src.senses import build_periphery
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    reg = build_periphery(cfg, tts_engine=None)
    mic, sp = reg.sensors["microphone"], reg.actuators["speak"]
    check("the microphone's barge-in stops the speaker", mic.on_barge_in == sp.stop and isinstance(mic.barge, BargeIn))
    check("early transcription on, no wake word by default", mic.speculative is True and mic.wake is None)
    check("nothing is switched on by building it", mic.state.value != "active")
    cfg["sensors"]["microphone"]["barge_in"] = False
    cfg["sensors"]["microphone"]["wake_word_model"] = "C:/nowhere/hey_ash.onnx"
    reg2 = build_periphery(cfg, tts_engine=None)
    m2 = reg2.sensors["microphone"]
    check("barge-in can be switched off; a wake word model that can't load means no wake word (not deafness)",
          m2.barge is None and m2.wake is None)


if __name__ == "__main__":
    for fn in (l1_barge_in, l2_wake_gate, l3_loop, l4_speech, l5_wired):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
