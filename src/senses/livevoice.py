"""
src/senses/livevoice.py

What makes talking to ASH feel live instead of walkie-talkie:

  BargeIn     you talk over ASH and it stops at once -- from the sound level alone, before a word is transcribed.
              While ASH speaks, the microphone hears ASH too (no echo cancellation here), so the first moments of
              each line teach it how loud its own voice is in this room; only a voice clearly louder than that, for
              a quarter of a second, counts as you. A cough is too short; ASH's own echo is not loud enough.
  WakeGate    an optional on-device wake word (openWakeWord, if installed and given a model): when it is on, an
              utterance is only transcribed if the wake word was heard just before or during it, or if a
              conversation with ASH is already open. Faster (no transcribing room chatter) and more private. Off by
              default -- the name in the transcript (address.py) does the job without it.

The third piece -- transcribing speculatively at the start of the end-of-turn silence, so the words are ready the
moment the silence is long enough -- lives in mic.py's loop, and the fourth -- speaking sentence by sentence while the
next sentence is synthesized -- in the speech actuator and tools/tts.py.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger("ash.senses.livevoice")


class BargeIn:
    """feed(rms, speaking, now) -> True once, when the user starts talking over ASH."""

    def __init__(self, learn_s: float = 0.35, ratio: float = 2.5, min_level: float = 0.02, hold_frames: int = 8,
                 cooldown_s: float = 1.0):
        self.learn_s, self.ratio, self.min_level = learn_s, ratio, min_level
        self.hold_frames, self.cooldown_s = hold_frames, cooldown_s
        self._started: Optional[float] = None
        self._echo: Deque[float] = collections.deque(maxlen=200)
        self._loud = 0
        self._fired_at = -1e9
        self.fired = 0

    def _echo_level(self) -> float:
        if not self._echo:
            return 0.0
        s = sorted(self._echo)
        return s[int(len(s) * 0.8)]                         # a loud-ish moment of ASH's own voice, not its peak

    def feed(self, rms: float, speaking: bool, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if not speaking:
            self._started, self._loud = None, 0
            return False
        if self._started is None:
            self._started = now
        if now - self._started < self.learn_s:
            self._echo.append(rms)                          # the start of each line: how loud ASH itself is here
            return False
        level = max(self.min_level, self._echo_level() * self.ratio)
        if rms >= level:
            self._loud += 1
        else:
            self._echo.append(rms)                          # quieter moments keep the picture of the echo current
            self._loud = 0
        if self._loud >= self.hold_frames and now - self._fired_at >= self.cooldown_s:
            self._loud = 0
            self._fired_at = now
            self.fired += 1
            return True
        return False


class WakeGate:
    """An on-device wake word in front of transcription. `detector(frame) -> score 0..1` (openWakeWord, or a fake)."""

    def __init__(self, detector: Callable[[Any], float], threshold: float = 0.5, hold_s: float = 4.0,
                 conversation_open: Callable[[], bool] = lambda: False, clock: Callable[[], float] = time.time):
        self.detector = detector
        self.threshold = threshold
        self.hold_s = hold_s
        self.conversation_open = conversation_open
        self.clock = clock
        self.heard_at = -1e9
        self.stats = {"woke": 0, "skipped": 0}

    def feed(self, frame) -> bool:
        try:
            score = float(self.detector(frame) or 0.0)
        except Exception:
            return False
        if score >= self.threshold:
            if self.clock() - self.heard_at > 1.0:
                self.stats["woke"] += 1
            self.heard_at = self.clock()
            return True
        return False

    def allows(self, utterance_s: float) -> bool:
        """Transcribe this utterance? Yes if the wake word came during it or just before, or a conversation is open."""
        now = self.clock()
        if now - self.heard_at <= self.hold_s + utterance_s or self.conversation_open():
            return True
        self.stats["skipped"] += 1
        return False


def openwakeword_detector(model_path: str, frame_samples: int = 480):
    """A detector from an openWakeWord model file (e.g. one trained for "hey ash"). None if openWakeWord is missing.
    openWakeWord wants 80 ms chunks (1280 samples at 16 kHz); the microphone's 30 ms frames are gathered up."""
    try:
        import numpy as np
        from openwakeword.model import Model                     # type: ignore
    except Exception:
        logger.info("openWakeWord is not installed; no wake word (pip install openwakeword)")
        return None
    try:
        model = Model(wakeword_models=[model_path], inference_framework="onnx")
    except Exception:
        logger.exception("Could not load the wake word model %s", model_path)
        return None
    buf: list = []
    need = 1280

    def detect(frame) -> float:
        buf.extend(np.asarray(frame, dtype=np.int16).tolist())
        best = 0.0
        while len(buf) >= need:
            chunk = np.asarray(buf[:need], dtype=np.int16)
            del buf[:need]
            scores = model.predict(chunk)
            best = max([best] + [float(v) for v in scores.values()])
        return best
    return detect
