"""
src/senses/mic.py

Always-on listening.

The design constraint that shapes everything here: a 24h microphone that
transcribes continuously is both expensive and creepy. So the pipeline is
three gates deep, and audio only advances when each one passes:

    1. LEVEL GATE   -- RMS above the adaptive noise floor. Free. Rejects ~95%
                       of a quiet room's samples without touching a model.
    2. SPEECH GATE  -- VAD (webrtcvad if present, energy+zero-crossing
                       heuristic otherwise) confirms it is voice, not a fan or
                       a door. Cheap.
    3. TRANSCRIBE   -- only complete utterances, bounded by silence on both
                       sides, ever reach the recognizer. Expensive.

Raw audio is never written to disk. A rolling in-memory ring buffer holds the
last few seconds so the pre-roll before speech onset isn't clipped, and it is
overwritten continuously. What persists is text.

Direct address
--------------
`ADDRESS_PATTERNS` is what separates "the user said something near ASH" from
"the user said something *to* ASH". A transcript matching one of these sets
`addressed=True`, which the attention gate treats as an unconditional
override -- being spoken to always earns a reply, regardless of drives,
refractory period, or how busy the user looks.

Everything else is heard, remembered, and usually not answered.
"""

from __future__ import annotations

import collections
import logging
import math
import re
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base import Capability, Modality, Sensor, SensorEvent

logger = logging.getLogger("ash.senses.mic")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

WAKE_WORDS = ["ash", "hey ash", "ok ash", "yo ash"]

# A transcript is treated as directly addressed if it matches any of these.
ADDRESS_PATTERNS = [
    re.compile(r"\b(hey |ok |okay |yo )?ash\b", re.I),
    re.compile(r"^\s*(can you|could you|would you|please)\b", re.I),
]


class MicrophoneSensor(Sensor):
    """Continuous capture with VAD-bounded utterance segmentation."""

    capability = Capability(
        name="microphone",
        description="Continuous ambient listening with speech detection and transcription.",
        requires=["numpy"],
        privacy_sensitive=True,
    )
    default_interval = 0.5      # how often the ambient loop drains the queue
    modality = Modality.TEXT

    def __init__(self, interval: Optional[float] = None, enabled: bool = False,
                 device: Optional[int] = None,
                 silence_ms: int = 700, min_speech_ms: int = 400,
                 max_utterance_s: float = 20.0,
                 transcriber=None, keep_audio: bool = False):
        self.device = device
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 // FRAME_MS)
        self.keep_audio = keep_audio
        self._transcriber = transcriber

        self._backend: Optional[str] = None
        self._vad = None
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._out: Deque[SensorEvent] = collections.deque(maxlen=64)
        self._lock = threading.Lock()

        # Adaptive noise floor -- a fixed RMS threshold works in one room and
        # fails in every other one.
        self._noise_floor = 0.01
        self._speech_seen = 0

        super().__init__(interval=interval, enabled=enabled)

    # ------------------------------------------------------------------
    def probe(self) -> Tuple[bool, str]:
        try:
            import sounddevice  # noqa: F401
            self._backend = "sounddevice"
        except Exception:
            try:
                import pyaudio  # noqa: F401
                self._backend = "pyaudio"
            except Exception as e:
                return False, f"no audio backend ({e}); pip install sounddevice"
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(2)
            logger.info("Microphone: using webrtcvad")
        except Exception:
            self._vad = None
            logger.info("Microphone: webrtcvad unavailable, using energy VAD")
        return True, ""

    def enable(self):
        super().enable()
        self._start_capture()

    def disable(self):
        super().disable()
        self._stop_capture()

    # ------------------------------------------------------------------
    # Capture thread
    # ------------------------------------------------------------------
    def _start_capture(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ash-mic")
        self._thread.start()
        logger.info("Microphone capture started (%s)", self._backend)

    def _stop_capture(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _frames(self):
        """Yield fixed-size int16 frames from whichever backend is present."""
        import numpy as np

        if self._backend == "sounddevice":
            import sounddevice as sd
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                blocksize=FRAME_SAMPLES, device=self.device) as stream:
                while not self._stop.is_set():
                    data, _ = stream.read(FRAME_SAMPLES)
                    yield np.asarray(data, dtype=np.int16).flatten()
        else:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                             input=True, frames_per_buffer=FRAME_SAMPLES,
                             input_device_index=self.device)
            try:
                while not self._stop.is_set():
                    raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                    yield np.frombuffer(raw, dtype=np.int16)
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

    def _is_speech(self, frame) -> bool:
        import numpy as np

        rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)) + 1e-9)

        # Track the noise floor from quiet frames only, so a long conversation
        # doesn't drag the threshold up until nothing registers as speech.
        if rms < self._noise_floor * 2.0:
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms

        if rms < max(0.006, self._noise_floor * 3.0):
            return False

        if self._vad is not None:
            try:
                return self._vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            except Exception:
                pass

        # Energy + zero-crossing fallback. Voice has moderate ZCR; hiss and
        # clicks sit at the extremes.
        zc = float(np.mean(np.abs(np.diff(np.sign(frame.astype(np.float32)))) > 0))
        return 0.02 < zc < 0.35

    def _loop(self):
        import numpy as np

        preroll: Deque = collections.deque(maxlen=int(400 // FRAME_MS))
        buf: List = []
        silence = 0
        speaking = False

        try:
            for frame in self._frames():
                speech = self._is_speech(frame)

                if not speaking:
                    preroll.append(frame)
                    if speech:
                        speaking = True
                        buf = list(preroll)   # keep the pre-onset audio
                        silence = 0
                    continue

                buf.append(frame)
                silence = 0 if speech else silence + 1

                too_long = len(buf) >= self.max_frames
                if silence >= self.silence_frames or too_long:
                    speaking = False
                    if len(buf) >= self.min_speech_frames:
                        self._finish(np.concatenate(buf))
                    buf, preroll = [], collections.deque(maxlen=int(400 // FRAME_MS))
        except Exception:
            logger.exception("Microphone loop died")
            self.state = self.state.__class__.ERROR

    # ------------------------------------------------------------------
    def _finish(self, audio):
        """One complete utterance: transcribe and queue it."""
        duration = len(audio) / SAMPLE_RATE
        text = self.transcribe(audio)
        if not text:
            return

        addressed = self.is_addressed(text)
        self._speech_seen += 1

        event = SensorEvent(
            source=self.name, modality=Modality.TEXT,
            text=text,
            data=(audio if self.keep_audio else None),
            salience=0.85 if addressed else 0.55,
            urgency=0.6 if addressed else 0.0,
            addressed=addressed,
            meta={"duration_s": round(duration, 2), "addressed": addressed},
        )
        with self._lock:
            self._out.append(event)

    @staticmethod
    def is_addressed(text: str) -> bool:
        return any(p.search(text or "") for p in ADDRESS_PATTERNS)

    def transcribe(self, audio) -> str:
        """Pluggable ASR. Whisper if installed, then Google via
        speech_recognition, then nothing. A custom callable can be injected
        for a local model."""
        if self._transcriber is not None:
            try:
                return (self._transcriber(audio) or "").strip()
            except Exception:
                logger.exception("Custom transcriber failed")
                return ""

        try:
            import numpy as np
            from faster_whisper import WhisperModel  # type: ignore

            if not hasattr(self, "_whisper"):
                self._whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
            segs, _ = self._whisper.transcribe(audio.astype(np.float32) / 32768.0,
                                               language="en", vad_filter=True)
            return " ".join(s.text for s in segs).strip()
        except Exception:
            pass

        try:
            import speech_recognition as sr

            rec = sr.Recognizer()
            data = sr.AudioData(audio.tobytes(), SAMPLE_RATE, 2)
            return rec.recognize_google(data)
        except Exception:
            logger.debug("Transcription unavailable", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    def read(self) -> List[SensorEvent]:
        with self._lock:
            out = list(self._out)
            self._out.clear()
        return out

    def close(self):
        self._stop_capture()

    def status(self) -> Dict[str, Any]:
        d = super().status()
        d.update({
            "backend": self._backend,
            "vad": "webrtcvad" if self._vad else "energy",
            "utterances": self._speech_seen,
            "noise_floor": round(self._noise_floor, 5),
            "queued": len(self._out),
        })
        return d
