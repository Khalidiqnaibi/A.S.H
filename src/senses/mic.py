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
What separates "the user said something near ASH" from "the user said
something *to* ASH" lives in src/senses/address.py, which weighs the name,
second person, question form, imperatives and whether a conversation is
already running. It answers with a confidence, not a yes/no, and this module
turns that into the event's `addressed` flag, its salience and its urgency.

Certainly addressed is an unconditional override at the attention gate --
being spoken to always earns a reply, regardless of drives, refractory period
or how busy the user looks. Probably addressed is carried in the event meta
and handled by the gate's route 1b.

Everything else is heard, remembered, and usually not answered.
"""

from __future__ import annotations

import collections
import logging
import math
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from . import address
from .base import Capability, Modality, Sensor, SensorEvent, State

logger = logging.getLogger("ash.senses.mic")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

WAKE_WORDS = ["ash", "hey ash", "ok ash", "yo ash"]

# The same name and openers address.py recognises: "Hey, Ash, ..." (whisper puts a comma after "hey"), "hi ash", "hello ash", "A.S.H.".
_ASH = r"(?:ash|a\.\s?s\.\s?h\.?)"
_LEADING_WAKE = re.compile(r"^\s*(?:(?:hey|hi|hello|ok|okay|yo)[\s,]+)?" + _ASH + r"(?=[\s,.!?:;-]|$)[\s,.!?:;-]*", re.I)
_TRAILING_WAKE = re.compile(r"[\s,]*(?<![\w.])(?:(?:hey|hi|hello|ok|okay|yo)[\s,]+)?" + _ASH + r"[\s.!?]*$", re.I)


# faster-whisper scores every phrase it writes. A phrase it is unsure of is
# usually room noise or someone else's voice, or the TV: "Ash brothers
# emote", "Your step is..." were all answered as if they were requests, and
# ASH made up a clarification ("did you mean to ask if I have ever purchased
# anything?") for one. Below these it is better to say nothing.
SECOND_PASS_LOGPROB = -0.5   # a kept phrase averaging below this is read again by the bigger model
MIN_LOGPROB = -1.0        # average log-probability of the words (0 = certain)
NOISE_PROB = 0.5          # chance the phrase is not speech at all...
NOISE_LOGPROB = -0.6      # ...and it is not sure of the words either


# What whisper writes on silence or music, learned from its training subtitles: credits, URLs, "thanks for watching", a string of letters.
_JUNK = re.compile(r"(?:\b(?:subs?|subtitles?|captions?|translated|transcribed)\s+by\b|\bwww\.(?:[a-z0-9-]+\.)+(?:co\.uk|org|net|tv)\b|"
                   r"thanks?(?:\s+you)?(?:\s+so\s+much)?\s+for\s+(?:watching|listening)|like\s+and\s+subscribe|please\s+subscribe|amara\.org)", re.I)


def is_junk(text: str) -> bool:
    """Whisper's invented text: "Subs by www.zeoranger.co.uk", "Thanks for watching", or a long run of single letters
    ("pres-en-d-b-a-g-a-d-a-g-a-d-a..."): nothing a person said."""
    t = text or ""
    if _JUNK.search(t):
        return True
    w = re.findall(r"[A-Za-z0-9']+", t)
    return len(w) >= 12 and sum(1 for x in w if len(x) == 1) / len(w) >= 0.5


def is_loop(text: str) -> bool:
    """Whisper stuck in a loop, on noise: the same few words said again and again ("I'm not sure if I'm going to work with Ash,
    I'm not sure if I'm going to work with Ash, ..."). A person does not say one phrase three times running as a request, and
    the lines were answered as if they were."""
    w = re.findall(r"[a-z0-9']+", (text or "").lower())
    if len(w) < 6:
        return False
    for n in range(2, len(w) // 3 + 1):
        for i in range(0, len(w) - 3 * n + 1):
            unit = w[i:i + n]
            if w[i + n:i + 2 * n] == unit and w[i + 2 * n:i + 3 * n] == unit:
                run = 3 * n
                j = i + 3 * n
                while w[j:j + n] == unit:
                    run += n
                    j += n
                if run >= 0.6 * len(w):
                    return True
    return False


def trusted_text(segs) -> str:
    """The text of whisper's phrases, leaving out the ones it is unsure of."""
    keep = []
    for s in segs:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue
        lp = getattr(s, "avg_logprob", None)
        ns = getattr(s, "no_speech_prob", None)
        if lp is not None and lp < MIN_LOGPROB:
            logger.info("Mic: dropped a doubtful phrase (confidence %.2f): %r", lp, text)
            continue
        # (a one- or two-word line is more often a real "yeah" or "no" than
        # noise, and whisper is always unsure of those: "Yeah." scored -0.73)
        if (lp is not None and ns is not None and ns > NOISE_PROB and lp < NOISE_LOGPROB
                and (len(text.split()) >= 3 or lp < -0.9)):
            logger.info("Mic: dropped a phrase that is probably not speech (%.2f, %.2f): %r", ns, lp, text)
            continue
        if is_loop(text) or is_junk(text):
            logger.info("Mic: dropped a phrase that is whisper repeating itself or inventing text: %r", text[:100])
            continue
        keep.append(text)
    out = " ".join(keep).strip()
    if is_loop(out):                                 # the loop can run across phrases
        logger.info("Mic: dropped a transcript that is whisper repeating itself: %r", out[:100])
        return ""
    return out


_MISHEARD = re.compile(r"^(\s*(?:hey|hi|ok|okay)?[\s,]*)(?:ass|asch|ashe)(?=[\s,]+\w)", re.I)


_MISHEARD_TAIL = re.compile(
    r"((?:what'?s\s+up|how\s+are\s+you(?:\s+doing)?|how'?s\s+it\s+going|are\s+you\s+there|you\s+there|"
    r"what'?s\s+your\s+status|what\s+are\s+you\s+doing|how\s+are\s+we\s+doing)[\s,]+)(?:bash|as|ass|ashe|asch)"
    r"(?=\s*[?.!]*\s*$)", re.I)
_MISHEARD_GREETED = re.compile(r"^(\s*(?:hey|hi|yo|ok|okay)[\s,]+)bash(?=[\s,]+\w)", re.I)


def fix_name(text: str) -> str:
    """Whisper's usual mishearings of the name at the start of a sentence
    ("Ass system status") read as the name. Only at the start, and only
    with more words after it."""
    t = _MISHEARD.sub(lambda m: m.group(1) + "Ash", text or "", count=1)
    # "Yo, what's up bash?" / "What's your status as?" -- the name at the END, heard as another word. Only after a
    # question that is plainly a question to a listener, so "as" and "bash" elsewhere are left alone. "bash" at the
    # start counts only after a greeting ("hey bash how are you"): on its own it is a shell.
    t = _MISHEARD_TAIL.sub(lambda m: m.group(1) + "Ash", t, count=1)
    return _MISHEARD_GREETED.sub(lambda m: m.group(1) + "Ash", t, count=1)


def build_event(source: str, text: str, verdict, data=None, **meta) -> SensorEvent:
    """One heard utterance as a SensorEvent, graded by the address verdict.

    Shared with the daemon's `simulate mic` so a simulated sentence carries
    exactly the numbers a real one would -- a test harness that scores its
    input differently from the real sensor tests nothing.

    Salience and urgency rise with the confidence rather than stepping at a
    single threshold, so an utterance that is probably-but-not-certainly for
    ASH still arrives at the attention gate carrying weight.
    """
    conf = float(verdict.confidence)
    return SensorEvent(
        source=source, modality=Modality.TEXT, text=text, data=data,
        salience=round(min(0.95, 0.45 + 0.50 * conf), 3),
        urgency=round(0.65 * conf if conf >= address.LIKELY_AT else 0.0, 3),
        addressed=verdict.addressed,
        meta={"addressed": verdict.addressed, "address": verdict.as_dict(), **meta},
    )


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
    # Live voice, off unless __init__ says otherwise (and present on any instance, however it was made).
    speculative = False
    barge = None
    wake = None
    on_barge_in = None

    def __init__(self, interval: Optional[float] = None, enabled: bool = False,
                 device: Optional[int] = None,
                 silence_ms: int = 700, min_speech_ms: int = 400,
                 max_utterance_s: float = 20.0,
                 transcriber=None, keep_audio: bool = False, voiceprint: bool = False,
                 speculative: bool = True, early_ms: int = 300, barge=None, wake=None, local_only: bool = True):
        self.device = device
        # privacy.require_local_asr: with it (the default) speech is recognised on this machine or not at all. The online
        # fallback used to run whenever Whisper failed to load or ran out of memory, which sent the room's audio to Google.
        self.local_only = bool(local_only)
        self.second_pass_model = "small.en"       # '' turns the second look off
        self.restarts = 0
        # Live voice (livevoice.py). speculative: start transcribing `early_ms` into the end-of-turn silence, so the
        # words are ready when the silence is long enough (a word after that throws the guess away). barge: a
        # BargeIn -- the user talking over ASH stops it at once (on_barge_in), from the level alone. wake: a WakeGate
        # -- with a wake word, only utterances it heard (or an open conversation) are transcribed.
        self.speculative = speculative
        self.early_frames = max(1, early_ms // FRAME_MS)
        self.barge = barge
        self.wake = wake
        self.speaking_fn = lambda: False          # set by the daemon: is ASH talking right now?
        self.on_barge_in = None                   # set by the daemon: stop ASH talking
        self._last_rms = 0.0
        self.live_stats = {"barge_ins": 0, "early_hits": 0, "early_wasted": 0}
        # A rough voiceprint per utterance (src/senses/voiceprint.py), kept in
        # memory with the event and never written out; off unless the
        # daemon's voice match is on.
        self.voiceprint = voiceprint
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 // FRAME_MS)
        self.keep_audio = keep_audio
        self._transcriber = transcriber
        # Words worth hearing right: names of people, classes, ports. Handed to the LOCAL recogniser only, as the start of its
        # prompt (it makes "Linda" likelier than "Lender" and "port" than "board"); never to an online service.
        self.hint_words: Callable[[], List[str]] = lambda: []
        self.corrector: Optional[Callable[[str], str]] = None       # learned mishearings put right (daemon/misheard.py)

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
        # Only listen if the sensor is really usable: an unavailable or errored
        # microphone must not start a capture thread behind its own back.
        if self.state == State.ACTIVE:
            self._start_capture()

    def disable(self):
        super().disable()
        self._stop_capture()
        with self._lock:                  # what was heard before "off" is not acted on after it
            self._out.clear()

    # ------------------------------------------------------------------
    # Capture thread
    # ------------------------------------------------------------------
    def on_retry(self):
        self.restarts = 0
        self._start_capture()

    def _start_capture(self):
        if self._thread and self._thread.is_alive():
            if not self._stop.is_set():
                return                                   # running
            # Being stopped, and still busy with the last phrase: wait for it to finish, or "on" straight after "off"
            # finds a live thread, returns, and the thread then ends -- a microphone that says it is on and hears nothing.
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Microphone: the previous capture is still finishing; try again in a moment")
                return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ash-mic")
        self._thread.start()
        logger.info("Microphone capture started (%s)", self._backend)

    def _stop_capture(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)

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

    def _rise_floor(self, rms: float) -> None:
        """The floor could only FALL: it follows frames under twice itself, so a room that got louder (a fan, a game, the street) than twice what it was
        in the quiet hour it started from never raised it, and everything counted as speech. The quietest tenth of the last minute or so is what the room
        is doing even while someone talks (people pause); the floor climbs towards it, a tenth at a time, never past 0.05."""
        hist = getattr(self, "_rms_hist", None)
        if hist is None:
            import collections
            hist = self._rms_hist = collections.deque(maxlen=2000)
            self._rms_n = 0
        hist.append(rms)
        self._rms_n += 1
        if self._rms_n % 100 or len(hist) < 1000:
            return
        quiet = sorted(hist)[len(hist) // 10]
        if quiet > self._noise_floor * 1.5:
            self._noise_floor = min(0.05, quiet, self._noise_floor * 1.1)

    def _is_speech(self, frame) -> bool:
        import numpy as np

        rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)) + 1e-9)
        self._last_rms = rms

        # Track the noise floor from quiet frames only, so a long conversation
        # doesn't drag the threshold up until nothing registers as speech.
        if rms < self._noise_floor * 2.0:
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms
        self._rise_floor(rms)

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
        early = None                  # (thread, holder) of a speculative transcription for the current utterance

        try:
            for frame in self._frames():
                if self.restarts:
                    self.restarts = 0                     # frames are flowing again: the next failure starts the count afresh
                speech = self._is_speech(frame)
                self._live(frame)

                if not speaking:
                    preroll.append(frame)
                    if speech:
                        speaking = True
                        buf = list(preroll)   # keep the pre-onset audio
                        silence = 0
                    continue

                buf.append(frame)
                if speech and early is not None:
                    early = None              # they went on talking: the early guess is for a shorter sentence
                    self.live_stats["early_wasted"] += 1
                silence = 0 if speech else silence + 1

                if (self.speculative and early is None and silence == self.early_frames
                        and len(buf) - silence >= self.min_speech_frames):
                    early = self._start_early(np.concatenate(buf))

                too_long = len(buf) >= self.max_frames
                if silence >= self.silence_frames or too_long:
                    speaking = False
                    if len(buf) >= self.min_speech_frames:
                        audio = np.concatenate(buf)
                        text = self._early_result(early) if early is not None else None
                        self._finish(audio, text=text)
                    early = None
                    buf, preroll = [], collections.deque(maxlen=int(400 // FRAME_MS))
        except Exception:
            # A headset unplugged, a game grabbing the device, an overflow: not the end of listening. Try again in a
            # little while, a little longer each time, instead of staying deaf until someone turns it off and on.
            self.restarts += 1
            wait = min(60.0, 5.0 * self.restarts)
            logger.exception("Microphone loop died (restart %d in %.0fs)", self.restarts, wait)
            if self.restarts <= 8 and not self._stop.is_set():
                def again():
                    if not self._stop.wait(wait) and self.state == self.state.__class__.ACTIVE:
                        if self._thread is None or not self._thread.is_alive():      # not over one that someone else started
                            self._start_capture()
                threading.Thread(target=again, daemon=True, name="ash-mic-retry").start()
            else:
                self._give_up()

    def _give_up(self):
        """Eight restarts did not bring the device back. ERROR, WITH the time: `Sensor.due()` tries a sensor that gave up again after
        RETRY_AFTER_S only when `errored_at` is set, and it was never set here -- so a headset unplugged for three minutes left the
        microphone dead until someone switched it off and on, the very thing the loop above says it is avoiding."""
        self.state = self.state.__class__.ERROR
        self.errored_at = time.time()

    def _live(self, frame):
        """Per frame: the wake word, and the user talking over ASH."""
        if self.wake is not None:
            self.wake.feed(frame)
        if self.barge is not None:
            try:
                talking = bool(self.speaking_fn())
            except Exception:
                talking = False
            if self.barge.feed(self._last_rms, talking) and self.on_barge_in is not None:
                self.live_stats["barge_ins"] += 1
                logger.info("Mic: talked over ASH -- stopping it")
                try:
                    self.on_barge_in()
                except Exception:
                    logger.exception("Barge-in stop failed")

    def _start_early(self, audio):
        """Transcribe now, on the side, as if the turn had ended. Returns (thread, holder)."""
        if self.wake is not None and not self.wake.allows(len(audio) / SAMPLE_RATE):
            return None
        holder: Dict[str, Any] = {}

        def run():
            try:
                holder["text"] = self.transcribe(audio)
            except Exception:
                holder["text"] = None
        t = threading.Thread(target=run, daemon=True, name="ash-mic-early")
        t.start()
        return (t, holder)

    def _early_result(self, early) -> Optional[str]:
        t, holder = early
        t.join(timeout=30.0)
        text = holder.get("text")
        if text is not None:
            self.live_stats["early_hits"] += 1
        return text

    # ------------------------------------------------------------------
    def _finish(self, audio, text: Optional[str] = None):
        """One complete utterance: transcribe (unless the early transcription already did) and queue it."""
        duration = len(audio) / SAMPLE_RATE
        if text is None:
            if self.wake is not None and not self.wake.allows(duration):
                logger.debug("Mic: %.2fs clip without the wake word (not transcribed)", duration)
                return
            text = self.transcribe(audio)
        text = fix_name(text)
        corrector = getattr(self, "corrector", None)         # (a sensor built without __init__, as the tests do, has none)
        if text and corrector is not None:
            try:
                text = corrector(text) or text
            except Exception:
                logger.debug("transcript correction failed", exc_info=True)
        if not text:
            # Very frequent with the energy-VAD fallback (no webrtcvad) --
            # noise/room tone clears the level gate but transcribes to
            # nothing. debug, not info, or this drowns the log.
            logger.debug("Mic: %.2fs clip transcribed empty (discarded)", duration)
            return

        # ASH's own voice, coming back through the speakers: not the user.
        try:
            from .echo import guard
            if guard.is_echo(text, time.time(), duration):
                logger.info("Mic: ignored ASH's own voice (%.2fs): %r", duration, text)
                return
        except Exception:
            logger.debug("echo check failed", exc_info=True)

        verdict = address.score(text)
        addressed = verdict.addressed
        self._speech_seen += 1
        # The only place the actual words ASH heard are visible anywhere --
        # without this, "did it hear me, and what did it think I said" is
        # unanswerable from the logs alone. The confidence and its reasons go
        # in too: "why didn't it answer me" is the other unanswerable one.
        logger.info("Mic transcript (%.2fs, address=%.2f %s): %r", duration,
                    verdict.confidence, "/".join(verdict.reasons) or "nothing", text)

        extra = {}
        conf = getattr(self, "_last_asr_conf", None)         # how sure the recogniser was (0 = certain): a doubtful command is said back first
        if conf is not None:
            extra["asr_conf"] = round(float(conf), 3)
        if self.voiceprint:
            from .voiceprint import fingerprint
            vp = fingerprint(audio, SAMPLE_RATE)
            if vp is not None:
                extra["voice"] = vp
        with self._lock:
            if getattr(self, "_stop", None) is not None and self._stop.is_set():       # switched off while this was being transcribed
                return
            self._out.append(build_event(self.name, text, verdict,
                                         data=(audio if self.keep_audio else None),
                                         duration_s=round(duration, 2), **extra))

    @staticmethod
    def is_addressed(text: str) -> bool:
        """Certainly spoken to. `address.score()` is the richer answer -- it
        also reports the cases that are probably but not certainly for ASH,
        which this boolean has no way to express."""
        return address.is_addressed(text)

    @staticmethod
    def strip_address(text: str) -> str:
        """Drop a leading/trailing wake phrase before the text is routed.
        The word "ash" on its own scores 0.43 for system_stats in the intent
        classifier, so leaving it in put a phantom system_stats candidate
        into every "hey ash ..." query and diluted the real match ("what
        time is it": get_time 0.42 bare vs 0.34 with "hey ash")."""
        stripped = _TRAILING_WAKE.sub("", _LEADING_WAKE.sub("", text or "")).strip()
        return stripped or (text or "")

    def prompt_hint(self) -> str:
        """"Talking to Ash, my assistant. Words I use: Khalid, Linda, NoSQL." -- short, because the recogniser reads it all."""
        base = "Talking to Ash, my assistant."
        try:
            words = [str(w).strip() for w in (self.hint_words() or []) if str(w).strip()]
        except Exception:
            logger.debug("hint words failed", exc_info=True)
            words = []
        seen, keep = set(), []
        for w in words:
            k = w.lower()
            if k not in seen and len(w) <= 24:
                seen.add(k)
                keep.append(w)
        return base if not keep else base + " Words I use: " + ", ".join(keep[:25]) + "."

    @staticmethod
    def _mean_conf(segs) -> Optional[float]:
        lps = [getattr(x, "avg_logprob", None) for x in segs if (getattr(x, "text", "") or "").strip()]
        lps = [v for v in lps if v is not None]
        return (sum(lps) / len(lps)) if lps else None

    @staticmethod
    def _unsure(segs) -> bool:
        """True when the phrases whisper kept average below the confidence that is worth a second look (0 is certain)."""
        lps = [getattr(x, "avg_logprob", None) for x in segs if (getattr(x, "text", "") or "").strip()]
        lps = [v for v in lps if v is not None]
        return bool(lps) and sum(lps) / len(lps) < SECOND_PASS_LOGPROB

    def _second_pass(self, audio) -> str:
        """The same audio read by a bigger model. Only one already on disk (local_files_only, so never a download), loaded the first time it is
        needed and left alone for good if it is not there. Empty means 'keep the first reading'."""
        if getattr(self, "_second_dead", False):
            return ""
        try:
            import numpy as np
            from faster_whisper import WhisperModel  # type: ignore
            if not hasattr(self, "_whisper2"):
                self._whisper2 = WhisperModel(self.second_pass_model, device="cpu", compute_type="int8", cpu_threads=2, local_files_only=True)
            segs, _ = self._whisper2.transcribe(audio.astype(np.float32) / 32768.0, language="en", vad_filter=True,
                                                initial_prompt=self.prompt_hint(), condition_on_previous_text=False)
            return trusted_text(list(segs))
        except Exception:
            self._second_dead = True
            logger.info("Mic: no second-pass model (%s) on disk; keeping the first reading", self.second_pass_model)
            return ""

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
                # Two threads, not every core: the game, or whatever else is running, still needs most of them.
                self._whisper = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=2,
                                             local_files_only=self.local_only)
            # The hint makes the name likelier to come out as "Ash" -- it was
            # heard as "Ass system status" and ignored.
            segs, _ = self._whisper.transcribe(audio.astype(np.float32) / 32768.0,
                                               language="en", vad_filter=True,
                                               initial_prompt=self.prompt_hint(),
                                               # each phrase is read on its own: carrying the last one
                                               # forward is how whisper invents words from room noise
                                               condition_on_previous_text=False)
            segs = list(segs)
            text = trusted_text(segs)
            self._last_asr_conf = self._mean_conf(segs)
            if text and getattr(self, "second_pass_model", "") and self._unsure(segs):
                better = self._second_pass(audio)          # the fast model was unsure: once more with a bigger one, if it is already on disk
                if better:
                    return better
            return text
        except Exception:
            if self.local_only:
                logger.warning("Local speech recognition failed; not sending audio anywhere (privacy.require_local_asr)")
                return ""

        if self.local_only:
            return ""           # privacy.require_local_asr: the room's audio never goes to an online service

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
            "wake_word": self.wake is not None, "barge_in": self.barge is not None, **self.live_stats,
        })
        return d
