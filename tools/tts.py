# tts.py
"""
Kokoro speech, synthesized as one utterance.

This used to split the reply on sentence boundaries and hand each sentence
to Kokoro as a separate call, yielding a separate WAV per sentence. Two
things were wrong with that, and together they are most of why ASH sounded
like a machine reading a list:

  * Every sentence was a fresh WAV played by a fresh blocking PlaySound, so
    there was a hard seam between them -- not a pause a person would make,
    a gap while the next clip was handed to the sound card.
  * Prosody never crossed the boundary. Kokoro decides pitch and pace over
    the whole span it is given; cut that span at every full stop and every
    sentence restarts at the same neutral baseline, which is exactly the
    flat newsreader cadence.

So the whole reply goes in at once. Kokoro has a finite phoneme budget per
call, so a long reply is still split -- but only when it has to be, at the
largest safe span rather than the smallest, and the samples are concatenated
into a single WAV before anything is yielded. Callers get one chunk, and
what plays is continuous.
"""

import io
import logging
import re

try:
    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

logger = logging.getLogger("ash.tts")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    logger.addHandler(logging.StreamHandler(stream=sys.stderr))

# Kokoro's per-call phoneme budget is 510 tokens and it truncates silently
# past that. English runs roughly one token per 2.5 characters, so this is a
# deliberately pessimistic character bound -- the daemon caps replies at 600
# characters anyway, so most of them go in one call and never split at all.
MAX_SEGMENT_CHARS = 420

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
# Markdown furniture ASH should not read out. Hyphens are deliberately not in
# here: stripping them turned "e-mail" into "email" and "well-known" into
# "wellknown", and a leading list dash is handled on its own below.
_MARKDOWN = re.compile(r"[*_`#]+|~~|^\s*[-+]\s+", re.M)
_WHITESPACE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Strip what is written-only, keep what is spoken."""
    out = _MARKDOWN.sub(" ", text or "")
    return _WHITESPACE.sub(" ", out).strip()


def segment(text: str, limit: int = MAX_SEGMENT_CHARS):
    """The fewest spans that each fit the model's budget.

    Sentences are packed greedily rather than emitted one by one, so a short
    reply is a single span and a long one is a handful -- the opposite of the
    old behaviour, which split as finely as it possibly could.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    pieces = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        # One sentence longer than the budget: fall back to clauses, then to
        # words, so the split lands somewhere a voice would breathe.
        run = ""
        for clause in _CLAUSE_END.split(sentence):
            for part in ([clause] if len(clause) <= limit else clause.split(" ")):
                if len(run) + len(part) + 1 > limit and run:
                    pieces.append(run.strip())
                    run = ""
                run = f"{run} {part}".strip()
        if run:
            pieces.append(run.strip())

    spans, run = [], ""
    for piece in pieces:
        if len(run) + len(piece) + 1 > limit and run:
            spans.append(run.strip())
            run = ""
        run = f"{run} {piece}".strip()
    if run:
        spans.append(run.strip())
    return spans


def sentence_chunks(text: str, min_chars: int = 40, max_chars: int = 240):
    """Split a reply where a listener would hear a pause, keeping short sentences together ("Sure. It's 3 PM." is one
    chunk) so each chunk is worth synthesizing on its own, and very long sentences at commas."""
    import re as _re
    sents = [s.strip() for s in _re.split(r"(?<=[.!?;:])\s+", (text or "").strip()) if s.strip()]
    out, cur = [], ""
    for s in sents:
        while len(s) > max_chars:
            cut = s.rfind(", ", 0, max_chars)
            cut = cut + 1 if cut > min_chars else max_chars
            out.append((cur + " " + s[:cut]).strip() if cur else s[:cut].strip())
            cur, s = "", s[cut:].strip()
        cur = (cur + " " + s).strip() if cur else s
        if len(cur) >= min_chars:
            out.append(cur)
            cur = ""
    if cur:
        if out and len(cur) < min_chars // 2:
            out[-1] = out[-1] + " " + cur
        else:
            out.append(cur)
    return out or ([text] if text else [])


class TTSEngine:
    """Always-on Kokoro TTS engine."""
    # Defaults match what README setup downloads and what app.py passes; the
    # old v0_19/voices.json pair was never fetched by any documented step.
    def __init__(self, model_path="models/kokoro-v1.0.onnx", voices_path="models/voices-v1.0.bin"):
        """Initializes the always-on Kokoro TTS engine."""
        self.engine = None
        if not KOKORO_AVAILABLE:
            logger.warning("Kokoro-ONNX or soundfile not installed. Local TTS disabled.")
            return

        try:
            logger.info(f"Loading Kokoro TTS Engine from {model_path}...")
            self.engine = Kokoro(model_path, voices_path)
            logger.info("Kokoro TTS Engine ready.")
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")

    def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0):
        """The whole reply as one WAV, or None if there is nothing to say.

        Where the text had to be split, the sample arrays are joined before
        the WAV is written -- the seam is inside one continuous stream of
        audio rather than between two separate playbacks.
        """
        if not self.engine:
            return None

        spans = segment(clean_for_speech(text))
        if not spans:
            return None

        chunks, rate = [], None
        for span in spans:
            try:
                samples, sample_rate = self.engine.create(
                    span, voice=voice, speed=speed, lang="en-us"
                )
            except Exception:
                logger.exception("Kokoro failed on a %d-char span", len(span))
                continue
            if samples is None or len(samples) == 0:
                continue
            chunks.append(samples)
            rate = sample_rate

        if not chunks:
            return None

        audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, rate, format="WAV", subtype="PCM_16")
        logger.debug("Spoke %d chars as %d span(s), %.1fs of audio",
                     len(text or ""), len(spans), len(audio) / float(rate or 1))
        return wav_io.getvalue()

    def stream_audio(self, text: str, voice: str = "af_heart", speed: float = 1.0, by_sentence: bool = True):
        """The reply as WAV chunks, a sentence or two at a time, so the first words can play while the rest is still
        being made (the speech actuator plays one chunk while the next is synthesized). by_sentence=False: one chunk.
        """
        parts = sentence_chunks(text) if by_sentence else [text]
        for part in parts:
            wav = self.synthesize(part, voice=voice, speed=speed)
            if wav:
                yield wav
