"""
src/senses/screenread.py

Turning a frame into words.

Before this, a screenshot became a 384-float ViT embedding and nothing else.
That is a fine similarity signal -- is this screen different from the last
one -- and completely useless for the thing anyone actually wants, which is
knowing *what* is on the screen. Nothing in the pipeline could turn pixels
into language, so the capture was an expensive way to learn that something
had changed, which the window title already said for free.

Two layers here, deliberately separate, because they have very different
costs and very different failure modes.

READING (`read_text`)
    OCR, through whichever backend is installed. Gives actual words: the
    heading of the document, the error in the terminal, the subject of the
    email. This is the layer worth having and the layer that costs money in
    CPU and in privacy.

LOOKING (`describe`)
    A few numpy statistics over the pixels -- edge density, colour spread,
    how much of the frame is near-uniform. Enough to tell a wall of text
    from a video from an empty desktop, and no more than that. Free, always
    available, nothing to install, and it cannot leak anything because it
    never produces a string it did not already know the vocabulary of.

On object detection
-------------------
YOLO was considered and is not here. The pre-trained weights everyone means
when they say YOLO are COCO: person, bicycle, car, dog, umbrella. On a
desktop screenshot that finds nothing, because a code editor contains no
umbrellas. Detecting *interface* objects -- buttons, tabs, dialogs, the
shape of an IDE versus a browser -- needs a model trained on screenshots,
and the honest version of that is a 100MB+ download and a torch dependency
to answer a question `describe()` answers from eight numbers. If a UI-class
detector ever earns its place it slots in beside `read_text` behind the same
backend probe.

Backends
--------
Probed once, in order, exactly like mic.py's transcriber chain and the
vision extractor's ViT-or-descriptor fallback. None installed is a
supported state: `read_text` returns "" and the sensor goes on working as
it did, with `describe()` and the window title carrying the context.

    rapidocr-onnxruntime   preferred: onnxruntime is already a dependency
                           here (Kokoro), the models are ~15MB, and there is
                           no torch anywhere near it
    pytesseract            if the tesseract binary is actually on PATH --
                           the python package alone is not enough and
                           installing it alone is the usual mistake
    winocr / winsdk        Windows.Media.Ocr, built into the OS, nothing to
                           download
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ash.senses.screenread")

# OCR on a full desktop returns hundreds of fragments, most of them chrome:
# menu bars, tab titles, status lines, clock. The gist is the few longest
# runs of real words, which is what a person glancing at the screen takes in.
# 20, not 12: a menu bar OCRs as "File Edit View Help", which clears a
# 12-char bar and three-real-words test and then turns up in the gist of
# every single frame, because it is on every single frame. Below twenty
# characters a desktop is almost always chrome rather than content, and the
# lines worth keeping -- an error, a heading, a sentence -- are all longer.
MIN_LINE_CHARS = 20
MAX_LINES = 6
MAX_CHARS = 240

# A fragment that is mostly punctuation or single characters is interface
# furniture, not text someone is reading.
_WORDY = re.compile(r"[A-Za-z]{3,}")
_WHITESPACE = re.compile(r"\s+")
# A URL, or a long opaque token: an address bar OCRs into a line like
# "mail.google.com/mail/u/2/#sent?compose=DmwnWrRvwLqDcSPRrTCb..." which is
# not something anyone is reading, and the token in it is a session handle.
_URLISH = re.compile(r"://|www\.|\.(?:com|org|net|io|dev|app)/|[#?]\w+=", re.I)
_OPAQUE = re.compile(r"\S{28,}")


def _gist(lines: List[str]) -> str:
    """The few lines worth keeping, longest first, in reading order."""
    scored = []
    for i, raw in enumerate(lines):
        line = _WHITESPACE.sub(" ", (raw or "").strip())
        if len(line) < MIN_LINE_CHARS:
            continue
        if _URLISH.search(line) or _OPAQUE.search(line):
            continue
        words = _WORDY.findall(line)
        if len(words) < 2:
            continue
        scored.append((len(line), i, line))
    scored.sort(reverse=True)
    keep = sorted(scored[:MAX_LINES], key=lambda s: s[1])

    out, total = [], 0
    for _n, _i, line in keep:
        if total + len(line) > MAX_CHARS:
            break
        out.append(line)
        total += len(line) + 1
    return " / ".join(out)


def _trigrams(text: str) -> set:
    flat = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    return {flat[i:i + 3] for i in range(len(flat) - 2)}


def similar(a: str, b: str, threshold: float = 0.5) -> bool:
    """Are two readings of the same screen?

    Character trigrams over the text with every space and symbol removed,
    because the way OCR differs from itself between two reads of an unchanged
    page is mostly in the spaces -- "teach the network to do" one time,
    "teachthe networktodo" the next -- which makes a word-set comparison call
    the same lecture slide two different pages. Trigrams do not care where
    the gaps fell.
    """
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / float(len(ta | tb)) >= threshold


class ScreenReader:
    """OCR with whichever backend this machine has, or none."""

    def __init__(self, backend: Optional[str] = None):
        self.backend: Optional[str] = None
        self._engine: Any = None
        self._failures = 0
        self._probe(backend)

    # ------------------------------------------------------------------
    def _probe(self, forced: Optional[str] = None):
        order = [forced] if forced else ["rapidocr", "tesseract", "winocr"]
        for name in order:
            try:
                if name == "rapidocr":
                    from rapidocr_onnxruntime import RapidOCR
                    self._engine = RapidOCR()
                elif name == "tesseract":
                    import pytesseract
                    # The python wrapper installs happily without the binary,
                    # and then every call raises. Prove the binary is there.
                    pytesseract.get_tesseract_version()
                    self._engine = pytesseract
                elif name == "winocr":
                    import winocr
                    self._engine = winocr
                else:
                    continue
            except Exception as e:
                logger.debug("Screen OCR backend %s unavailable: %s", name, e)
                continue
            self.backend = name
            logger.info("Screen OCR backend: %s", name)
            return
        logger.info("No screen OCR backend installed; screens will be looked "
                    "at but not read (pip install rapidocr-onnxruntime)")

    @property
    def available(self) -> bool:
        return self._engine is not None

    # ------------------------------------------------------------------
    def read_text(self, img) -> str:
        """The gist of the words on screen, or "" for any reason at all.

        Never raises. A screen reader that can take the daemon down is worse
        than one that reads nothing.
        """
        if not self.available or img is None:
            return ""
        try:
            lines = self._lines(img)
        except Exception:
            self._failures += 1
            if self._failures in (1, 10, 100):
                logger.exception("Screen OCR failed (%d times)", self._failures)
            return ""
        return _gist(lines)

    def _lines(self, img) -> List[str]:
        if self.backend == "rapidocr":
            import numpy as np
            result, _elapse = self._engine(np.asarray(img))
            # [[box, text, score], ...], already in reading order.
            return [str(r[1]) for r in (result or []) if len(r) > 1]
        if self.backend == "tesseract":
            return (self._engine.image_to_string(img) or "").splitlines()
        if self.backend == "winocr":
            res = self._engine.recognize_pil_sync(img, "en")
            return [ln.text for ln in getattr(res, "lines", [])]
        return []

    def status(self) -> Dict[str, Any]:
        return {"backend": self.backend or "none", "failures": self._failures}


# ----------------------------------------------------------------------
# Looking: what kind of thing is on screen, from the pixels alone.
# ----------------------------------------------------------------------
# The useful axis turned out not to be either number on its own but the
# ratio between them: glyphs make many edges in almost no colour, photographs
# and video make colour with comparatively soft edges. Measured rather than
# guessed -- a dense page of text scores 15.9, a real working screen 0.65, a
# colour gradient with noise 0.07 -- which is why the cut is a ratio and the
# absolute edge count is only used to notice an empty screen.
_TEXTY_RATIO = 0.40
_MEDIA_RATIO = 0.12
_FLAT_EDGES = 0.012         # below this, nothing much is on screen


def describe(img) -> Dict[str, Any]:
    """What sort of screen this is, from eight numbers and no model.

    Returns {"kind": ..., "edges": ..., "colour": ..., "ink": ...}. `kind` is
    one of "text", "media", "mixed", "blank", or "unknown" when numpy or the
    image is not usable. Deliberately coarse: this is the difference between
    "they are reading" and "they are watching something", which is most of
    what an ambient assistant needs and all that pixels can honestly say.
    """
    try:
        import numpy as np

        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.size == 0:
            return {"kind": "unknown"}

        grey = arr.mean(axis=2)
        # Fine edge density: glyphs produce a great many short edges, video
        # and photographs produce fewer and softer ones.
        gx = np.abs(np.diff(grey, axis=1)).mean() if grey.shape[1] > 1 else 0.0
        gy = np.abs(np.diff(grey, axis=0)).mean() if grey.shape[0] > 1 else 0.0
        edges = float((gx + gy) / 2.0 / 255.0)
        # How far the channels diverge: text is grey on grey, photos are not.
        colour = float(arr.std(axis=2).mean()) if arr.shape[2] >= 3 else 0.0
        # Proportion of the frame that is near-uniform: margins, empty editor.
        ink = float((np.abs(grey - grey.mean()) > 18.0).mean())

        ratio = edges * 255.0 / max(colour, 1.0)
        if edges < _FLAT_EDGES and ink < 0.08:
            kind = "blank"
        elif ratio >= _TEXTY_RATIO:
            kind = "text"
        elif ratio <= _MEDIA_RATIO:
            kind = "media"
        else:
            kind = "mixed"
        return {"kind": kind, "edges": round(edges, 4), "colour": round(colour, 1),
                "ink": round(ink, 3), "ratio": round(ratio, 2)}
    except Exception:
        logger.debug("describe() failed", exc_info=True)
        return {"kind": "unknown"}


# Two forms, because "-- reading" on its own is not a sentence. With words
# to show, the phrase introduces them; without, it has to stand up alone.
_KIND_PHRASE = {
    "text": ("reading", "a screen of text"),
    "media": ("watching", "video or images"),
    "mixed": ("working in", "a mix of text and images"),
    "blank": ("", "an idle screen"),
}


def phrase(scene: Dict[str, Any], text: str = "") -> str:
    """One short clause for the event line: what this screen looks like."""
    lead, alone = _KIND_PHRASE.get((scene or {}).get("kind", "unknown"), ("", ""))
    if text:
        return f"{lead}: {text}" if lead else text
    return alone
