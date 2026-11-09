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

import contextlib
import gc
import logging
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ash.senses.screenread")

# ----------------------------------------------------------------------
# What reading is allowed to cost.
#
# An always-on assistant that reads the screen has to be the quietest thing
# on the machine. Left to its defaults, onnxruntime opens one thread per
# core for each of rapidocr's three models, so a read pinned every core at
# once -- on a PC already at 99% memory with a video playing, that ended in
# "bad allocation" and a daemon that went down with it. Every number here is
# about being a good neighbour rather than being fast: a read that takes six
# seconds instead of three, on two cores, at low priority, only when the
# machine has room, is the right trade for something that runs every two
# minutes all day.
# ----------------------------------------------------------------------
OCR_THREADS = 2                 # per model; the default is "every core"
OCR_MAX_SIDE = 1600             # longest edge handed to the engine
OCR_MAX_PIXELS = 1_500_000      # ...and its area, so a 4:3 frame is not bigger
MEMORY_CEILING_PCT = 88.0       # ambient reads wait while RAM use is above this
MEMORY_FLOOR_MB = 900           # ...or while less than this is available
CPU_CEILING_PCT = 85.0          # ...or while the whole machine is this busy
MIN_CONFIDENCE = 0.6            # recogniser score below this is a guess
BACKOFF_FIRST_S = 60.0          # after a failed read, wait this, doubling...
BACKOFF_MAX_S = 1800.0          # ...up to half an hour

# One read at a time in the whole process. The ambient worker and a spoken
# "what do you see" are separate callers; two engines inferring at once is
# double the memory at exactly the moment there is least of it.
_RUN_LOCK = threading.Lock()


def _is_memory_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return isinstance(exc, MemoryError) or "bad allocation" in text or "bad_alloc" in text \
        or "out of memory" in text or "failed to allocate" in text


def pressure() -> str:
    """Why an optional read should wait, or "" when the machine has room.

    Checked right before every ambient read, because the earlier answer is
    stale by the time a six-second job starts. psutil is already a dependency
    of the window sensor; without it nothing is checked and nothing breaks.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        if vm.percent >= MEMORY_CEILING_PCT or vm.available < MEMORY_FLOOR_MB * 1024 * 1024:
            return "memory"
        # Non-blocking: compares against the previous call, so the first
        # reading is meaningless and is thrown away rather than trusted.
        busy = psutil.cpu_percent(interval=None)
        if busy >= CPU_CEILING_PCT:
            return "cpu"
    except Exception:
        pass
    return ""


def fit_for_ocr(img, max_side: int = OCR_MAX_SIDE, max_pixels: int = OCR_MAX_PIXELS):
    """A copy of `img` no bigger than the engine should be asked to read.

    One allocation at the target size. The old route was `img.copy()` and
    then `thumbnail()`, which duplicated a full 4K frame (33MB) only to throw
    most of it away -- twice per read, on a machine that was out of memory.
    """
    try:
        w, h = img.size
        scale = min(1.0, max_side / float(max(w, h)), (max_pixels / float(max(1, w * h))) ** 0.5)
        if scale >= 1.0:
            return img.copy()
        from PIL import Image
        return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    except Exception:
        return img


@contextlib.contextmanager
def _low_priority():
    """Run the enclosed work on a below-normal-priority thread (Windows).

    Set on the thread doing the reading and put back afterwards, because a
    spoken question runs on a thread the daemon keeps using. This covers the
    image preparation and result handling; the engine's own pool threads are
    bounded by OCR_THREADS instead. Elsewhere it does nothing -- `nice` is a
    per-process setting on Linux and not worth changing for a worker.
    """
    if os.name != "nt":
        yield
        return
    k32, handle, before = None, None, None
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentThread.restype = ctypes.c_void_p
        k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        handle = k32.GetCurrentThread()
        k32.GetThreadPriority.argtypes = [ctypes.c_void_p]
        before = k32.GetThreadPriority(handle)
        k32.SetThreadPriority(handle, -1)           # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:
        k32 = None
    try:
        yield
    finally:
        if k32 is not None and before is not None:
            try:
                k32.SetThreadPriority(handle, before)
            except Exception:
                pass

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


# ASH's own output on a screen. Window titles cannot say a terminal is ASH's
# (it is "Windows PowerShell" like any other), and the Claude app's terminal
# panel is not an ancestor of anything, so the last line of defence is what
# the words themselves look like. Without it ASH read its log aloud -- "the
# part I can read says: or': 0.066, 'mean_weight_before'" -- and the reply it
# stored became the next thing it read. The patterns tolerate what OCR does
# to a log: INFO as "INF0", and so on.
_OWN_LOG = re.compile(
    r"\bash\.(?:daemon|brain|senses|classify|router|ner|tts|memory|concepts|tools|ui|face|disamb|"
    r"shared_model|actuators)\b"
    r"|\b(?:INF[O0]|WARNING|ERROR|DEBUG)\s+ash(?:d|\.)"
    r"|\bashd\.py\b|\bashctl\b"
    r"|Intent candidate|classify_intent called|get_intent_candidates|sentiment_tool called"
    r"|mean_weight_before|Sleep phase|Ambient runtime|VLA: registered|Constitution: \d+ hard"
    r"|\[ANSWER via (?:speak|log|notify)\]",
    re.I)
_LOG_LINE = re.compile(r"\b20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d[,.]\d{2,3}\s+[A-Z0]{4,8}\b")


def looks_like_own_output(lines: List[str]) -> bool:
    """Do these lines look like ASH's own log or its own answers?

    One line naming one of its modules is enough -- nothing else says
    "ash.daemon.ambient". Timestamped log lines in general are not: another
    program's log is the user's to have read, but three of them in a frame is
    a terminal tailing a log, and a daemon's own is the likelier one.
    """
    stamped = 0
    for raw in lines or []:
        line = raw or ""
        if _OWN_LOG.search(line):
            return True
        if _LOG_LINE.search(line):
            stamped += 1
    return stamped >= 3


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

    def __init__(self, backend: Optional[str] = None, threads: int = OCR_THREADS):
        self.backend: Optional[str] = None
        self._engine: Any = None
        self._threads = max(1, int(threads or OCR_THREADS))
        self._failures = 0           # all time
        self._streak = 0             # since the last good read
        self._wait_until = 0.0       # no reading before this, after a failure
        self._skipped = {"memory": 0, "cpu": 0, "backoff": 0, "busy": 0}
        self._last_error = ""
        # Why the last read returned nothing, when the reason is worth acting
        # on: "self" means the screen was showing ASH's own output.
        self.last_note = ""
        self._probe(backend)

    # ------------------------------------------------------------------
    def _build_rapidocr(self):
        """RapidOCR with its thread count set, not left at "every core".

        The package reads thread counts from its yaml and offers no working
        keyword for two of its three models, so the stock config is loaded,
        changed, and handed back as a file. The file is only needed while the
        sessions are being created.
        """
        from rapidocr_onnxruntime import RapidOCR
        try:
            import yaml
            from rapidocr_onnxruntime.main import DEFAULT_CFG_PATH
            with open(DEFAULT_CFG_PATH, "rb") as f:
                cfg = yaml.load(f, Loader=yaml.Loader)
            for part in ("Det", "Cls", "Rec"):
                cfg[part]["intra_op_num_threads"] = self._threads
                cfg[part]["inter_op_num_threads"] = 1
            fd, path = tempfile.mkstemp(suffix=".yaml", prefix="ash-ocr-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)
                return RapidOCR(config_path=path)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(path)
        except Exception:
            logger.debug("Could not cap OCR threads; using the engine defaults", exc_info=True)
            return RapidOCR()

    def _probe(self, forced: Optional[str] = None):
        order = [forced] if forced else ["rapidocr", "tesseract", "winocr"]
        for name in order:
            try:
                if name == "rapidocr":
                    self._engine = self._build_rapidocr()
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
        """An engine is installed. It may have been put away after a failure
        and be rebuilt on the next read -- that is still "available"."""
        return self.backend is not None

    @property
    def waiting(self) -> bool:
        """Cooling off after a failure: reads are refused until it ends."""
        return time.time() < self._wait_until

    # ------------------------------------------------------------------
    def read_text(self, img, ambient: bool = False) -> str:
        """The gist of the words on screen, or "" for any reason at all.

        Never raises. A screen reader that can take the daemon down is worse
        than one that reads nothing.

        `ambient` is the background loop reading on its own initiative, and
        it yields: to a failure it is still backing off from, to a machine
        that is short of memory or busy, and to another read in progress. A
        read somebody asked for is not turned away for load -- they are
        waiting on it -- but it still takes its turn and still backs off after
        a failure, so a broken engine is not retried in a tight loop either.
        """
        self.last_note = ""
        if not self.available or img is None:
            return ""
        if self.waiting:
            self._skipped["backoff"] += 1
            return ""
        if ambient:
            why = pressure()
            if why:
                self._skipped[why] += 1
                return ""
        # Ambient never queues behind a read; a question waits for its turn.
        if not _RUN_LOCK.acquire(blocking=not ambient, timeout=60 if not ambient else -1):
            self._skipped["busy"] += 1
            return ""
        try:
            with _low_priority():
                lines = self._lines(fit_for_ocr(img))
        except Exception as e:
            self._failed(e)
            return ""
        finally:
            _RUN_LOCK.release()
        self._streak = 0
        if looks_like_own_output(lines):
            self.last_note = "self"
            return ""
        return _gist(lines)

    def _failed(self, exc: BaseException):
        """Count it, say so once and briefly, back off, and let go of memory.

        The traceback used to be logged in full, three screens of it for one
        ONNX message. The message is the information; the stack is debug.
        When the failure is an allocation the engine is dropped as well --
        its sessions are most of what this process holds -- and rebuilt on the
        next read rather than inherited half-broken.
        """
        self._failures += 1
        self._streak += 1
        reason = str(exc).strip().splitlines()[-1][:160] if str(exc).strip() else type(exc).__name__
        self._last_error = reason
        wait = min(BACKOFF_MAX_S, BACKOFF_FIRST_S * (2 ** (self._streak - 1)))
        self._wait_until = time.time() + wait
        if self._streak == 1 or self._failures in (10, 100):
            logger.warning("Screen OCR failed (%d times, next try in %.0fs): %s",
                           self._failures, wait, reason)
        logger.debug("Screen OCR failure detail", exc_info=exc)
        if _is_memory_error(exc) and self.backend == "rapidocr":
            self._engine = None
            gc.collect()

    def _lines(self, img) -> List[str]:
        if self.backend == "rapidocr":
            import numpy as np
            if self._engine is None:
                self._engine = self._build_rapidocr()
            result, _elapse = self._engine(np.asarray(img))
            # [[box, text, score], ...], already in reading order. A line the
            # recogniser itself is unsure of is a guess, and guesses are how
            # "INFO" became "INF0" and a port number became "1oo61" in what
            # ASH then said, spoke and remembered. Better to read less.
            keep = []
            for r in (result or []):
                if len(r) < 2:
                    continue
                try:
                    score = float(r[2]) if len(r) > 2 else 1.0
                except (TypeError, ValueError):
                    score = 1.0
                if score >= MIN_CONFIDENCE:
                    keep.append(str(r[1]))
            return keep
        if self.backend == "tesseract":
            return (self._engine.image_to_string(img) or "").splitlines()
        if self.backend == "winocr":
            res = self._engine.recognize_pil_sync(img, "en")
            return [ln.text for ln in getattr(res, "lines", [])]
        return []

    def status(self) -> Dict[str, Any]:
        return {"backend": self.backend or "none", "failures": self._failures,
                "waiting_s": round(max(0.0, self._wait_until - time.time()), 1),
                "skipped": dict(self._skipped), "threads": self._threads,
                "last_error": self._last_error}


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
