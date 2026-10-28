"""
src/senses/screen.py

What the user is actually doing: active window, screen content, idle time.

This is the highest-value ambient sensor and the most privacy-sensitive one,
so it is built around three ideas:

  1. TITLES ARE CHEAP, PIXELS ARE NOT. The active-window sensor polls every
     few seconds and costs nothing. Screenshots are taken only when the
     window *changes* or on a slow heartbeat, and only if enabled.

  2. DEDUPLICATE PERCEPTUALLY. A 24h loop capturing a static screen would
     write thousands of identical frames. Every capture is reduced to a 64-bit
     average hash; frames within a small Hamming distance of the last one are
     dropped before they ever reach memory. In practice this cuts capture
     volume by well over 90% during normal work.

  3. BLOCKLIST BEFORE CAPTURE, NOT AFTER. Password managers, banking sites,
     calls, and anything the user names are checked against the *window
     title* first. If it matches, no pixels are read at all -- there is no
     sensitive frame sitting in RAM waiting to be filtered.

  4. CAPTURE THE FOCUSED WINDOW, NOT THE MONITOR. Point 3 checks the title
     of the window with focus, so capturing the whole screen quietly broke
     it: a blocked app merely *visible* beside an allowed one was captured
     and read anyway. The first time OCR was switched on, what it read and
     journaled was a lecture being presented over Google Meet in another
     window -- meeting URL, presenter's name and all -- while the focused
     window was a terminal and the blocklist, checking that terminal, said
     yes. Cropping to the focused window makes the check mean what it says.
     Where no rectangle is available the monitor is still used, and that is
     reported in the sensor status rather than assumed.

Idle time matters independently: it is what tells the attention gate whether
the user is at the keyboard (don't interrupt) or has been away for an hour
(a good moment to surface something that has been waiting).
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import Capability, Modality, Sensor, SensorEvent

logger = logging.getLogger("ash.senses.screen")

SCREEN_DIR = os.environ.get("ASH_SCREEN_DIR", os.path.join(os.getcwd(), "state", "screen"))

# Window titles matching any of these are never captured. Extend via config.
DEFAULT_BLOCKLIST = [
    "password", "keepass", "bitwarden", "1password", "lastpass", "keychain",
    "banking", "bank of", "paypal", "stripe dashboard", "wallet", "seed phrase",
    "private browsing", "incognito", "authenticator", "2fa", "recovery code",

    # Other people. Everything above protects the user from their own screen;
    # none of it covered a call or a chat, where the screen belongs partly to
    # someone who was never asked. Turning OCR on for the first time, the very
    # first thing ASH read and wrote to the journal was a lecture someone was
    # presenting over Google Meet, complete with the meeting URL and the
    # presenter's name. That is the clearest case the blocklist exists for and
    # it was not in it.
    # Matched against the real titles these put in a title bar, not the
    # product names. Google Meet is "Meet - abc-defg-hij - Opera", which
    # contains neither "google meet" nor "meet.google" -- the first draft of
    # this list missed a call that was on screen at the time.
    "meet - ", "meet.google", "google meet", "zoom meeting", "zoom.us",
    "- zoom", "microsoft teams", "teams meeting", "webex", "gotomeeting",
    "bluejeans", "jitsi", "skype", "is sharing", "screen sharing",
    "whatsapp", "signal", "telegram", "messenger", "facetime",
]
# A substring list will always trail whatever the next application decides to
# call its windows, and it only ever sees the focused one. It is the cheap
# layer. The structural protections are capturing just that focused window
# (so a blocked app beside an allowed one is out of frame) and
# privacy.blocked_apps, which is the user's own list and the only one that
# knows what their work looks like.


# A window title is "<what you are looking at> - <maybe a project> - <App>",
# near enough universally: "brain.py - A.S.H - Visual Studio Code",
# "Trust-Weighted Retrieval - Google Docs - Google Chrome", "todo.txt -
# Notepad". The first field is the only one that says what the user is
# actually doing, and it was being thrown away everywhere downstream in
# favour of the process name, so ASH knew you had been in Code for forty
# minutes and never which file.
# OCR reads from a frame this wide at most: enough for ordinary UI text,
# far cheaper than running it across a full 4K capture.
OCR_MAX_WIDTH = 1600

_TITLE_SEP = re.compile(r"\s+[-–—|·•:]\s+")
# Leading junk: VS Code's unsaved dot, a modified marker, Chrome's unread
# count, a browser's muted/playing glyph.
_TITLE_NOISE = re.compile(r"^\s*(?:[●•*]\s*|\(\d+\)\s*|[▶\U0001F507\U0001F508]\s*)+")


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def window_fields(title: str, process: str = "", limit: int = 80) -> Tuple[str, str]:
    """(subject, project) -- what is open, and what it belongs to.

    A title is fielded: "brain.py - A.S.H - Visual Studio Code" is a file, a
    project and an app. Keeping only the first of those threw away the half
    that says what the work *was*: "40 minutes on brain.py" is a fact about a
    file, "40 minutes on brain.py in A.S.H" is a fact about a project.

    Either field can come back empty. "Discord" in Discord.exe says nothing
    the process does not already, so it yields ("", "") rather than repeating
    the app back at the caller.
    """
    cleaned = _TITLE_NOISE.sub("", title or "").strip()
    if not cleaned:
        return "", ""

    parts = [p.strip() for p in _TITLE_SEP.split(cleaned) if p.strip()]
    if not parts:
        return "", ""
    bare = len(parts) == 1      # a title with no fields at all

    # Drop trailing segments that are just the application naming itself.
    # Trailing only, and never the last one standing: a document really
    # called "code" is still a document.
    stem = _squash(re.sub(r"\.(?:exe|app|bin)$", "", (process or "").strip(), flags=re.I))
    while len(parts) > 1 and stem:
        tail = _squash(parts[-1])
        if tail and (stem in tail or tail in stem):
            parts.pop()
        else:
            break

    subject = parts[0]
    # "Discord" in Discord.exe, "Visual Studio Code" in Code.exe: a title with
    # no fields that only names the app. Checked only for a bare title, so a
    # file genuinely called Code.md in Code.exe survives.
    if bare and stem:
        sq = _squash(subject)
        if sq and (stem in sq or sq in stem):
            return "", ""
    project = parts[1] if len(parts) > 1 else ""
    return subject[:limit], project[:limit]


def window_subject(title: str, process: str = "", limit: int = 80) -> str:
    """Just the subject. window_fields() is the whole of it."""
    return window_fields(title, process, limit)[0]


def _ahash(img, size: int = 8) -> int:
    """64-bit average hash. Robust to compression and tiny motion, sensitive
    to actual content change -- exactly the tradeoff wanted for 'is this a
    new screen or the same one'."""
    import numpy as np

    a = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32)
    mean = a.mean()
    bits = (a > mean).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
class ActiveWindowSensor(Sensor):
    """Which application has focus, and for how long.

    Cross-platform by backend detection rather than by assuming Windows:
    Win32 via pygetwindow/ctypes, macOS via AppleScript, Linux via xdotool or
    wmctrl. Each backend is probed once at startup; whichever answers first
    wins, and if none do the sensor reports UNAVAILABLE instead of silently
    returning nothing forever.
    """

    capability = Capability(
        name="active_window",
        description="Title and process of the focused window, plus dwell time.",
        privacy_sensitive=True,
    )
    default_interval = 3.0
    modality = Modality.TEXT

    def __init__(self, interval: Optional[float] = None, enabled: bool = True,
                 blocklist: Optional[List[str]] = None):
        self._backend: Optional[str] = None
        self.blocklist = [b.lower() for b in (blocklist or DEFAULT_BLOCKLIST)]
        self._last_title: Optional[str] = None
        self._since: float = time.time()
        # Why the last rect() lookup came back empty. Cropping to the
        # focused window is what makes the title blocklist honest, so
        # falling back to the whole monitor is worth being able to see.
        self.rect_reason: str = "not looked up yet"
        super().__init__(interval=interval, enabled=enabled)

    def probe(self) -> Tuple[bool, str]:
        sysname = platform.system()
        if sysname == "Windows":
            try:
                import ctypes  # noqa: F401
                self._backend = "win32"
                return True, ""
            except Exception as e:
                return False, str(e)
        if sysname == "Darwin":
            self._backend = "applescript"
            return True, ""
        # Linux / BSD
        for tool in ("xdotool", "wmctrl"):
            if _which(tool):
                self._backend = tool
                return True, ""
        return False, "no window backend (install xdotool, or run on Windows/macOS)"

    # -- backends -------------------------------------------------------
    def _title_win32(self) -> Tuple[str, str]:
        import ctypes
        from ctypes import wintypes

        u32 = ctypes.windll.user32
        hwnd = u32.GetForegroundWindow()
        n = u32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u32.GetWindowTextW(hwnd, buf, n + 1)
        pid = wintypes.DWORD()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proc = ""
        try:
            import psutil
            proc = psutil.Process(pid.value).name()
        except Exception:
            pass
        return buf.value, proc

    def _title_applescript(self) -> Tuple[str, str]:
        script = (
            'tell application "System Events" to get '
            'name of first application process whose frontmost is true'
        )
        out = subprocess.run(["osascript", "-e", script], capture_output=True,
                             text=True, timeout=3)
        app = out.stdout.strip()
        return app, app

    def _title_xdotool(self) -> Tuple[str, str]:
        out = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip(), ""

    def _title_wmctrl(self) -> Tuple[str, str]:
        out = subprocess.run(["wmctrl", "-a", ":ACTIVE:", "-v"],
                             capture_output=True, text=True, timeout=3)
        return out.stderr.strip(), ""

    def current(self) -> Tuple[str, str]:
        fn = {
            "win32": self._title_win32,
            "applescript": self._title_applescript,
            "xdotool": self._title_xdotool,
            "wmctrl": self._title_wmctrl,
        }.get(self._backend)
        return fn() if fn else ("", "")

    def rect(self) -> Optional[Tuple[int, int, int, int]]:
        """(left, top, right, bottom) of the focused window, or None.

        Win32 only for now; every other backend reports None and the caller
        falls back to the whole monitor.
        """
        if self._backend != "win32":
            self.rect_reason = f"no rectangle on the {self._backend or 'unprobed'} backend"
            return None
        try:
            import ctypes
            from ctypes import wintypes

            u32 = ctypes.windll.user32
            # HWND is 64-bit on Win64 and ctypes defaults the return to a
            # 32-bit int, which silently truncates a large handle.
            u32.GetForegroundWindow.restype = wintypes.HWND
            hwnd = u32.GetForegroundWindow()
            if not hwnd:
                self.rect_reason = "nothing has focus"
                return None
            r = wintypes.RECT()
            if not u32.GetWindowRect(hwnd, ctypes.byref(r)):
                self.rect_reason = "GetWindowRect refused the handle"
                return None
            if r.right - r.left < 64 or r.bottom - r.top < 64:
                self.rect_reason = "focused window is minimised or a sliver"
                return None          # not worth cropping to
            self.rect_reason = ""
            return int(r.left), int(r.top), int(r.right), int(r.bottom)
        except Exception as e:
            self.rect_reason = f"{type(e).__name__}: {e}"
            logger.debug("GetWindowRect failed", exc_info=True)
            return None

    def status(self) -> Dict[str, Any]:
        d = super().status()
        d["backend"] = self._backend or "none"
        if self.rect_reason:
            d["no_crop_because"] = self.rect_reason
        return d

    def is_blocked(self, title: str) -> bool:
        low = (title or "").lower()
        return any(b in low for b in self.blocklist)

    # -- read ------------------------------------------------------------
    def read(self) -> List[SensorEvent]:
        title, proc = self.current()
        if not title:
            return []

        blocked = self.is_blocked(title)
        now = time.time()

        if title == self._last_title:
            return []   # no change; dwell keeps accumulating silently

        dwell = now - self._since
        prev = self._last_title
        self._last_title, self._since = title, now

        shown = "[redacted window]" if blocked else title
        # A blocked window gets no subject either -- the title is exactly what
        # the blocklist exists to keep out, and the subject is made of it.
        subject, project = ("", "") if blocked else window_fields(title, proc)
        text = f"switched to {shown}" + (f" (was on {prev[:60]} for {dwell:.0f}s)" if prev else "")

        # A rapid flurry of switches is a different signal from settling into
        # one app for an hour -- salience tracks how unusual the change is.
        salience = 0.35 if dwell > 30 else 0.2

        return [SensorEvent(
            source=self.name, modality=Modality.TEXT, text=text,
            salience=salience,
            data={"title": None if blocked else title, "process": proc,
                  "subject": subject, "project": project, "dwell": dwell},
            persistable=not blocked,
            meta={"process": proc, "subject": subject, "project": project,
                  "blocked": blocked, "dwell_s": round(dwell, 1)},
        )]


# ----------------------------------------------------------------------
class ScreenshotSensor(Sensor):
    """Periodic screen capture, deduplicated and blocklist-filtered.

    Disabled by default. This is the single most invasive thing ASH can do,
    and it should be a decision the user makes explicitly rather than a
    default they discover later.
    """

    capability = Capability(
        name="screenshot",
        description="Deduplicated screen captures for visual context.",
        requires=["PIL"],
        privacy_sensitive=True,
    )
    default_interval = 20.0
    modality = Modality.VISION

    def __init__(self, interval: Optional[float] = None, enabled: bool = False,
                 window_sensor: Optional[ActiveWindowSensor] = None,
                 hamming_threshold: int = 6, save_frames: bool = False,
                 max_frames: int = 500, read_text: bool = False,
                 describe: bool = True, read_text_every_s: float = 120.0):
        self._grab = None
        self.window_sensor = window_sensor
        self.hamming_threshold = hamming_threshold
        self.save_frames = save_frames
        self.max_frames = max_frames
        # Two switches, because the two layers cost very differently.
        # `describe` is eight numpy statistics and is on when captures are:
        # it cannot leak anything, since it only ever emits a word from its
        # own small vocabulary. `read_text` is OCR -- seconds of CPU and the
        # actual contents of the screen turned into text that gets journaled
        # -- so it stays off until asked for, the same way capture itself
        # does.
        self.read_text_enabled = read_text
        self.describe_enabled = describe
        self.read_text_every_s = float(read_text_every_s)
        self._reader = None
        # OCR measured 6.6s on a 1600px frame of a 1080p screen on this
        # machine. read() is called from the ambient loop's own polling
        # thread, so doing that inline would freeze sensing for six seconds
        # every time the screen changed -- a third of a core, continuously,
        # on a project that targets Pi-class hardware. It runs on a worker.
        #
        # The result then gets an event of its own rather than being attached
        # to whatever frame happens to come next. Gluing it on produced
        # journal lines like "screen changed while in A.S.H -- reading:
        # <four-minute-old lecture slide>", which is not a stale fact but a
        # false one: it asserts that text is on a screen it is not on.
        self._ocr_lock = threading.Lock()
        self._ocr_pending: Optional[Dict[str, Any]] = None
        self._ocr_at = 0.0
        self._ocr_busy = False
        self._last_hash: Optional[int] = None
        self._dropped = 0
        self._cropped = False
        super().__init__(interval=interval, enabled=enabled)

    @property
    def reader(self):
        """Built on first use, so a daemon with captures off never loads it."""
        if self._reader is None and self.read_text_enabled:
            from .screenread import ScreenReader
            self._reader = ScreenReader()
        return self._reader

    def probe(self) -> Tuple[bool, str]:
        try:
            import mss  # noqa: F401
            self._grab = "mss"
            return True, ""
        except Exception:
            pass
        try:
            from PIL import ImageGrab
            ImageGrab.grab  # noqa: B018
            self._grab = "pil"
            return True, ""
        except Exception as e:
            return False, f"no screen capture backend ({e}); pip install mss"

    def _capture(self):
        """The focused window where its rectangle is known, else the monitor.

        Cropping is what makes the title blocklist honest: it is a statement
        about the focused window, so anything else on the monitor is outside
        what it can promise. It is also cheaper -- less area to hash, less to
        read -- and quieter, since a second window's text no longer lands in
        the gist of the first.
        """
        rect = None
        if self.window_sensor is not None:
            rect = self.window_sensor.rect()

        if self._grab == "mss":
            import mss
            from PIL import Image
            with mss.mss() as sct:
                mon = sct.monitors[1]
                if rect is not None:
                    left, top, right, bottom = rect
                    # Clamp into the monitor: a maximised window reports a
                    # frame a few pixels outside it on Windows.
                    box = {"left": max(mon["left"], left), "top": max(mon["top"], top)}
                    box["width"] = max(1, min(right, mon["left"] + mon["width"]) - box["left"])
                    box["height"] = max(1, min(bottom, mon["top"] + mon["height"]) - box["top"])
                    mon = box
                self._cropped = rect is not None
                shot = sct.grab(mon)
                return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        from PIL import ImageGrab
        self._cropped = rect is not None
        return ImageGrab.grab(bbox=rect) if rect is not None else ImageGrab.grab()

    def read(self) -> List[SensorEvent]:
        # A reading that finished since the last poll goes out on its own,
        # describing the frame it was taken from. It is emitted even when
        # this poll captures nothing, or the reading would be lost whenever
        # the screen settled.
        out: List[SensorEvent] = []
        with self._ocr_lock:
            done, self._ocr_pending = self._ocr_pending, None
        if done and done.get("text"):
            seen = f" in {done['title'][:60]}" if done.get("title") else ""
            out.append(SensorEvent(
                source=self.name, modality=Modality.TEXT,
                text=f"read the screen{seen}: {done['text']}",
                salience=0.45,
                meta={"on_screen": done["text"], "scene": done.get("scene", ""),
                      "ahash": done.get("ahash"), "read_s": done.get("took", 0.0)},
            ))

        # Blocklist check happens BEFORE capture -- see module docstring.
        title = ""
        if self.window_sensor is not None:
            title, _ = self.window_sensor.current()
            if self.window_sensor.is_blocked(title):
                self._dropped += 1
                return out

        img = self._capture()
        h = _ahash(img)

        if self._last_hash is not None and _hamming(h, self._last_hash) <= self.hamming_threshold:
            self._dropped += 1
            return out   # screen hasn't meaningfully changed
        self._last_hash = h

        path = None
        if self.save_frames:
            path = self._save(img)

        from . import screenread

        # OCR runs BEFORE the downscale, and the two layers want different
        # sizes. Reading 12px terminal text out of a 640px-wide thumbnail
        # returns "fspesed 4 actator(s) to the clasifler" -- recognisably
        # "exposed 4 actuator(s) to the classifier", and recognisably
        # useless. 1600px is wide enough for ordinary UI text without
        # paying for OCR across a full 4K frame.
        # Downscale AFTER the OCR hand-off below: the ViT resizes to 224px
        # anyway and a 4K frame in the event queue is pure waste, but OCR
        # needs the pixels.
        legible = img
        img = img.copy()
        img.thumbnail((640, 400))

        scene = screenread.describe(img) if self.describe_enabled else {}
        self._maybe_read(legible, title, scene.get("kind", ""), h)

        # "screen changed while in X" said only that something moved; the
        # scene says what sort of screen it is. What it *says* is a separate
        # event, because by the time OCR finishes this frame is history.
        where = f" while in {title[:60]}" if title else ""
        what = screenread.phrase(scene)
        text = f"screen changed{where}" + (f" -- {what}" if what else "")

        out.append(SensorEvent(
            source=self.name, modality=Modality.VISION,
            text=text,
            data=img, salience=0.4,
            meta={"ahash": h, "path": path, "dropped_since": self._dropped,
                  "scene": scene.get("kind", "")},
        ))
        return out

    def _maybe_read(self, img, title: str, scene: str, ahash: int):
        """Kick off an OCR pass if one is due and none is running.

        Never blocks: the caller is the ambient loop's polling thread, and a
        six-second read there stops ASH sensing anything at all for six
        seconds. The frame's title, scene and hash are carried along so the
        result can describe the screen it was actually taken from.
        """
        reader = self.reader
        if reader is None:
            return
        now = time.time()
        with self._ocr_lock:
            if self._ocr_busy or now - self._ocr_at < self.read_text_every_s:
                return
            self._ocr_busy = True

        legible = img.copy()
        if legible.width > OCR_MAX_WIDTH:
            legible.thumbnail((OCR_MAX_WIDTH, OCR_MAX_WIDTH))

        def _run():
            started, text = time.time(), ""
            try:
                text = reader.read_text(legible)
            finally:
                with self._ocr_lock:
                    self._ocr_at, self._ocr_busy = time.time(), False
                    if text:
                        self._ocr_pending = {
                            "text": text, "title": title, "scene": scene,
                            "ahash": ahash, "took": round(time.time() - started, 2),
                        }

        threading.Thread(target=_run, name="ash-screen-ocr", daemon=True).start()

    def _save(self, img) -> Optional[str]:
        try:
            os.makedirs(SCREEN_DIR, exist_ok=True)
            path = os.path.join(SCREEN_DIR, f"{int(time.time())}.jpg")
            img.copy().convert("RGB").save(path, "JPEG", quality=60, optimize=True)
            self._rotate()
            return path
        except Exception:
            logger.exception("Failed to save frame")
            return None

    def _rotate(self):
        """Bounded on-disk history. An always-on capture loop with no cap
        fills a disk in days."""
        try:
            files = sorted(
                (os.path.join(SCREEN_DIR, f) for f in os.listdir(SCREEN_DIR)
                 if f.endswith(".jpg")),
                key=os.path.getmtime,
            )
            for f in files[:-self.max_frames]:
                os.remove(f)
        except Exception:
            logger.exception("Frame rotation failed")

    def status(self) -> Dict[str, Any]:
        d = super().status()
        d["dropped_duplicates"] = self._dropped
        d["captures"] = "focused window" if self._cropped else "whole monitor"
        d["describes"] = self.describe_enabled
        d["reads_text"] = self.read_text_enabled
        if self._reader is not None:
            d["ocr"] = self._reader.status()
        return d


# ----------------------------------------------------------------------
class IdleSensor(Sensor):
    """Seconds since the last keyboard/mouse input.

    Not a privacy sensor: it reads a single integer from the OS and never
    sees *what* was typed. But it is what lets ASH distinguish "the user is
    mid-sentence, do not interrupt" from "the user left an hour ago, now is a
    fine time to mention that thing".
    """

    capability = Capability(
        name="idle", description="Seconds since last user input.",
    )
    default_interval = 10.0
    modality = Modality.TELEMETRY

    def __init__(self, interval: Optional[float] = None, enabled: bool = True,
                 away_threshold: float = 300.0):
        self._backend: Optional[str] = None
        self.away_threshold = away_threshold
        self._was_away = False
        super().__init__(interval=interval, enabled=enabled)

    def probe(self) -> Tuple[bool, str]:
        s = platform.system()
        if s == "Windows":
            self._backend = "win32"
            return True, ""
        if s == "Darwin" and _which("ioreg"):
            self._backend = "ioreg"
            return True, ""
        if _which("xprintidle"):
            self._backend = "xprintidle"
            return True, ""
        return False, "no idle backend (install xprintidle on Linux)"

    def seconds_idle(self) -> float:
        try:
            if self._backend == "win32":
                import ctypes
                from ctypes import wintypes

                class LASTINPUTINFO(ctypes.Structure):
                    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

                info = LASTINPUTINFO()
                info.cbSize = ctypes.sizeof(info)
                ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))
                millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
                return millis / 1000.0
            if self._backend == "xprintidle":
                out = subprocess.run(["xprintidle"], capture_output=True, text=True, timeout=3)
                return int(out.stdout.strip()) / 1000.0
            if self._backend == "ioreg":
                out = subprocess.run(
                    "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF/1000000000; exit}'",
                    shell=True, capture_output=True, text=True, timeout=3)
                return float(out.stdout.strip() or 0.0)
        except Exception:
            logger.debug("idle read failed", exc_info=True)
        return 0.0

    def read(self) -> List[SensorEvent]:
        idle = self.seconds_idle()
        away = idle >= self.away_threshold

        # Only emit on transitions. A per-poll "still idle" event would be
        # 8,640 useless journal rows a day.
        if away == self._was_away:
            return []
        self._was_away = away

        return [SensorEvent(
            source=self.name, modality=Modality.TELEMETRY,
            text=("user went away from the machine" if away
                  else f"user came back after {idle:.0f}s away"),
            salience=0.45 if not away else 0.25,
            data={"idle_seconds": idle, "away": away},
            meta={"idle_seconds": round(idle, 1), "away": away},
        )]


def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)
