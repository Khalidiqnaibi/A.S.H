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
     and anything the user names are checked against the *window title* first.
     If it matches, no pixels are read at all -- there is no sensitive frame
     sitting in RAM waiting to be filtered.

Idle time matters independently: it is what tells the attention gate whether
the user is at the keyboard (don't interrupt) or has been away for an hour
(a good moment to surface something that has been waiting).
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
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
]


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
        text = f"switched to {shown}" + (f" (was on {prev[:60]} for {dwell:.0f}s)" if prev else "")

        # A rapid flurry of switches is a different signal from settling into
        # one app for an hour -- salience tracks how unusual the change is.
        salience = 0.35 if dwell > 30 else 0.2

        return [SensorEvent(
            source=self.name, modality=Modality.TEXT, text=text,
            salience=salience,
            data={"title": None if blocked else title, "process": proc, "dwell": dwell},
            persistable=not blocked,
            meta={"process": proc, "blocked": blocked, "dwell_s": round(dwell, 1)},
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
                 max_frames: int = 500):
        self._grab = None
        self.window_sensor = window_sensor
        self.hamming_threshold = hamming_threshold
        self.save_frames = save_frames
        self.max_frames = max_frames
        self._last_hash: Optional[int] = None
        self._dropped = 0
        super().__init__(interval=interval, enabled=enabled)

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
        if self._grab == "mss":
            import mss
            from PIL import Image
            with mss.mss() as sct:
                mon = sct.monitors[1]
                shot = sct.grab(mon)
                return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        from PIL import ImageGrab
        return ImageGrab.grab()

    def read(self) -> List[SensorEvent]:
        # Blocklist check happens BEFORE capture -- see module docstring.
        title = ""
        if self.window_sensor is not None:
            title, _ = self.window_sensor.current()
            if self.window_sensor.is_blocked(title):
                self._dropped += 1
                return []

        img = self._capture()
        h = _ahash(img)

        if self._last_hash is not None and _hamming(h, self._last_hash) <= self.hamming_threshold:
            self._dropped += 1
            return []   # screen hasn't meaningfully changed
        self._last_hash = h

        path = None
        if self.save_frames:
            path = self._save(img)

        # Downscale before handing to the vision extractor: the ViT resizes to
        # 224px anyway, and a 4K frame in the event queue is pure waste.
        img.thumbnail((640, 400))

        return [SensorEvent(
            source=self.name, modality=Modality.VISION,
            text=f"screen changed while in {title[:60]}" if title else "screen changed",
            data=img, salience=0.4,
            meta={"ahash": h, "path": path, "dropped_since": self._dropped},
        )]

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
