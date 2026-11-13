"""
src/daemon/desktop.py

Doing things on the desktop: opening, switching to and closing apps, and
running a routine the user kept.

    "open Outlook", "launch Spotify"     opened, no question asked
    "switch to Chrome", "go to Teams"    brought to the front if it is running
    "close Word"                         ASH asks first ("Close Word? Anything
                                         unsaved may be lost."), and only a
                                         yes within a minute does it; closing
                                         is a polite request to the app, never
                                         a kill, so the app can still ask to
                                         save
    "run my morning check"               the steps of a routine kept through
                                         wondering (routines.py), opened in order

Only names it knows as apps are taken (a built-in list plus `apps` in
daemon.json), so "open the readme" or "run the tests" still go to the brain
and its tools. Nothing is typed into windows and nothing is deleted. The
machine is reached through an `Executor`, so tests hand in their own.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.desktop")

# spoken name -> (what to start, the process it runs as). Several processes
# are separated by "|": the new Outlook for Windows runs as olk.exe, the
# classic one as outlook.exe, and either is "Outlook".
APPS: Dict[str, Tuple[str, str]] = {
    "outlook": ("outlook", "outlook.exe|olk.exe"), "teams": ("ms-teams:", "ms-teams.exe"),
    "chrome": ("chrome", "chrome.exe"), "google chrome": ("chrome", "chrome.exe"),
    "edge": ("msedge", "msedge.exe"), "firefox": ("firefox", "firefox.exe"), "opera": ("opera", "opera.exe"),
    # "open the browser": any of them counts as running; Chrome is what gets started (add desktop.apps to change it)
    "browser": ("chrome", "chrome.exe|msedge.exe|firefox.exe"), "web browser": ("chrome", "chrome.exe|msedge.exe|firefox.exe"),
    "the browser": ("chrome", "chrome.exe|msedge.exe|firefox.exe"),
    "word": ("winword", "winword.exe"), "excel": ("excel", "excel.exe"),
    "powerpoint": ("powerpnt", "powerpnt.exe"), "notepad": ("notepad", "notepad.exe"),
    "calculator": ("calc", "calculatorapp.exe"), "spotify": ("spotify:", "spotify.exe"),
    "slack": ("slack:", "slack.exe"), "discord": ("discord:", "discord.exe"), "zoom": ("zoommtg:", "zoom.exe"),
    "vs code": ("code", "code.exe"), "vscode": ("code", "code.exe"), "visual studio code": ("code", "code.exe"),
    "file explorer": ("explorer", "explorer.exe"), "explorer": ("explorer", "explorer.exe"),
    "terminal": ("wt", "windowsterminal.exe"), "paint": ("mspaint", "mspaint.exe"),
    "whatsapp": ("whatsapp:", "whatsapp.exe"), "telegram": ("tg:", "telegram.exe"),
    "obsidian": ("obsidian:", "obsidian.exe"), "pycharm": ("pycharm64", "pycharm64.exe"),
}

_OPEN = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:open|launch|start|fire up|bring up)\s+"
                   r"(?:up\s+)?(?:the\s+|my\s+)?(?P<t>[\w .+\-]{2,40}?)(?:\s+(?:app|application|program))?"
                   r"(?:\s+(?:please|for me|now))?\s*[.!]?\s*$", re.I)
_SWITCH = re.compile(r"^\s*(?:please\s+)?(?:switch|go|take me|change)\s+(?:back\s+)?to\s+(?:the\s+|my\s+)?"
                     r"(?P<t>[\w .+\-]{2,40}?)(?:\s+(?:app|window))?\s*[.!]?\s*$", re.I)
_CLOSE = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:close|quit|exit|shut down)\s+"
                    r"(?:the\s+|my\s+)?(?P<t>[\w .+\-]{2,40}?)(?:\s+(?:app|application|program|window))?"
                    r"(?:\s+(?:please|for me|now))?\s*[.!]?\s*$", re.I)
_ROUTINE = re.compile(r"^\s*(?:please\s+)?(?:run|do|start|open|launch)\s+(?:my|the|our)\s+(?P<t>[\w '\-]{2,40}?)"
                      r"(?:\s+routine)?\s*[.!]?\s*$", re.I)
_YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|do it|go ahead|close it|please)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip(" .!?"))


class Executor:
    """Does it on Windows. Every method returns True if it worked."""

    def open(self, target: str) -> bool:
        try:
            if sys.platform.startswith("win"):
                if target.endswith(":"):
                    os.startfile(target)                            # a protocol handler
                else:
                    subprocess.Popen(["cmd", "/c", "start", "", target], shell=False,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            exe = shutil.which(target)
            if exe:
                subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        except Exception:
            logger.exception("Could not open %s", target)
        return False

    def running(self, process: str) -> bool:
        try:
            import psutil
            p = process.lower()
            return any((x.info.get("name") or "").lower() == p for x in psutil.process_iter(["name"]))
        except Exception:
            return False

    def close(self, process: str) -> bool:
        """A polite close (WM_CLOSE via taskkill without /F): the app may
        still ask to save. Never a kill."""
        try:
            if sys.platform.startswith("win"):
                r = subprocess.run(["taskkill", "/IM", process], capture_output=True, timeout=10)
                return r.returncode == 0
            import psutil
            n = 0
            for x in psutil.process_iter(["name"]):
                if (x.info.get("name") or "").lower() == process.lower():
                    x.terminate()
                    n += 1
            return n > 0
        except Exception:
            logger.exception("Could not close %s", process)
            return False

    def focus(self, process: str) -> bool:
        if not sys.platform.startswith("win"):
            return False
        try:
            import ctypes
            import ctypes.wintypes as wt
            import psutil
            pids = {x.pid for x in psutil.process_iter(["name"])
                    if (x.info.get("name") or "").lower() == process.lower()}
            if not pids:
                return False
            user32 = ctypes.windll.user32
            found = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
            def cb(hwnd, _l):
                pid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in pids and user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd):
                    found.append(hwnd)
                    return False
                return True
            user32.EnumWindows(cb, 0)
            if not found:
                return False
            user32.ShowWindow(found[0], 9)            # SW_RESTORE
            return bool(user32.SetForegroundWindow(found[0]))
        except Exception:
            logger.exception("Could not switch to %s", process)
            return False


@dataclass
class DesktopConfig:
    enabled: bool = True
    confirm_close: bool = True
    confirm_window_s: float = 60.0
    apps: Dict[str, Any] = field(default_factory=dict)   # extra: name -> target or [target, process]

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DesktopConfig":
        c = cls()
        for k, v in (d or {}).items():
            if k == "apps" and isinstance(v, dict):
                c.apps = dict(v)
            elif k in ("enabled", "confirm_close"):
                setattr(c, k, bool(v))
            elif k == "confirm_window_s":
                try:
                    c.confirm_window_s = float(v)
                except (TypeError, ValueError):
                    pass
        return c


class Desktop:
    def __init__(self, cfg: Optional[DesktopConfig] = None, executor: Optional[Executor] = None,
                 knowledge=None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or DesktopConfig()
        self.ex = executor or Executor()
        self.kb = knowledge                       # wondering's Knowledge, for routines
        self.clock = clock
        self._pending: Optional[Tuple[str, str, float]] = None   # (name, process, asked at)
        self.stats = {"opened": 0, "closed": 0, "switched": 0, "routines": 0, "failed": 0}
        self.apps: Dict[str, Tuple[str, str]] = dict(APPS)
        for name, v in self.cfg.apps.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                self.apps[_norm(name)] = (str(v[0]), str(v[1]).lower())
            elif isinstance(v, str):
                self.apps[_norm(name)] = (v, os.path.basename(v).lower()
                                          if v.lower().endswith(".exe") else f"{_norm(name)}.exe")

    def _app(self, name: str) -> Optional[Tuple[str, str, str]]:
        n = _norm(name)
        if n in self.apps:
            t, p = self.apps[n]
            return name.strip(" .!?"), t, p
        return None

    def _routine(self, name: str) -> Optional[Dict[str, Any]]:
        if self.kb is None:
            return None
        try:
            return self.kb.procedure(re.sub(r"^(?:the|my|our)\s+", "", _norm(name)))
        except Exception:
            return None

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """A desktop command: returns what to say back, or None if the
        sentence is not one this handles."""
        if not self.cfg.enabled:
            return None
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t or len(t) > 80:
            return None

        # the answer to "Close Word?"
        p = self._pending
        if p is not None:
            self._pending = None
            if now - p[2] <= self.cfg.confirm_window_s and _YES.match(t):
                return self._do_close(p[0], p[1])
            if now - p[2] <= self.cfg.confirm_window_s and (re.match(r"^\s*(?:no|nope|nah|don'?t|cancel|leave it)\b",
                                                                      t, re.I) or _yesno.is_no(t)):
                return "Okay, I'll leave it open."

        m = _ROUTINE.match(t)
        if m:
            r = self._routine(m.group("t"))
            if r is not None:
                opened = [s for s in r["steps"] if self._open_one(s)]
                self.stats["routines"] += 1
                if not opened:
                    return f"I couldn't open anything for {r['name']}."
                return f"Running {r['name']}: opening " + ", then ".join(opened) + "."
        m = _OPEN.match(t)
        if m:
            app = self._app(m.group("t"))
            if app is None:
                return None
            return f"Opening {app[0]}." if self._open_one(m.group("t")) else f"I couldn't open {app[0]}."
        m = _SWITCH.match(t)
        if m:
            app = self._app(m.group("t"))
            if app is None:
                return None
            proc = self._running(app[2])
            if proc is None:
                return f"{app[0]} isn't open. Want me to open it? Say \"open {app[0]}\"."
            ok = self.ex.focus(proc)
            self.stats["switched" if ok else "failed"] += 1
            return f"Switching to {app[0]}." if ok else f"I couldn't bring {app[0]} to the front."
        m = _CLOSE.match(t)
        if m:
            app = self._app(m.group("t"))
            if app is None:
                return None
            proc = self._running(app[2])
            if proc is None:
                return f"{app[0]} isn't open."
            if self.cfg.confirm_close:
                self._pending = (app[0], proc, now)
                return f"Close {app[0]}? Anything unsaved there may be lost. Say yes to close it."
            return self._do_close(app[0], proc)
        return None

    def _running(self, procs: str) -> Optional[str]:
        """The one of an app's processes ("outlook.exe|olk.exe") that is running."""
        for p in (procs or "").split("|"):
            if p and self.ex.running(p):
                return p
        return None

    def _open_one(self, name: str) -> bool:
        app = self._app(name)
        target = app[1] if app else name
        ok = self.ex.open(target)
        self.stats["opened" if ok else "failed"] += 1
        return ok

    def _do_close(self, name: str, process: str) -> str:
        ok = self.ex.close(process)
        self.stats["closed" if ok else "failed"] += 1
        return f"Closing {name}." if ok else f"I couldn't close {name}."

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "apps": len(self.apps), **self.stats}


# Any ordinary yes (ye, yea, okay, do it, please...) answers a question, not only the words listed above (yesno.py).
from . import yesno as _yesno  # noqa: E402
_YES = _yesno.yes_or(_YES)
