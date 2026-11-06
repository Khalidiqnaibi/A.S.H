"""
src/daemon/crashwatch.py

ASH noticing that the computer crashed.

Windows records every unexpected shutdown (Kernel-Power, event 41) with the
bug-check code when there was one. ASH reads that log in the background, at
start and every six hours after, and tells you about the ones it hasn't
told you about yet -- in the morning briefing, or when asked ("did my
computer crash?", "any crashes lately?"):

    The computer shut down unexpectedly at 05:33 this morning. That's 6
    times in the last 30 days, which is a pattern: worth testing the memory
    (run mdsched.exe) and updating the drivers.

It is a thing to see happening: 24 crashes since August had nothing in
common but the machine, and nobody was counting. Read-only; only the times
and codes are kept.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.daemon.crashwatch")

DAY = 86400.0
CODES = {
    0xA: "a driver touching memory it shouldn't (IRQL_NOT_LESS_OR_EQUAL)",
    0x12B: "corrupted memory (FAULTY_HARDWARE_CORRUPTED_PAGE)",
    0x1A: "a memory-management error",
    0x50: "a bad memory reference (PAGE_FAULT_IN_NONPAGED_AREA)",
    0x7E: "a system thread hitting an exception",
    0x3B: "a system service hitting an exception",
    0x124: "a hardware error (WHEA_UNCORRECTABLE_ERROR)",
    0x116: "the graphics driver stopped responding",
    0x133: "a driver taking too long (DPC_WATCHDOG_VIOLATION)",
}
_ASK = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:did|has)\s+(?:my\s+|the\s+)?(?:computer|laptop|pc|machine|system|it)\s+"
                  r"(?:crash(?:ed)?|(?:blue[\s-]?screen|bsod)(?:ed)?|restart(?:ed)?\s+(?:by\s+itself|on\s+its\s+own)|"
                  r"shut\s+down\s+(?:by\s+itself|unexpectedly))(?:\s+(?:lately|recently|again|today|last\s+night|"
                  r"this\s+(?:morning|week)))?|any\s+(?:crashes|blue\s+screens|bsods)(?:\s+(?:lately|recently|today))?|"
                  r"how\s+many\s+times\s+(?:has|did)\s+(?:my\s+|the\s+)?(?:computer|laptop|pc)\s+crash(?:ed)?"
                  r"(?:\s+(?:lately|recently))?)\s*[?.!]*\s*$", re.I)

_PS = ("Get-WinEvent -FilterHashtable @{LogName='System'; Id=41} -MaxEvents 300 -ErrorAction SilentlyContinue | "
       "ForEach-Object { [pscustomobject]@{ t = ([DateTimeOffset]$_.TimeCreated).ToUnixTimeSeconds(); "
       "code = [int64]$_.Properties[0].Value } } | ConvertTo-Json -Compress")


def read_windows_log(timeout: float = 25.0) -> Optional[List[Dict[str, int]]]:
    """[{"t": unix seconds, "code": bug-check or 0}] newest first; None if it can't be read."""
    if not sys.platform.startswith("win"):
        return None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS],
                           capture_output=True, text=True, timeout=timeout, creationflags=flags)
        out = (r.stdout or "").strip()
        if r.returncode != 0 or not out:
            return [] if r.returncode == 0 else None
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        return [{"t": int(x["t"]), "code": int(x.get("code") or 0)} for x in data]
    except Exception:
        logger.debug("Reading the Windows event log failed", exc_info=True)
        return None


def _when(t: float, now: float) -> str:
    d = (date.fromtimestamp(now) - date.fromtimestamp(t)).days
    clock = datetime.fromtimestamp(t).strftime("%H:%M")
    if d == 0:
        return f"at {clock} today"
    if d == 1:
        return f"at {clock} yesterday"
    return f"at {clock} on {datetime.fromtimestamp(t).strftime('%A')}" if d < 7 else \
        f"on {datetime.fromtimestamp(t).strftime('%d %b')} at {clock}"


class CrashWatch:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 reader: Callable[[], Optional[List[Dict[str, int]]]] = read_windows_log,
                 background: bool = True, refresh_s: float = 6 * 3600.0, enabled: bool = True):
        self.path = path
        self.clock = clock
        self.reader = reader
        self.background = background
        self.refresh_s = refresh_s
        self.enabled = enabled
        self.events: List[Dict[str, int]] = []
        self.read_at = -1e18
        self.told_until = 0.0              # crashes up to here have been told
        self._first = True
        self._inflight = False
        self._lock = threading.Lock()
        self.stats = {"reads": 0, "failed": 0}
        self._load()

    # -- keeping what was told ------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self.told_until = float(json.load(fh).get("told_until", 0.0))
                self._first = False
        except Exception:
            logger.warning("Could not read %s", self.path, exc_info=True)

    def _save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"told_until": self.told_until}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("Could not save %s", self.path, exc_info=True)

    # -- reading ---------------------------------------------------------------
    def refresh(self, now: Optional[float] = None, force: bool = False):
        if not self.enabled or self._inflight:
            return
        now = self.clock() if now is None else now
        if not force and now - self.read_at < self.refresh_s:
            return
        self.read_at = now
        self._inflight = True
        if self.background:
            threading.Thread(target=self._read, daemon=True, name="ash-crashwatch").start()
        else:
            self._read()

    def _read(self):
        try:
            got = self.reader()
            if got is None:
                self.stats["failed"] += 1
                return
            with self._lock:
                self.events = sorted(got, key=lambda e: -e["t"])
                if self._first:
                    # The first time ever: what is already in the log is
                    # history, not news. Say what the last month looked like
                    # when asked, but don't announce each old crash.
                    self.told_until = self.events[0]["t"] if self.events else self.clock()
                    self._first = False
                    self._save()
            self.stats["reads"] += 1
        except Exception:
            self.stats["failed"] += 1
            logger.exception("Crash watch failed")
        finally:
            self._inflight = False

    # -- saying ----------------------------------------------------------------
    def _recent(self, now: float, days: float = 30.0) -> List[Dict[str, int]]:
        with self._lock:
            return [e for e in self.events if now - e["t"] <= days * DAY]

    def _line(self, e: Dict[str, int], now: float, n_month: int) -> str:
        s = f"The computer shut down unexpectedly {_when(e['t'], now)}"
        why = CODES.get(e["code"])
        if e["code"] and why:
            s += f", on {why}"
        elif e["code"]:
            s += f" (stop code 0x{e['code']:X})"
        else:
            s += ", with no error recorded -- a freeze or a power loss"
        s += "."
        if n_month >= 2:
            s += f" That's {n_month} times in the last 30 days"
            s += ", which is a pattern: worth testing the memory (run mdsched.exe) and updating the drivers." \
                if n_month >= 3 else "."
        return s

    def unreported(self, now: float) -> List[Dict[str, int]]:
        with self._lock:
            return [e for e in self.events if e["t"] > self.told_until]

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        if not self.enabled or kind not in ("morning", "evening"):
            return None
        new = self.unreported(now)
        if not new:
            return None
        n = len(self._recent(now))
        text = self._line(new[0], now, n)
        if len(new) > 1:
            text = f"{len(new)} unexpected shutdowns since I last told you; the latest: " + text[0].lower() + text[1:]
        with self._lock:
            self.told_until = max(e["t"] for e in new)
        self._save()
        return text

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        t = (text or "").strip()
        if not self.enabled or not t or len(t) > 90 or not _ASK.match(t):
            return None
        now = self.clock() if now is None else now
        if not self.events and self.stats["reads"] == 0:
            if self.stats["failed"]:
                return "I can't read the Windows event log from here, so I can't tell."
            self.refresh(force=True)
            return "I haven't read the crash log yet -- give me a moment and ask again."
        recent = self._recent(now)
        if not recent:
            return "No unexpected shutdowns in the last 30 days."
        with self._lock:
            self.told_until = max(self.told_until, recent[0]["t"])
        self._save()
        return self._line(recent[0], now, len(recent))

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "events": len(self.events), "told_until": self.told_until, **self.stats}
