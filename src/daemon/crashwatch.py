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

from src.memory.atomic import load_json, log_io_failure, write_json

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

_CAUSE = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:is|are|was|were)\s+(?:it\s+|you\s+)?(?:ash|you|a\.?s\.?h\.?)\s+"
                    r"(?:the\s+(?:one\s+)?(?:causing|cause\s+of)|causing|behind|responsible\s+for)\s+"
                    r"(?:the\s+|my\s+|these\s+)?(?:crash(?:es)?|blue\s+screens?|bsods?|shut\s*downs?|restarts?)|"
                    r"(?:did|do|does|could|can)\s+(?:ash|you)\s+(?:crash|cause|break)\s+(?:my\s+|the\s+)?"
                    r"(?:computer|laptop|pc|machine|system|crash(?:es)?|blue\s+screens?))\s*[?.!]*\s*$", re.I)

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
        # When ASH itself was running, so a crash can be set against it. Kept in
        # its own file: told_until above decides whether the first read is
        # "history", and writing it early would turn old crashes into news.
        self.runs_path = (path + ".runs") if path else None
        self.runs: List[Dict[str, float]] = []
        self._beat_saved = 0.0
        self._load_runs()

    # -- keeping what was told ------------------------------------------------
    def _load(self):
        if not self.path:
            return
        d = load_json(self.path, default=None, logger=logger)
        if isinstance(d, dict):
            try:
                self.told_until = float(d.get("told_until", 0.0))
                self._first = False
            except (TypeError, ValueError):
                logger.warning("Could not read %s", self.path)

    def _save(self):
        if not self.path:
            return
        try:
            write_json(self.path, {"told_until": self.told_until})
        except Exception as e:
            log_io_failure(logger, self.path, e)

    # -- when ASH was running ----------------------------------------------------
    def _load_runs(self):
        now = self.clock()
        if self.runs_path:
            d = load_json(self.runs_path, default=None, logger=logger)
            try:
                got = d.get("runs", []) if isinstance(d, dict) else []
                self.runs = [r for r in got if isinstance(r, dict) and "start" in r
                             and now - float(r.get("beat", r["start"])) <= 90 * DAY][-200:]
            except Exception:
                logger.warning("Could not read %s", self.runs_path, exc_info=True)
        if self.enabled and self.runs_path:
            self.runs.append({"start": now, "beat": now})
            self._save_runs(now)

    def _save_runs(self, now: float):
        if not self.runs_path:
            return
        self._beat_saved = now
        try:
            write_json(self.runs_path, {"runs": self.runs})
        except Exception as e:
            log_io_failure(logger, self.runs_path, e)

    def heartbeat(self, now: Optional[float] = None):
        """Called often; writes at most once a minute."""
        if not self.runs or not self.enabled:
            return
        now = self.clock() if now is None else now
        self.runs[-1]["beat"] = now
        if now - self._beat_saved >= 60.0:
            self._save_runs(now)

    def shutdown(self, now: Optional[float] = None):
        """A clean stop: this run ended on purpose, not with the machine."""
        if not self.runs or not self.enabled:
            return
        now = self.clock() if now is None else now
        self.runs[-1]["beat"] = self.runs[-1]["end"] = now
        self._save_runs(now)

    def _intervals(self, now: float) -> List[tuple]:
        """[(start, end)] for every run. One that never recorded a clean end was
        cut off -- and the crash is what cut it, so it counts as running until
        the next start (the Windows log stamps a crash at the reboot, not the
        moment)."""
        out = []
        for i, r in enumerate(self.runs):
            a = float(r["start"])
            if "end" in r:
                b = float(r["end"])
            elif i + 1 < len(self.runs):
                b = float(self.runs[i + 1]["start"])
            else:
                b = now
            out.append((a, max(a, b)))
        return out

    def overlap(self, now: float, days: float = 30.0) -> Optional[Dict[str, Any]]:
        """How many of the recent crashes ASH was running through, and how much
        of the time it runs at all. None until there is a day of history."""
        if not self.runs:
            return None
        iv = self._intervals(now)
        first = iv[0][0]
        if now - first < DAY:
            return None
        lo = max(now - days * DAY, first)
        crashes = [e for e in self._recent(now, days) if e["t"] >= first]
        with_ash = sum(1 for e in crashes if any(a <= e["t"] <= b for a, b in iv))
        ran = sum(max(0.0, min(b, now) - max(a, lo)) for a, b in iv)
        return {"crashes": len(crashes), "with_ash": with_ash, "share": min(1.0, ran / max(1.0, now - lo)),
                "days": (now - lo) / DAY}

    def _overlap_sentence(self, now: float) -> str:
        o = self.overlap(now)
        if not o or o["crashes"] < 2:
            return ""
        s = f"ASH was running during {o['with_ash']} of those {o['crashes']}"
        s += f", and it runs about {round(o['share'] * 100)}% of the time"
        if o["with_ash"] == o["crashes"] and o["share"] < 0.6:
            s += " -- more than chance would give, so it is worth switching me off for a few days to see"
        elif o["with_ash"] / o["crashes"] <= o["share"] + 0.15:
            s += " -- about what chance would give, so nothing here points at me"
        return s + "."

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
        if not self.enabled or not t or len(t) > 90 or not (_ASK.match(t) or _CAUSE.match(t)):
            return None
        now = self.clock() if now is None else now
        if _CAUSE.match(t):
            if not self.events and self.stats["reads"] == 0:
                self.refresh(force=True)
                return "I haven't read the crash log yet -- give me a moment and ask again."
            if not self._recent(now):
                return "No unexpected shutdowns in the last 30 days, so there is nothing to blame me for."
            o = self.overlap(now)
            if o is None:
                return ("I only started keeping track of when I'm running recently, so I can't tell yet -- ask again "
                        "in a few days.")
            if o["crashes"] < 2:
                return "Only one crash since I started counting, which is too few to say anything about me."
            return (self._overlap_sentence(now) + " I can only show a link, not a cause: a crash this deep (a "
                    "driver or the hardware) takes everything down, whatever was running.")
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
        line = self._line(recent[0], now, len(recent))
        extra = self._overlap_sentence(now)
        return line + (" " + extra if extra else "")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "events": len(self.events), "told_until": self.told_until,
                    "runs": len(self.runs), **self.stats}
