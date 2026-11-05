"""
src/daemon/routines.py

Noticing a routine, and asking whether to keep it.

Most people start a working session the same way: mail, then chat, then the
ticket board. Done by hand every morning, it is exactly the kind of thing an
assistant should offer to take over -- but only once it has seen it, and only
if the user says it is a routine and not a coincidence.

    watch    the first few apps opened at the start of a sitting (after being
             away, or at the start of a day), in order, de-duplicated
    notice   an opening order that repeats on several different days (any
             shorter opening of a longer one counts towards it too)
    ask      once, at a free moment, through wondering: "You often start your
             morning by opening Outlook, then Teams, then Jira. Want me to
             remember that as a routine?" A yes, with or without a name for
             it, keeps the steps in Knowledge (and entity memory), where
             running them by name can find them.

Only app names are kept (from the window's process), never titles or what
was on the screen. ASH's own window, the shell and system windows do not
count as steps. The state file lives under state/routines/.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .social import friendly_app

logger = logging.getLogger("ash.daemon.routines")

# Windows that are not a step of anything: the shell, ASH itself, system UI.
IGNORE = frozenset(
    "explorer shellexperiencehost searchhost searchapp startmenuexperiencehost lockapp applicationframehost "
    "taskmgr task manager systemsettings settings ash python pythonw ashd claude textinputhost "
    "rundll32 dwm ctfmon".split())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


@dataclass
class RoutinesConfig:
    enabled: bool = True
    away_s: float = 1800.0          # this long without a window change starts a new sitting
    window_s: float = 600.0         # the opening of a sitting is its first ten minutes
    min_steps: int = 2
    max_steps: int = 4
    min_days: int = 3               # seen on this many different days before it is asked about
    ask: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RoutinesConfig":
        c = cls()
        for k, v in (d or {}).items():
            if not hasattr(c, k):
                continue
            cur = getattr(c, k)
            try:
                if isinstance(cur, bool):
                    setattr(c, k, bool(v))
                elif isinstance(cur, int):
                    setattr(c, k, int(v))
                else:
                    setattr(c, k, float(v))
            except (TypeError, ValueError):
                pass
        return c


class Routines:
    def __init__(self, cfg: Optional[RoutinesConfig] = None, path: Optional[str] = None,
                 wonder=None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or RoutinesConfig()
        self.path = path
        self.wonder = wonder
        self.clock = clock
        self._lock = threading.RLock()
        self.seen: Dict[str, Dict[str, Any]] = {}     # "A > B > C" -> {"steps", "days", "hours", "asked"}
        self._last_event = -1e18
        self._seq: List[str] = []
        self._seq_start = 0.0
        self._seq_done = True
        self.stats = {"sittings": 0, "recorded": 0, "asked": 0}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.seen = {k: v for k, v in (raw.get("seen") or {}).items() if isinstance(v, dict)}
            self.stats.update({k: int(v) for k, v in (raw.get("stats") or {}).items() if k in self.stats})
        except Exception:
            logger.exception("Routine state unreadable; starting empty")
            try:
                os.replace(self.path, self.path + ".bad")
            except Exception:
                pass
            self.seen = {}

    def save(self):
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "seen": self.seen, "stats": self.stats}, fh, indent=1)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("Could not save routines")

    # -- watching ------------------------------------------------------------
    def observe(self, ev, now: Optional[float] = None) -> None:
        if not self.cfg.enabled:
            return
        src = str(getattr(ev, "source", "") or "").lower()
        src = src[4:] if src.startswith("sim:") else src
        if src != "active_window":
            return
        meta = getattr(ev, "meta", None) or {}
        if meta.get("blocked"):
            return
        now = self.clock() if now is None else now
        app = friendly_app(str(meta.get("process") or ""))
        stem = re.sub(r"\.(?:exe|app|bin)$", "", str(meta.get("process") or "").strip().lower())
        with self._lock:
            self.tick(now)
            new_day = (datetime.fromtimestamp(now).date()
                       != datetime.fromtimestamp(self._last_event).date()) if self._last_event > 0 else True
            if now - self._last_event >= self.cfg.away_s or new_day:
                self._seq, self._seq_start, self._seq_done = [], now, False
                self.stats["sittings"] += 1
            self._last_event = now
            # (the tick above has already closed an opening whose window passed)
            if not app or stem in IGNORE or _norm(app) in IGNORE or self._seq_done:
                return
            if app not in self._seq and len(self._seq) < self.cfg.max_steps:
                self._seq.append(app)

    def tick(self, now: Optional[float] = None) -> None:
        """Close the opening of a sitting once its window has passed."""
        if not self.cfg.enabled:
            return
        now = self.clock() if now is None else now
        with self._lock:
            if not self._seq_done and now - self._seq_start > self.cfg.window_s:
                self._seq_done = True
                if len(self._seq) >= self.cfg.min_steps:
                    self._record(list(self._seq), self._seq_start, now)

    def _record(self, seq: List[str], start: float, now: float):
        day = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        hour = datetime.fromtimestamp(start).hour
        for n in range(self.cfg.min_steps, len(seq) + 1):
            steps = seq[:n]
            key = " > ".join(steps)
            r = self.seen.setdefault(key, {"steps": steps, "days": [], "hours": [], "asked": False})
            if day not in r["days"]:
                r["days"] = (r["days"] + [day])[-30:]
                r["hours"] = (r["hours"] + [hour])[-30:]
        if len(self.seen) > 300:
            for k in sorted(self.seen, key=lambda k: len(self.seen[k]["days"]))[:len(self.seen) - 300]:
                del self.seen[k]
        self.stats["recorded"] += 1
        self._maybe_ask(now)
        self.save()

    def candidate(self) -> Optional[Dict[str, Any]]:
        """The routine most worth asking about: seen on enough days, and the
        longest of those (a longer opening says more than its own start)."""
        ok = [r for r in self.seen.values()
              if len(r["days"]) >= self.cfg.min_days and not r.get("asked")]
        if not ok:
            return None
        return max(ok, key=lambda r: (len(r["steps"]), len(r["days"])))

    def _maybe_ask(self, now: float):
        if not self.cfg.ask or self.wonder is None:
            return
        r = self.candidate()
        if r is None:
            return
        steps = r["steps"]
        hours = r.get("hours") or [12]
        when = "your morning" if sum(hours) / len(hours) < 12 else "a session"
        listed = ", then ".join(steps)
        q = (f"You often start {when} by opening {listed}. "
             f"Want me to remember that as a routine?")
        try:
            w = self.wonder.raise_wonder("routine", " then ".join(steps), q,
                                         key="routine:" + _norm(" ".join(steps)),
                                         data={"steps": list(steps)}, now=now, source="routine")
        except Exception:
            logger.exception("Could not raise a routine question")
            return
        # Whatever came of it (asked now, or already closed), every opening
        # that is part of this one has been considered.
        for k, rr in self.seen.items():
            if rr["steps"] == steps[:len(rr["steps"])]:
                rr["asked"] = True
        if w is not None:
            self.stats["asked"] += 1

    # -- inspection ----------------------------------------------------------
    def report(self) -> str:
        with self._lock:
            rows = sorted(self.seen.values(), key=lambda r: (-len(r["days"]), -len(r["steps"])))[:15]
            lines = [f"Routines: {'on' if self.cfg.enabled else 'off'}. {len(self.seen)} openings seen."]
            for r in rows:
                lines.append(f"  {len(r['days'])}d  {' > '.join(r['steps'])}{'  (asked)' if r.get('asked') else ''}")
            return "\n".join(lines)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.cfg.enabled, "openings": len(self.seen), **self.stats}

    def forget(self):
        with self._lock:
            self.seen = {}
            self.save()
