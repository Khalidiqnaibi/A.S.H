"""
src/daemon/dwell.py

Where the user is, and for how long: the window in front, whether they are typing, reading, or gone, and what the day
has been spent on.

ASH has always been told which window has focus and when the keyboard goes quiet. What it did with that was thin: a
free moment needed two minutes of steady typing followed by a short pause, so reading a long page, a PDF, or a set of
drawings -- no typing at all -- was never a moment. Being gone for an hour went unremarked. And nothing the user said
to ASH was ever read against what was on their screen.

This keeps, by rule and without a model:

  NOW        the program and the window in front, since when; whether they are TYPING (input in the last few seconds),
             PAUSING (a short gap), READING (the same window for minutes with no input, still there), or AWAY
             (no input for a long while: ten minutes by default, so a long read is not mistaken for leaving).

  BACK       when someone who was away returns: how long they were gone, once, so ASH can say so.

  READING    a stretch in one window with hands off the keyboard, long enough to be a thing in itself ("you've been on
             the body blueprints for 20 minutes"): the moment for a question about what they are looking at.

  TODAY      the day's time by program and by window, kept in state/dwell.json: "what have I been doing?", "how long
             was I on that?", and a line for the model ("they have been on X for 12 minutes, before that Y").

Private windows (a chat, a mailbox) and windows the privacy rules withhold are counted as a program only: their
titles are never kept, never said, never given to the model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.memory.atomic import load_json, log_io_failure, write_json

from .social import friendly_app

logger = logging.getLogger("ash.daemon.dwell")

_QUESTIONS = [
    ("here", re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?how\s+long\s+(?:have\s+i\s+been|am\s+i\s+been|i'?ve\s+been)\s+(?:on|in|at|using|looking\s+at|reading|here|on\s+this|in\s+this)"
                        r"(?:\s+(?:this|that|it|here))?(?:\s+(?:window|page|document|screen|app|program))?\W*$", re.I)),
    ("doing", re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?what\s+(?:have\s+i\s+been|was\s+i|am\s+i|did\s+i\s+(?:just\s+)?(?:do|work\s+on)|have\s+i\s+been)\s+"
                         r"(?:doing|working\s+on|up\s+to|looking\s+at|reading)(?:\s+(?:today|this\s+morning|this\s+afternoon|so\s+far|just\s+now|lately|earlier))?\W*$"
                         r"|^\s*(?:(?:hey\s+)?ash[,\s]+)?what\s+(?:did|have)\s+i\s+(?:do|done|work\s+on)\s+(?:today|so\s+far|this\s+morning)\W*$", re.I)),
    ("away", re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:how\s+long\s+was\s+i\s+(?:gone|away|out)|when\s+did\s+i\s+(?:leave|go\s+away|get\s+back|come\s+back)|"
                        r"was\s+i\s+(?:gone|away)\s+long)\W*$", re.I)),
]


@dataclass
class Back:
    away_s: float
    since: float
    at: float


def _span(seconds: float) -> str:
    m = int(round(seconds / 60.0))
    if m < 1:
        return "under a minute"
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    h, r = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''}" + (f" {r} minutes" if r else "")


class Dwell:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, away_s: float = 600.0,
                 reading_min_s: float = 240.0, pause_s: float = 6.0, reading_idle_s: float = 25.0):
        self.path = path
        self.clock = clock
        self.away_s = float(away_s)                 # no input this long: gone, not reading
        self.reading_min_s = float(reading_min_s)   # in one window this long with no typing: a thing in itself
        self.pause_s = float(pause_s)
        self.reading_idle_s = float(reading_idle_s)
        self.process = ""
        self.subject = ""
        self.project = ""
        self.private = False
        self.since = 0.0
        self._last_tick = 0.0
        self._idle = 0.0
        self.away_from: Optional[float] = None      # when the input stopped, once it has been long enough to count as away
        self.last_back: Optional[Back] = None
        self.last_away_s = 0.0
        self.day: Dict[str, Dict[str, float]] = {}  # "YYYY-MM-DD" -> {"app|subject": seconds}
        self.recent: List[Tuple[float, float, str, str]] = []   # (start, end, app, subject), newest last
        self._lock = threading.RLock()
        self._saved = 0.0
        self._load()

    # -- feeding it ------------------------------------------------------------------------------------------
    def window(self, process: str, subject: str = "", project: str = "", now: Optional[float] = None,
               private: bool = False, blocked: bool = False) -> None:
        """The window in front changed."""
        now = self.clock() if now is None else now
        with self._lock:
            self._close(now)
            self.process = str(process or "")
            hide = private or blocked
            self.subject = "" if hide else str(subject or "")[:80]
            self.project = "" if hide else str(project or "")[:60]
            self.private = bool(hide)
            self.since = now

    def tick(self, now: Optional[float], idle_s: Optional[float]) -> Optional[Back]:
        """Once per loop tick. Returns a Back the moment someone who was away starts typing again."""
        now = self.clock() if now is None else now
        if idle_s is None:
            return None
        back = None
        with self._lock:
            self._idle, self._last_tick = float(idle_s), now
            if idle_s >= self.away_s:
                if self.away_from is None:
                    self.away_from = now - idle_s
            elif self.away_from is not None and idle_s < self.pause_s * 2:
                start = self.away_from
                gone = now - start
                back = Back(away_s=gone, since=start, at=now)
                self.last_back, self.last_away_s = back, gone
                self.away_from = None
                if self.process and self.since:
                    # the time gone is not time on the window that was left open
                    self._flush(start)
                    self.since = now
            if now - self._saved >= 60.0:
                self.save()
        return back

    # -- the state -------------------------------------------------------------------------------------------
    def mode(self, now: Optional[float] = None, idle_s: Optional[float] = None) -> str:
        """typing | pausing | reading | away | unknown"""
        now = self.clock() if now is None else now
        idle = self._idle if idle_s is None else idle_s
        if not self._last_tick and idle_s is None:
            return "unknown"
        if idle >= self.away_s:
            return "away"
        if idle < 5.0:
            return "typing"
        if idle < self.reading_idle_s:
            return "pausing"
        return "reading"

    def minutes_here(self, now: Optional[float] = None) -> float:
        now = self.clock() if now is None else now
        return max(0.0, (now - self.since) / 60.0) if self.since else 0.0

    def reading(self, now: Optional[float] = None, idle_s: Optional[float] = None) -> bool:
        """Still here, hands off the keyboard, in the same window for long enough to count."""
        now = self.clock() if now is None else now
        if not self.process or self.private or not self.since:
            return False
        return self.mode(now, idle_s) == "reading" and (now - self.since) >= self.reading_min_s

    def where(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        return {"app": friendly_app(self.process), "subject": self.subject, "project": self.project, "private": self.private,
                "minutes": round(self.minutes_here(now), 1), "mode": self.mode(now)}

    # -- the day ---------------------------------------------------------------------------------------------
    def _flush(self, end: float) -> None:
        """Count the stretch on the current window up to `end`."""
        if self.process and self.since and end - self.since >= 5.0:
            keep, self.away_from = self.away_from, None
            try:
                self._record(self.since, end)
            finally:
                self.away_from = keep

    def _record(self, start: float, end: float) -> None:
        app = friendly_app(self.process)
        day = time.strftime("%Y-%m-%d", time.localtime(start))
        key = f"{app}|{self.subject}"
        d = self.day.setdefault(day, {})
        d[key] = d.get(key, 0.0) + (end - start)
        self.recent.append((start, end, app, self.subject))
        del self.recent[:-40]
        for old in sorted(self.day)[:-7]:
            self.day.pop(old, None)

    def _close(self, now: float) -> None:
        if not self.process or not self.since:
            return
        end = self.away_from if self.away_from is not None else now
        if end - self.since >= 5.0:
            self._record(self.since, end)

    def today(self, now: Optional[float] = None, include_current: bool = True) -> List[Tuple[str, str, float]]:
        now = self.clock() if now is None else now
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        with self._lock:
            rows = dict(self.day.get(day, {}))
            if include_current and self.process and self.since and self.mode(now) != "away":
                k = f"{friendly_app(self.process)}|{self.subject}"
                rows[k] = rows.get(k, 0.0) + max(0.0, now - self.since)
        out = []
        for k, sec in rows.items():
            app, subj = k.split("|", 1)
            out.append((app, subj, sec))
        return sorted(out, key=lambda r: -r[2])

    # -- what is said -----------------------------------------------------------------------------------------
    def context(self, now: Optional[float] = None) -> str:
        """A line for the model, or ''. Never the title of a private window."""
        now = self.clock() if now is None else now
        if not self.process or not self.since:
            return ""
        app = friendly_app(self.process)
        mode = self.mode(now)
        if mode == "away":
            return ""
        what = app if (self.private or not self.subject) else f"\"{self.subject}\" in {app}"
        line = f"[The user has been on {what} for {_span(now - self.since)}" + (", not typing -- reading or watching" if mode == "reading" else "") + "."
        rows = [r for r in self.today(now) if not (r[0] == app and r[1] == self.subject)][:2]
        if rows:
            line += " Earlier today: " + "; ".join(f"{(s or a)} {_span(sec)}" for a, s, sec in rows) + "."
        return line + "]"

    def parse(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if len(t) > 100:
            return None
        for kind, rx in _QUESTIONS:
            if rx.match(t):
                return kind
        return None

    def answer(self, kind: str, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        if kind == "here":
            if not self.process or not self.since:
                return "I haven't seen you switch windows yet."
            app = friendly_app(self.process)
            on = app if (self.private or not self.subject) else f"{self.subject} ({app})"
            return f"About {_span(now - self.since)} on {on}" + (", and you haven't touched the keyboard for a while." if self.mode(now) == "reading" else ".")
        if kind == "away":
            if self.away_from is not None:
                return f"You've been away about {_span(now - self.away_from)} -- or just reading; I can't tell which."
            if self.last_back is None:
                return "You haven't been away for long since I started."
            return f"You were away about {_span(self.last_back.away_s)}, and came back {_span(now - self.last_back.at)} ago."
        rows = self.today(now)
        rows = [r for r in rows if r[2] >= 60.0][:4]
        if not rows:
            return "Not much yet today that I've seen."
        parts = []
        for app, subj, sec in rows:
            parts.append(f"{subj} in {app} ({_span(sec)})" if subj else f"{app} ({_span(sec)})")
        return "So far today: " + "; ".join(parts) + "."

    def welcome_back(self, back: Back) -> str:
        return f"Welcome back. You were away about {_span(back.away_s)}."

    # -- persistence ------------------------------------------------------------------------------------------
    def save(self) -> None:
        self._saved = self.clock()
        if not self.path:
            return
        try:
            write_json(self.path, {"v": 1, "day": self.day})
        except Exception as e:
            log_io_failure(logger, "the dwell record", e)

    def _load(self) -> None:
        if not self.path:
            return
        d = load_json(self.path, default=None, logger=logger)      # the file, else its backup, else nothing
        if not isinstance(d, dict):
            return
        try:
            self.day = {str(k): {str(a): float(b) for a, b in v.items()} for k, v in d.get("day", {}).items()}
        except Exception:
            logger.exception("Dwell record unreadable; starting fresh")
            self.day = {}

    def forget(self) -> None:
        with self._lock:
            self.day, self.recent = {}, []
            self.save()

    def status(self) -> Dict[str, Any]:
        return {**self.where(), "away": self.away_from is not None, "days": len(self.day)}


# -- a free-moment remark for a long read ---------------------------------------------------------------------
from .freemoment import Activity, Candidate  # noqa: E402

_YES = re.compile(r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|please|go\s+ahead|do\s+it|alright)\b", re.I)
_NO = re.compile(r"^\s*(?:no|nope|nah|not\s+now|later|no\s+thanks?|i'?m\s+fine)\b", re.I)


class ReadingNote(Activity):
    """Something to say when the user has been looking at one thing for a long while with their hands off the keyboard:
    ASH noticed, says what it is, and offers to keep a note. By rule, from the window title and the clock."""
    name = "reading_note"
    weight = 1.0
    cooldown_s = 90 * 60.0

    def __init__(self, dwell: Dwell):
        self.dwell = dwell

    def propose(self, env, now):
        d = self.dwell
        if env.idle_s is None or not d.reading(now, env.idle_s) or d.private:
            return None
        mins = d.minutes_here(now)
        if mins < 8:
            return None
        what = f"\"{d.subject}\"" if d.subject else friendly_app(d.process)
        return Candidate(self.name, f"You've been on {what} for {_span(now - d.since)} without typing. Anything from it you want me to keep a note of?",
                         value=min(0.6, 0.3 + mins / 120.0))

    def hear(self, text, now):
        if _YES.match(text or ""):
            return "Go ahead -- say \"remember that\" and what, and I'll keep it."
        if _NO.match(text or ""):
            return "Okay."
        return None


# Any ordinary yes (ye, yea, okay, do it, please...) answers a question, not only the words listed above (yesno.py).
from . import yesno as _yesno  # noqa: E402
_YES = _yesno.yes_or(_YES)
_NO = _yesno.no_or(_NO)
