"""
src/daemon/weekly.py

A weekly review, from what ASH already keeps:

    "This week: 9 hours on A.S.H and 2 on thesis. 3 follow-ups closed, 4 of
     my questions answered, 2 reminders. Omar came up most, then Sara."

    time on each project   activity memory's per-day seconds, last 7 days
    follow-ups closed      follow-ups answered or told about this week
    questions answered     wondering: questions closed as answered
    reminders              reminders and timers that went off
    who came up most       names you said, counted per day here (names
                           only, kept two weeks)

Asked for ("how was my week?", "weekly review", "what did I do this
week?"), and part of the Sunday evening briefing (`weekly.day` in
daemon.json; "" turns that off).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.daemon.weekly")

DAY = 86400.0
KEEP_DAYS = 14
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_ASK = re.compile(r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?ash[,\s]+)?(?:(?:can|could)\s+you\s+)?(?:give\s+me\s+)?"
                  r"(?:how\s+(?:was|has\s+been|did)\s+my\s+week(?:\s+go)?|how(?:'s|\s+has)\s+my\s+week\s+been|(?:my\s+|a\s+|the\s+)?week(?:ly)?\s+(?:in\s+)?review|"
                  r"what\s+did\s+i\s+(?:do|get\s+done|work\s+on)\s+this\s+week|"
                  r"(?:sum\s+up|summari[sz]e|recap)\s+(?:my\s+|the\s+|this\s+)?week|my\s+week\s+so\s+far)"
                  r"(?:\s+so\s+far)?\s*[?.!]*\s*$", re.I)


def _hours(seconds: float) -> str:
    h = seconds / 3600.0
    if h >= 1.5:
        return f"{h:.0f} hours"
    if h >= 0.75:
        return "an hour"
    return f"{max(1, round(seconds / 60.0))} minutes"


class Weekly:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 activity=None, followups=None, wonder=None, reminders=None, day: str = "sunday",
                 extra: Optional[Callable[[float], Optional[str]]] = None):
        self.path = path
        self.extra = extra          # one more line for the review from somewhere else (the browser ledger), or None
        self.clock = clock
        self.activity = activity
        self.followups = followups
        self.wonder = wonder
        self.reminders = reminders
        self.day = (day or "").strip().lower()
        self.people: Dict[str, Dict[str, int]] = {}     # date -> name -> count
        self._lock = threading.Lock()
        self._dirty = False
        self._load()

    # -- the names --------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.people = {str(d): {str(k): int(v) for k, v in (m or {}).items()}
                           for d, m in (raw.get("people") or {}).items()}
        except Exception:
            logger.warning("Could not read %s", self.path, exc_info=True)

    def save(self):
        if not self.path or not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with self._lock:
                data = {"people": self.people}
                self._dirty = False
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("Could not save %s", self.path, exc_info=True)

    def named(self, name: str, now: Optional[float] = None):
        """Someone came up in what the user said."""
        name = (name or "").strip()
        if not name:
            return
        now = self.clock() if now is None else now
        d = date.fromtimestamp(now).isoformat()
        cut = date.fromtimestamp(now - KEEP_DAYS * DAY).isoformat()
        with self._lock:
            day = self.people.setdefault(d, {})
            day[name] = day.get(name, 0) + 1
            for k in [k for k in self.people if k < cut]:
                del self.people[k]
            self._dirty = True

    # -- the review ---------------------------------------------------------
    def _since(self, now: float) -> float:
        return now - 7 * DAY

    def _week_dates(self, now: float) -> List[str]:
        today = date.fromtimestamp(now)
        return [(today - timedelta(days=i)).isoformat() for i in range(7)]

    def projects(self, now: float) -> List[tuple]:
        """[(name, seconds)] over the last 7 days, most first."""
        a = self.activity
        if a is None:
            return []
        days = set(self._week_dates(now))
        totals: Counter = Counter()
        with a._lock:
            topics = list(a.topics.values())
        have_projects = any(t.kind == "project" for t in topics)
        for t in topics:
            if have_projects and t.kind != "project":
                continue
            s = sum(v for d, v in (t.by_day or {}).items() if d in days)
            if s >= 600:
                totals[t.name] += s
        return totals.most_common()

    def counts(self, now: float) -> Dict[str, int]:
        since = self._since(now)
        out = {"followups": 0, "answered": 0, "reminders": 0}
        try:
            if self.followups is not None:
                with self.followups._lock:
                    out["followups"] = sum(1 for f in self.followups.items.values()
                                           if f.status in ("answered", "told") and f.closed_at >= since)
        except Exception:
            logger.debug("followups count failed", exc_info=True)
        try:
            if self.wonder is not None:
                with self.wonder._lock:
                    out["answered"] = sum(1 for w in self.wonder.wonders.values()
                                          if w.status == "answered" and w.closed_at >= since)
        except Exception:
            logger.debug("wonder count failed", exc_info=True)
        try:
            if self.reminders is not None:
                with self.reminders._lock:
                    out["reminders"] = sum(1 for r in self.reminders.items.values()
                                           if r.status == "fired" and r.fired_at >= since)
        except Exception:
            logger.debug("reminders count failed", exc_info=True)
        return out

    def top_people(self, now: float, n: int = 2) -> List[str]:
        days = set(self._week_dates(now))
        c: Counter = Counter()
        with self._lock:
            for d, m in self.people.items():
                if d in days:
                    c.update(m)
        return [k for k, v in c.most_common(n) if v >= 2]

    def compose(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        parts: List[str] = []
        proj = self.projects(now)
        if proj:
            bits = [f"{_hours(s)} on {name}" for name, s in proj[:3]]
            parts.append("This week: " + (", ".join(bits[:-1]) + " and " + bits[-1] if len(bits) > 1 else bits[0]) + ".")
        c = self.counts(now)
        done = []
        if c["followups"]:
            done.append(f"{c['followups']} follow-up{'s' if c['followups'] != 1 else ''} closed")
        if c["answered"]:
            done.append(f"{c['answered']} of my questions answered")
        if c["reminders"]:
            done.append(f"{c['reminders']} reminder{'s' if c['reminders'] != 1 else ''}")
        if done:
            s = ", ".join(done)
            parts.append(s[0].upper() + s[1:] + ".")
        ppl = self.top_people(now)
        if ppl:
            parts.append(f"{ppl[0]} came up most" + (f", then {ppl[1]}." if len(ppl) > 1 else "."))
        if self.extra is not None:
            try:
                more = self.extra(now)
                if more:
                    parts.append(more)
            except Exception:
                pass
        if not parts:
            return "A quiet week -- I don't have much on it yet."
        return " ".join(parts)

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        t = (text or "").strip()
        if not t or len(t) > 80 or not _ASK.match(t):
            return None
        return self.compose(now)

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        if kind != "evening" or self.day not in DAYS:
            return None
        if datetime.fromtimestamp(now).weekday() != DAYS.index(self.day):
            return None
        s = self.compose(now)
        return None if s.startswith("A quiet week") else s

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"day": self.day, "days_kept": len(self.people)}
