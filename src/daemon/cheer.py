"""
src/daemon/cheer.py

Remembering the good moments, and celebrating -- BMO's side of ASH.

Good moments
    A follow-up that ended well ("how did the interview go?" -- "great!") is
    a good moment. When the user later says they are nervous or worried
    about something, ASH brings one back:
        "You've done this before -- remember your interview in March? That
         went well."
    the same kind of thing first (an exam for an exam), else the latest.
    "Tell me something good" lists the recent ones.

Celebrations (said at a free moment, never more than one a day of each)
    birthday      the stored fact ("my birthday is March 3"), on the day; also
                  first thing in the morning briefing
    streaks       5, 7, 10, 14, 21, 30... days in a row on the same project
    long stretch  two hours or more of unbroken work on one thing, finished

Rule-based throughout. What was celebrated is kept, so a restart does not
celebrate it twice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .freemoment import Activity, Candidate

logger = logging.getLogger("ash.daemon.cheer")

DAY = 86400.0
STREAKS = (5, 7, 10, 14, 21, 30, 50, 75, 100)
LONG_S = 2 * 3600.0
GAP_S = 180.0                       # pieces of one stretch closer than this are one stretch

_WORRY = re.compile(r"\b(?:i'?m|i\s+am|feeling|feel|so|really|a\s+bit|kind\s+of|kinda)\s+(?:so\s+|really\s+|a\s+bit\s+|"
                    r"kind\s+of\s+|kinda\s+|pretty\s+)?(?:nervous|anxious|worried|scared|stressed|terrified|"
                    r"freaking\s+out)\b(?:\s+(?:about|for|before)\s+(?P<what>[^.?!]{2,60}))?", re.I)
_GOOD = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:tell\s+me\s+something\s+good|remind\s+me\s+(?:of\s+)?something\s+good|"
                   r"what(?:'s|\s+has)?\s+(?:went|gone)\s+well(?:\s+lately|\s+recently)?|"
                   r"cheer\s+me\s+up|what\s+good\s+things\s+happened(?:\s+lately)?)\s*[?.!]*\s*$", re.I)

_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                       "dec"), 1)}


def parse_day(text: str) -> Optional[Tuple[int, int]]:
    """(month, day) from "March 3", "3rd of March", "3 Mar", "03/03" (day
    first) or "2026-03-03"; None if it can't be read."""
    t = (text or "").lower()
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        mo, d = int(m.group(2)), int(m.group(3))
        return (mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else None
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m:
        return (_MONTHS[m.group(1)], int(m.group(2))) if 1 <= int(m.group(2)) <= 31 else None
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", t)
    if m:
        return (_MONTHS[m.group(2)], int(m.group(1))) if 1 <= int(m.group(1)) <= 31 else None
    m = re.search(r"\b(\d{1,2})[/.](\d{1,2})\b", t)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        return (mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else None
    return None


def _when(t: float, now: float) -> str:
    days = (date.fromtimestamp(now) - date.fromtimestamp(t)).days
    if days <= 1:
        return "yesterday" if days == 1 else "today"
    if days < 7:
        return "on " + datetime.fromtimestamp(t).strftime("%A")
    if days < 14:
        return "last week"
    return "in " + datetime.fromtimestamp(t).strftime("%B")


class Cheer:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 followups=None, knowledge=None, activity=None):
        self.path = path
        self.clock = clock
        self.followups = followups
        self.kb = knowledge
        self.activity = activity
        self.done: Dict[str, float] = {}             # what was celebrated -> when
        self.pending: List[Tuple[str, str]] = []     # (key, line) to say at a free moment
        self._run: Optional[Dict[str, Any]] = None   # the stretch being added up: name, start, end
        self._lock = threading.RLock()
        self.stats = {"recalled": 0, "celebrated": 0}
        self._load()

    # -- persistence ------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self.done = {str(k): float(v) for k, v in (json.load(fh).get("done") or {}).items()}
        except Exception:
            logger.warning("Could not read %s", self.path, exc_info=True)

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            now = self.clock()
            with self._lock:
                keep = {k: v for k, v in self.done.items() if now - v <= 400 * DAY}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"done": keep}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("Could not save %s", self.path, exc_info=True)

    # -- good moments -------------------------------------------------------------
    def good_moments(self, now: float, within_days: float = 365) -> List[Dict[str, Any]]:
        f = self.followups
        if f is None:
            return []
        try:
            with f._lock:
                rows = [x for x in f.items.values() if x.outcome == "good" and x.status in ("answered", "told")
                        and x.closed_at and now - x.closed_at <= within_days * DAY]
        except Exception:
            return []
        return [{"what": x.what, "noun": x.noun, "t": x.closed_at}
                for x in sorted(rows, key=lambda x: -x.closed_at)]

    def recall(self, about: str, now: float) -> Optional[str]:
        """A good moment to bring back to someone worried about `about`."""
        rows = self.good_moments(now)
        if not rows:
            return None
        about = (about or "").lower()
        same = [r for r in rows if r["noun"] and re.search(r"\b" + re.escape(r["noun"].lower()) + r"s?\b", about)]
        r = (same or rows)[0]
        self.stats["recalled"] += 1
        return f"You've done this before -- remember {r['what']} {_when(r['t'], now)}? That went well."

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t or len(t) > 160:
            return None
        if _GOOD.match(t):
            rows = self.good_moments(now, within_days=90)
            if not rows:
                return None                              # nothing on record: the brain answers
            items = [f"{r['what']} {_when(r['t'], now)}" for r in rows[:3]]
            self.stats["recalled"] += 1
            return "Some good ones: " + "; ".join(items) + " -- all went well."
        m = _WORRY.search(t)
        if m:
            return self.recall(m.group("what") or t, now)
        return None

    # -- celebrations ---------------------------------------------------------
    def _celebrated(self, key: str) -> bool:
        with self._lock:
            return key in self.done

    def _queue(self, key: str, line: str, now: float):
        with self._lock:
            if key in self.done or any(k == key for k, _ in self.pending):
                return
            self.pending.append((key, line))

    def birthday_today(self, now: float) -> bool:
        if self.kb is None:
            return False
        try:
            md = parse_day(self.kb.fact("birthday") or "")
        except Exception:
            return False
        d = date.fromtimestamp(now)
        return md is not None and md == (d.month, d.day)

    def stretch(self, name: str, start: float, end: float):
        """A stretch of work recorded (activity memory's pieces, joined up)."""
        if not name:
            return
        with self._lock:
            r = self._run
            if r is not None and r["name"] == name and start - r["end"] <= GAP_S:
                r["end"] = max(r["end"], end)
                return
            self._finish_run()
            self._run = {"name": name, "start": start, "end": end}

    def _finish_run(self):
        r, self._run = self._run, None
        if r is None or r["end"] - r["start"] < LONG_S:
            return
        hours = (r["end"] - r["start"]) / 3600.0
        h = f"{hours:.1f}".rstrip("0").rstrip(".")
        key = f"stretch:{r['name']}:{date.fromtimestamp(r['end']).isoformat()}"
        self._queue(key, f"That was a solid {h} hours on {r['name']}. Nice work.", r["end"])

    def _streaks(self, now: float):
        a = self.activity
        if a is None:
            return
        today = date.fromtimestamp(now)
        try:
            with a._lock:
                topics = [t for t in a.topics.values() if t.kind == "project"]
        except Exception:
            return
        for t in topics:
            days = set(k for k, v in (t.by_day or {}).items() if v >= 900)
            n, d = 0, today
            while d.isoformat() in days:
                n += 1
                d -= timedelta(days=1)
            hit = max([s for s in STREAKS if s <= n], default=0)
            if hit:
                self._queue(f"streak:{t.name}:{hit}:{(today - timedelta(days=n - 1)).isoformat()}",
                            f"That's {hit} days in a row on {t.name}. Nice streak.", now)

    def tick(self, now: Optional[float] = None):
        now = self.clock() if now is None else now
        with self._lock:
            if self._run is not None and now - self._run["end"] > GAP_S * 2:
                self._finish_run()
        if self.birthday_today(now):
            d = date.fromtimestamp(now)
            self._queue(f"birthday:{d.year}", "Happy birthday! I hope it's a good one.", now)
        self._streaks(now)

    def next_line(self) -> Optional[Tuple[str, str]]:
        with self._lock:
            return self.pending[0] if self.pending else None

    def said(self, key: str, now: float):
        with self._lock:
            self.pending = [p for p in self.pending if p[0] != key]
            self.done[key] = now
        self.stats["celebrated"] += 1
        self.save()

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        if kind != "morning" or not self.birthday_today(now):
            return None
        key = f"birthday:{date.fromtimestamp(now).year}"
        if self._celebrated(key):
            return None
        self.said(key, now)
        return "Happy birthday!"

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"pending": len(self.pending), "celebrated_keys": len(self.done), **self.stats}


class Celebrate(Activity):
    """A celebration waiting to be said, at the next free moment."""
    name = "celebrate"
    weight = 1.2
    cooldown_s = 600.0

    def __init__(self, cheer: Cheer):
        self.cheer = cheer
        self._key: Optional[str] = None

    def propose(self, env, now):
        nxt = self.cheer.next_line()
        if nxt is None:
            return None
        if env.mood is not None and env.mood < -0.3:
            return None                                   # not to someone having a hard time
        self._key = nxt[0]
        return Candidate(self.name, nxt[1], 0.7)

    def spoken(self, now):
        if self._key:
            self.cheer.said(self._key, now)
            self._key = None
