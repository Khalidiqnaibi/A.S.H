"""
src/daemon/briefing.py

Morning and evening briefings.

One short, spoken summary at the edges of the day, instead of a dozen
scattered remarks:

    morning   the first time the user is at the computer in the morning:
              what is on today (the events they mentioned, reminders due),
              what they were working on yesterday, and whether ASH has
              questions waiting
    evening   at a free moment in the evening: what the day went on, and
              what is on tomorrow

and on request ("brief me", "what's my day look like", "what did I do
today"). Each part comes from something ASH already keeps -- follow-ups for
events, activity memory for what was worked on, wondering for open
questions -- plus any number of extra providers (reminders, weather) handed
in as functions. A part with nothing to say is left out; a briefing with
nothing in it is not given at all. Rule-based: no model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.briefing")

Provider = Callable[[str, float], Optional[str]]     # (kind, now) -> a sentence or None

_ASKED = re.compile(
    r"\b(?:brief me|(?:give me|what'?s|what is)\s+(?:my|the|a)\s+(?:morning |evening |daily )?(?:briefing|brief|"
    r"rundown|summary)|morning briefing|evening briefing|daily briefing|what'?s (?:on )?my day(?: look(?:ing)? like)?|"
    r"what(?:'s| is| does) (?:my day|today) look(?:ing)? like|what did i do today|how was my day|"
    r"what'?s (?:on )?(?:for )?tomorrow)\b", re.I)


def asked_for_briefing(text: str) -> Optional[str]:
    """None, or which briefing a request asks for ("morning" / "evening" / "")."""
    t = (text or "").strip()
    if not t or len(t.split()) > 12 or not _ASKED.search(t):
        return None
    if re.search(r"\b(?:evening|what did i do today|how was my day|tomorrow)\b", t, re.I):
        return "evening"
    if re.search(r"\bmorning\b", t, re.I):
        return "morning"
    return ""


def _hm(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def _listing(items: List[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _dur(seconds: float) -> str:
    m = int(round(seconds / 60.0))
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    return f"{h}h" if m < 10 else f"{h}h{m:02d}"


@dataclass
class BriefingConfig:
    enabled: bool = True
    morning: bool = True
    evening: bool = True
    morning_from: int = 5               # local hours
    morning_until: int = 12
    evening_from: int = 18
    evening_until: int = 23
    min_work_s: float = 600.0           # less than this on a topic is not worth saying
    max_topics: int = 3

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "BriefingConfig":
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


class Briefings:
    def __init__(self, cfg: Optional[BriefingConfig] = None, path: Optional[str] = None,
                 followups=None, activity=None, wonder=None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg or BriefingConfig()
        self.path = path
        self.followups = followups
        self.activity = activity
        self.wonder = wonder
        self.clock = clock
        self.providers: List[Provider] = []
        self.given: Dict[str, str] = {}          # kind -> the local date it was last given
        self.stats = {"given": 0, "asked": 0, "empty": 0}
        self._load()

    def add_provider(self, fn: Provider):
        """Another part: fn(kind, now) returns one sentence, or None."""
        self.providers.append(fn)

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.given = {str(k): str(v) for k, v in (raw.get("given") or {}).items()}
        except Exception:
            logger.exception("Briefing state unreadable; starting empty")

    def _save(self):
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"given": self.given}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("Could not save briefing state")

    # -- when ----------------------------------------------------------------
    @staticmethod
    def _date(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    def window(self, now: float) -> str:
        """Which briefing the hour of `now` belongs to, if any."""
        h = datetime.fromtimestamp(now).hour
        if self.cfg.morning and self.cfg.morning_from <= h < self.cfg.morning_until:
            return "morning"
        if self.cfg.evening and self.cfg.evening_from <= h < self.cfg.evening_until:
            return "evening"
        return ""

    def due(self, now: Optional[float] = None) -> str:
        """The briefing that is owed now and not yet given today ('' if none)."""
        now = self.clock() if now is None else now
        if not self.cfg.enabled:
            return ""
        kind = self.window(now)
        if not kind or self.given.get(kind) == self._date(now):
            return ""
        return kind

    def mark(self, kind: str, now: Optional[float] = None, said: bool = True):
        """Done for today: said, or (said=False) found to have nothing in it."""
        now = self.clock() if now is None else now
        if kind:
            self.given[kind] = self._date(now)
            if said:
                self.stats["given"] += 1
            self._save()

    # -- what ----------------------------------------------------------------
    def _events(self, day: datetime) -> List[Tuple[float, str]]:
        fu = self.followups
        if fu is None:
            return []
        out = []
        try:
            for f in fu.items.values():
                if f.status not in ("pending", "asked"):
                    continue
                d = datetime.fromtimestamp(f.start)
                if d.date() == day.date():
                    out.append((f.start, f"{f.what} at {_hm(f.start)}" if f.clock else f.what))
        except Exception:
            logger.exception("Reading follow-ups for a briefing failed")
        return sorted(out)

    def _worked(self, day: datetime) -> List[Tuple[float, str]]:
        act = self.activity
        if act is None:
            return []
        key = day.strftime("%Y-%m-%d")
        out = []
        try:
            for t in act.topics.values():
                s = float((getattr(t, "by_day", None) or {}).get(key, 0.0))
                if s >= self.cfg.min_work_s:
                    out.append((s, t.name))
        except Exception:
            logger.exception("Reading activity for a briefing failed")
        out.sort(key=lambda x: -x[0])
        return out[:self.cfg.max_topics]

    def compose(self, kind: str, now: Optional[float] = None) -> str:
        """The briefing as one short paragraph, or '' if there is nothing in it."""
        now = self.clock() if now is None else now
        kind = kind or ("morning" if datetime.fromtimestamp(now).hour < 15 else "evening")
        today = datetime.fromtimestamp(now)
        parts: List[str] = []
        if kind == "morning":
            ev = self._events(today)
            if ev:
                parts.append("Today: " + _listing([e for _, e in ev]) + ".")
            worked = self._worked(today - timedelta(days=1))
            if worked:
                parts.append("Yesterday you worked mostly on " + _listing([n for _, n in worked]) + ".")
        else:
            worked = self._worked(today)
            if worked:
                parts.append("Today you spent most of your time on "
                             + _listing([f"{n} ({_dur(s)})" for s, n in worked]) + ".")
            ev = self._events(today + timedelta(days=1))
            if ev:
                parts.append("Tomorrow: " + _listing([e for _, e in ev]) + ".")
        for fn in self.providers:
            try:
                s = fn(kind, now)
            except Exception:
                logger.exception("A briefing part failed; leaving it out")
                s = None
            if s:
                parts.append(str(s).strip())
        w = self.wonder
        if w is not None:
            try:
                n = sum(1 for x in w.wonders.values() if x.status == "pending")
            except Exception:
                n = 0
            if n:
                parts.append("I have a question for you when you have a minute." if n == 1
                             else f"I have {n} questions for you when you have a minute.")
        if not parts:
            self.stats["empty"] += 1
            return ""
        lead = "Good morning." if kind == "morning" else "Here's your day."
        return " ".join([lead] + parts)

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "given": dict(self.given), **self.stats}
