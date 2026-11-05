"""
src/daemon/reminders.py

Reminders, timers and alarms that actually go off.

The reminder tool parsed "remind me in 20 minutes to check the oven" and
returned a due time -- and nothing ever looked at it again. This keeps them
and says them when they are due:

    reminder   "remind me to call mom in 20 minutes", "remind me tomorrow at
               9 to send the report"      -> "Reminder: call mom."
    timer      "set a timer for 5 minutes", "timer for half an hour"
                                          -> "Your 5-minute timer is done."
    alarm      "wake me up at 7", "set an alarm for 6:30am"
                                          -> "Good morning -- it's 07:00."

Then "snooze" (or "snooze 10 minutes") right after one goes off brings it
back later; "cancel the timer" / "cancel my reminders" drops them; "what
are my reminders?" lists them. Times are read with the same rules as
follow-ups. They are kept under state/, so a restart does not lose them, and
one that came due while ASH was off is said late rather than never. Being
asked for, they are said even in quiet hours; muted or paused, they wait.
No model is used.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .followups import resolve_when

logger = logging.getLogger("ash.daemon.reminders")

_REMIND = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|can you\s+|could you\s+)?"
                     r"(?:remind me|set (?:a |an )?reminder)\b(?P<rest>.*)$", re.I)
_TIMER = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|can you\s+)?(?:set |start )?(?:a |an )?timer\s+"
                    r"(?:for\s+)?(?P<rest>.+)$", re.I)
_TIMER_N = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|can you\s+)?(?:set |start )?(?:a |an )?"
                      r"(?P<n>\d{1,3})[\s-]*(?P<u>min(?:ute)?s?|h(?:ou)?rs?|hours?|sec(?:ond)?s?)\s+timer\s*[.!]?\s*$", re.I)
_ALARM = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|can you\s+)?(?:wake me(?: up)?|set (?:an |a )?alarm"
                    r"(?: for)?)\s+(?P<rest>.+)$", re.I)
_SNOOZE = re.compile(r"^\s*snooze(?:\s+(?:for\s+)?(?P<n>\d{1,3})\s*(?:min(?:ute)?s?))?\s*[.!]?\s*$", re.I)
_CANCEL = re.compile(r"^\s*(?:cancel|delete|stop|clear|never ?mind)\s+(?:the\s+|my\s+|all\s+(?:my\s+)?)?"
                     r"(?P<k>timers?|alarms?|reminders?)\b", re.I)
# The time words, taken out of a reminder to leave what it is about.
_WD = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?"
_TIME_PHRASES = [re.compile(p, re.I) for p in (
    r"\bin\s+(?:\d{1,3}|an?|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a few)\s+"
    r"(?:minutes?|mins?|hours?|hrs?|days?|weeks?)\b",
    r"\b(?:the\s+)?day after tomorrow\b",
    r"\b(?:tomorrow|tonight|today|later today)(?:\s+(?:morning|afternoon|evening|night))?\b",
    r"\bthis\s+(?:morning|afternoon|evening|weekend)\b",
    r"\b(?:on\s+|next\s+|this\s+|coming\s+)?" + _WD + r"\b",
    r"\b(?:at|by|around)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?",
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)",
    r"\bat\s+(?:noon|midday|midnight)\b",
    r"\b(?:next week|next month|next weekend)\b",
)]
_LIST = re.compile(r"\b(?:what(?:'s| are| is)?|list|any)\s+(?:my\s+)?(?:reminders?|timers?|alarms?)\b"
                   r"|\bdo i have any (?:reminders?|timers?|alarms?)\b", re.I)


@dataclass
class Reminder:
    id: str
    kind: str              # reminder | timer | alarm
    text: str              # what to be reminded of ("" for timers and alarms)
    due: float
    created: float
    label: str = ""        # "5-minute" for a timer
    status: str = "set"    # set | fired | cancelled
    fired_at: float = 0.0
    snoozes: int = 0


def _minutes_label(seconds: float) -> str:
    m = int(round(seconds / 60.0))
    if m < 60:
        return f"{m}-minute"
    h, mm = divmod(m, 60)
    return f"{h}-hour" if mm == 0 else f"{h}-hour {mm}-minute"


def _normalise_time(rest: str) -> str:
    r = " " + rest.strip() + " "
    r = re.sub(r"\bhalf an hour\b", "30 minutes", r, flags=re.I)
    r = re.sub(r"\ban hour and a half\b", "90 minutes", r, flags=re.I)
    r = re.sub(r"\bquarter of an hour\b", "15 minutes", r, flags=re.I)
    return r.strip()


class Reminders:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 enabled: bool = True, snooze_s: float = 300.0, snooze_window_s: float = 180.0):
        self.path = path
        self.clock = clock
        self.enabled = enabled
        self.snooze_s = snooze_s
        self.snooze_window_s = snooze_window_s
        self._lock = threading.RLock()
        self.items: Dict[str, Reminder] = {}
        self._last_fired: Optional[str] = None
        self.stats = {"set": 0, "fired": 0, "snoozed": 0, "cancelled": 0, "late": 0}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for d in raw.get("items", []):
                r = Reminder(**{k: v for k, v in d.items() if k in Reminder.__dataclass_fields__})
                if r.status == "set":
                    self.items[r.id] = r
        except Exception:
            logger.exception("Reminder state unreadable; starting empty")
            try:
                os.replace(self.path, self.path + ".bad")
            except Exception:
                pass

    def save(self):
        if not self.path:
            return
        with self._lock:
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                tmp = f"{self.path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"items": [asdict(r) for r in self.items.values() if r.status == "set"]}, fh, indent=1)
                os.replace(tmp, self.path)
            except Exception:
                logger.exception("Could not save reminders")

    # -- requests --------------------------------------------------------------
    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """A reminder, timer or alarm request (or snooze, cancel, list):
        returns what to say back, or None."""
        if not self.enabled:
            return None
        t = (text or "").strip()
        if not t or len(t) > 160:
            return None
        now = self.clock() if now is None else now
        m = _SNOOZE.match(t)
        if m:
            return self.snooze(int(m.group("n")) * 60 if m.group("n") else None, now)
        m = _CANCEL.match(t)
        if m:
            return self.cancel(m.group("k").lower().rstrip("s"), now)
        if _LIST.search(t) and len(t.split()) <= 8:
            return self.listing(now)
        n = _TIMER_N.match(t)
        if n:
            t = f"set a timer for {n.group('n')} {n.group('u')}"
        m = _TIMER.match(t)
        if m:
            w = resolve_when("in " + _normalise_time(m.group("rest")), now)
            if w is None or not w.clock or w.start <= now:
                return "How long should the timer be? Say something like \"a timer for 5 minutes\"."
            secs = w.start - now
            return self._add("timer", "", w.start, now, label=_minutes_label(secs),
                             say=f"Okay, a {_minutes_label(secs)} timer, starting now.")
        m = _ALARM.match(t)
        if m:
            rest = _normalise_time(m.group("rest")).strip(" .!")
            bare = re.fullmatch(r"(?:at|for)?\s*(\d{1,2})(?::(\d{2}))?(?:\s*o'?clock)?", rest, re.I)
            if bare and int(bare.group(1)) < 24 and int(bare.group(2) or 0) < 60:
                # An alarm "at 7" is seven in the morning, the next time it comes round.
                d = datetime.fromtimestamp(now)
                due = d.replace(hour=int(bare.group(1)), minute=int(bare.group(2) or 0), second=0,
                                microsecond=0).timestamp()
            else:
                w = resolve_when(rest if re.search(r"\b(?:at|in|tomorrow)\b", rest, re.I) else "at " + rest, now)
                if w is None or not w.clock:
                    return "What time should I wake you? Say something like \"wake me up at 7\"."
                due = w.start
            if due <= now:
                due += 86400.0
            hm = datetime.fromtimestamp(due).strftime("%H:%M")
            day = "tomorrow" if datetime.fromtimestamp(due).date() != datetime.fromtimestamp(now).date() else "today"
            return self._add("alarm", "", due, now, say=f"Okay, I'll wake you at {hm} {day}.")
        m = _REMIND.match(t)
        if m:
            rest = _normalise_time(m.group("rest"))
            w = resolve_when(rest, now)
            if w is None or w.start <= now:
                return "When should I remind you? Say something like \"in 20 minutes\" or \"tomorrow at 9\"."
            what = rest
            for rx in _TIME_PHRASES:
                what = rx.sub(" ", what)
            what = re.sub(r"\s+", " ", what).strip(" ,.")
            what = re.sub(r"^(?:to|that|about|of)\s+", "", what, flags=re.I)
            what = re.sub(r"\s+(?:to|that|about)$", "", what, flags=re.I).strip(" ,.")
            if not what:
                return "What should I remind you about?"
            when_s = datetime.fromtimestamp(w.start).strftime("%H:%M")
            if datetime.fromtimestamp(w.start).date() != datetime.fromtimestamp(now).date():
                when_s = datetime.fromtimestamp(w.start).strftime("%a %H:%M")
            return self._add("reminder", what, w.start, now, say=f"Okay, I'll remind you to {what} at {when_s}.")
        return None

    def _add(self, kind: str, text: str, due: float, now: float, label: str = "", say: str = "") -> str:
        with self._lock:
            r = Reminder(id=uuid.uuid4().hex[:6], kind=kind, text=text[:120], due=due, created=now, label=label)
            self.items[r.id] = r
            self.stats["set"] += 1
            self.save()
        return say

    def cancel(self, kind: str, now: float) -> str:
        with self._lock:
            hit = [r for r in self.items.values() if r.status == "set" and r.kind == kind]
            for r in hit:
                r.status = "cancelled"
                self.stats["cancelled"] += 1
            self.save()
        if not hit:
            return f"You don't have any {kind}s set."
        return f"Cancelled your {kind}." if len(hit) == 1 else f"Cancelled your {len(hit)} {kind}s."

    def listing(self, now: float) -> str:
        rows = sorted((r for r in self.items.values() if r.status == "set"), key=lambda r: r.due)
        if not rows:
            return "You don't have any reminders, timers or alarms set."
        parts = []
        for r in rows[:6]:
            when = datetime.fromtimestamp(r.due).strftime("%H:%M")
            if r.kind == "timer":
                parts.append(f"a {r.label} timer ending at {when}")
            elif r.kind == "alarm":
                parts.append(f"an alarm at {when}")
            else:
                parts.append(f"{r.text} at {when}")
        return "You have " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + "."

    def snooze(self, seconds: Optional[float], now: float) -> Optional[str]:
        with self._lock:
            r = self.items.get(self._last_fired or "")
            if r is None or r.status != "fired" or now - r.fired_at > self.snooze_window_s:
                return None
            s = float(seconds) if seconds else self.snooze_s
            r.status = "set"
            r.due = now + s
            r.snoozes += 1
            self.stats["snoozed"] += 1
            self._last_fired = None
            self.save()
        return f"Okay, again in {int(round(s / 60))} minutes."

    # -- firing ----------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> List[str]:
        if not self.enabled:
            return []
        now = self.clock() if now is None else now
        out: List[str] = []
        with self._lock:
            for r in sorted(self.items.values(), key=lambda r: r.due):
                if r.status != "set" or r.due > now:
                    continue
                late = now - r.due > 120.0
                r.status = "fired"
                r.fired_at = now
                self._last_fired = r.id
                self.stats["fired"] += 1
                if late:
                    self.stats["late"] += 1
                out.append(self._say(r, late))
            if out:
                self.save()
        return out

    @staticmethod
    def _say(r: Reminder, late: bool) -> str:
        when = datetime.fromtimestamp(r.due).strftime("%H:%M")
        if r.kind == "timer":
            s = f"Your {r.label} timer is done."
        elif r.kind == "alarm":
            s = f"Good morning -- it's {when}." if datetime.fromtimestamp(r.due).hour < 12 else f"It's {when} -- your alarm."
        else:
            s = f"Reminder: {r.text}."
        return f"I was off when this was due at {when}. {s}" if late else s

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        """Reminders still to come today (morning) or tomorrow (evening)."""
        day = datetime.fromtimestamp(now).date()
        if kind == "evening":
            from datetime import timedelta
            day = day + timedelta(days=1)
        rows = sorted((r for r in self.items.values() if r.status == "set" and r.kind == "reminder"
                       and datetime.fromtimestamp(r.due).date() == day), key=lambda r: r.due)
        if not rows:
            return None
        items = [f"{datetime.fromtimestamp(r.due).strftime('%H:%M')} {r.text}" for r in rows[:4]]
        return ("Reminders today: " if kind != "evening" else "Reminders tomorrow: ") + ", ".join(items) + "."

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "active": sum(1 for r in self.items.values() if r.status == "set"),
                    **self.stats}
