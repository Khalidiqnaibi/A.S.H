"""
src/daemon/calendar_ics.py

The user's calendar, read from iCalendar (.ics) files or feeds.

ASH has no account access, and should not need any: every calendar program
can export an .ics file or publish a private .ics feed (Google Calendar's
"secret address in iCal format", Outlook's "publish calendar"). This reads
those, with no extra library:

    events     one-off events, all-day events, times in UTC or local, and the
               common repeats (daily, weekly on given days, monthly, yearly;
               with COUNT, UNTIL, INTERVAL and EXDATE); cancelled ones skipped
    sources    files on disk, re-read when they change; feeds fetched over
               HTTPS on their own thread every half hour
    uses       the briefing ("On your calendar today: 10:00 standup, 14:00
               design review"), and "what's on my calendar today / tomorrow"

Read-only: nothing is ever written back, and nothing is sent anywhere but the
feed's own address.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.calendar")

_DAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


@dataclass
class Event:
    start: datetime          # local, naive
    end: Optional[datetime]
    summary: str
    location: str = ""
    allday: bool = False


def _unfold(text: str) -> List[str]:
    out: List[str] = []
    for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line[:1] in (" ", "\t") and out:
            out[-1] += line[1:]
        else:
            out.append(line)
    return out


def _unescape(v: str) -> str:
    return v.replace("\\n", " ").replace("\\N", " ").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\").strip()


def _when(params: str, value: str) -> Optional[Tuple[datetime, bool]]:
    """An iCalendar date or date-time as a local naive datetime, and whether it
    was a whole day."""
    v = value.strip()
    try:
        if "VALUE=DATE" in params.upper() and "VALUE=DATE-TIME" not in params.upper() or re.fullmatch(r"\d{8}", v):
            return datetime.strptime(v[:8], "%Y%m%d"), True
        if v.endswith("Z"):
            dt = datetime.strptime(v[:-1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            return dt.astimezone().replace(tzinfo=None), False
        return datetime.strptime(v[:15], "%Y%m%dT%H%M%S"), False      # floating or TZID: taken as local
    except ValueError:
        return None


def _rrule(rule: str) -> Dict[str, str]:
    return dict(p.split("=", 1) for p in rule.split(";") if "=" in p)


def parse_ics(text: str, frm: datetime, to: datetime) -> List[Event]:
    """Every event (repeats expanded) that starts within [frm, to)."""
    out: List[Event] = []
    cur: Optional[Dict[str, Any]] = None
    for line in _unfold(text):
        if line == "BEGIN:VEVENT":
            cur = {"exdates": []}
            continue
        if line == "END:VEVENT":
            if cur is not None:
                out.extend(_expand(cur, frm, to))
            cur = None
            continue
        if cur is None or ":" not in line:
            continue
        head, value = line.split(":", 1)
        name, _, params = head.partition(";")
        name = name.upper()
        if name in ("DTSTART", "DTEND"):
            w = _when(params, value)
            if w:
                cur[name] = w
        elif name == "SUMMARY":
            cur["summary"] = _unescape(value)
        elif name == "LOCATION":
            cur["location"] = _unescape(value)
        elif name == "RRULE":
            cur["rrule"] = _rrule(value.strip())
        elif name == "EXDATE":
            for v in value.split(","):
                w = _when(params, v)
                if w:
                    cur["exdates"].append(w[0])
        elif name == "STATUS":
            cur["status"] = value.strip().upper()
    out.sort(key=lambda e: e.start)
    return out


def _expand(ev: Dict[str, Any], frm: datetime, to: datetime) -> List[Event]:
    if "DTSTART" not in ev or ev.get("status") == "CANCELLED":
        return []
    start, allday = ev["DTSTART"]
    end = ev.get("DTEND", (None, False))[0]
    dur = (end - start) if end else None
    summary = ev.get("summary") or "(untitled)"
    loc = ev.get("location", "")
    ex = set(ev.get("exdates") or [])
    rr = ev.get("rrule")

    def mk(s):
        return Event(s, (s + dur) if dur else None, summary, loc, allday)

    if not rr:
        return [mk(start)] if frm <= start < to else []
    freq = rr.get("FREQ", "").upper()
    interval = max(1, int(rr.get("INTERVAL", "1") or 1))
    count = int(rr["COUNT"]) if rr.get("COUNT", "").isdigit() else None
    until = _when("", rr["UNTIL"])[0] if rr.get("UNTIL") and _when("", rr["UNTIL"]) else None
    days = [_DAYS[d[-2:]] for d in rr.get("BYDAY", "").split(",") if d[-2:] in _DAYS]
    out: List[Event] = []
    n = 0
    occ = start
    # With no COUNT to keep, jump straight to the window instead of walking
    # every repeat since the event began (a daily one from years ago).
    if count is None and occ < frm - timedelta(days=7):
        gap = (frm - occ).days - 7
        if freq == "DAILY":
            occ = occ + timedelta(days=(gap // interval) * interval)
        elif freq == "WEEKLY":
            occ = occ + timedelta(weeks=((gap // 7) // interval) * interval)
    step = 0
    while occ < to and step < 20000:
        step += 1
        cands = [occ]
        if freq == "WEEKLY" and days:
            week0 = occ - timedelta(days=occ.weekday())
            cands = sorted(week0 + timedelta(days=d) for d in days)
            cands = [c for c in cands if c >= start]
        for c in cands:
            if until and c > until:
                return out
            if count is not None and n >= count:
                return out
            n += 1
            if c in ex:
                continue
            if frm <= c < to:
                out.append(mk(c))
        if freq == "DAILY":
            occ = occ + timedelta(days=interval)
        elif freq == "WEEKLY":
            base = occ - timedelta(days=occ.weekday()) if days else occ
            occ = base + timedelta(weeks=interval)
            if days:
                occ = occ.replace(hour=start.hour, minute=start.minute, second=start.second)
        elif freq == "MONTHLY":
            m = occ.month - 1 + interval
            y = occ.year + m // 12
            try:
                occ = occ.replace(year=y, month=m % 12 + 1)
            except ValueError:
                occ = occ.replace(year=y, month=m % 12 + 1, day=28)
        elif freq == "YEARLY":
            try:
                occ = occ.replace(year=occ.year + interval)
            except ValueError:
                occ = occ.replace(year=occ.year + interval, day=28)
        else:
            break
    return out


@dataclass
class CalendarConfig:
    enabled: bool = True
    files: Tuple[str, ...] = ()
    urls: Tuple[str, ...] = ()
    refresh_s: float = 1800.0
    timeout_s: float = 10.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CalendarConfig":
        c = cls()
        for k, v in (d or {}).items():
            if k in ("files", "urls") and isinstance(v, (list, tuple)):
                setattr(c, k, tuple(str(x) for x in v))
            elif k == "enabled":
                c.enabled = bool(v)
            elif k in ("refresh_s", "timeout_s"):
                try:
                    setattr(c, k, float(v))
                except (TypeError, ValueError):
                    pass
        return c


def _fetch(url: str, timeout: float) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "ASH-local-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:            # noqa: S310 -- the user's own feed
        return r.read(5_000_000).decode("utf-8", "replace")


class Calendar:
    def __init__(self, cfg: Optional[CalendarConfig] = None, fetch: Callable[[str, float], str] = _fetch,
                 clock: Callable[[], float] = time.time, background: bool = True):
        self.cfg = cfg or CalendarConfig()
        self.fetch = fetch
        self.clock = clock
        self.background = background
        self._lock = threading.RLock()
        self._texts: Dict[str, str] = {}             # source -> its .ics text
        self._mtimes: Dict[str, float] = {}
        self._fetched: Dict[str, float] = {}
        self.stats = {"read": 0, "fetched": 0, "errors": 0}

    def refresh(self, now: Optional[float] = None):
        """Re-read changed files; fetch feeds that are due (off the caller's thread)."""
        if not self.cfg.enabled:
            return
        now = self.clock() if now is None else now
        for p in self.cfg.files:
            path = os.path.expanduser(p)
            try:
                mt = os.path.getmtime(path)
            except OSError:
                continue
            if self._mtimes.get(path) == mt:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read(5_000_000)
                with self._lock:
                    self._texts[path] = text
                    self._mtimes[path] = mt
                self.stats["read"] += 1
            except OSError:
                self.stats["errors"] += 1
        for u in self.cfg.urls:
            if now - self._fetched.get(u, -1e18) < self.cfg.refresh_s:
                continue
            self._fetched[u] = now
            if self.background:
                threading.Thread(target=self._get, args=(u,), daemon=True, name="ash-calendar").start()
            else:
                self._get(u)

    def _get(self, url: str):
        try:
            text = self.fetch(url, self.cfg.timeout_s)
            if "BEGIN:VCALENDAR" not in text:
                raise ValueError("not a calendar")
            with self._lock:
                self._texts[url] = text
            self.stats["fetched"] += 1
        except Exception as e:
            self.stats["errors"] += 1
            logger.info("Calendar feed failed: %s", type(e).__name__)

    def events_on(self, day: date) -> List[Event]:
        frm = datetime(day.year, day.month, day.day)
        to = frm + timedelta(days=1)
        with self._lock:
            texts = list(self._texts.values())
        out: List[Event] = []
        for t in texts:
            try:
                out.extend(parse_ics(t, frm, to))
            except Exception:
                logger.exception("A calendar could not be read")
        seen = set()
        uniq = []
        for e in sorted(out, key=lambda e: e.start):           # all-day ones start at midnight: first
            k = (e.start, e.summary)
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return uniq

    @staticmethod
    def describe(events: List[Event]) -> str:
        parts = [e.summary if e.allday else f"{e.start.strftime('%H:%M')} {e.summary}" for e in events[:6]]
        more = len(events) - len(parts)
        s = ", ".join(parts)
        return s + (f", and {more} more" if more > 0 else "")

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        self.refresh(now)
        d = datetime.fromtimestamp(now).date() + (timedelta(days=1) if kind == "evening" else timedelta(0))
        ev = self.events_on(d)
        if not ev:
            return None
        return ("On your calendar today: " if kind != "evening" else "On your calendar tomorrow: ") + \
            self.describe(ev) + "."

    _ASK = re.compile(r"\b(?:what'?s|what is|anything|what do i have)\s+(?:on\s+)?(?:in\s+)?my\s+(?:calendar|schedule|agenda)"
                      r"(?:\s+(?P<when>today|tomorrow))?\b|\bdo i have (?:anything|any meetings) "
                      r"(?P<when2>today|tomorrow)\b", re.I)

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self.cfg.enabled or not (self.cfg.files or self.cfg.urls):
            return None
        m = self._ASK.search(text or "")
        if not m or len(text) > 100:
            return None
        now = self.clock() if now is None else now
        self.refresh(now)
        when = (m.group("when") or m.group("when2") or "today").lower()
        d = datetime.fromtimestamp(now).date() + (timedelta(days=1) if when == "tomorrow" else timedelta(0))
        ev = self.events_on(d)
        if not ev:
            return f"Nothing on your calendar {when}."
        return f"{when.capitalize()}: {self.describe(ev)}."

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.cfg.enabled, "sources": len(self._texts), **self.stats}
