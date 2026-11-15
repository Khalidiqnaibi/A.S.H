"""
src/daemon/timetable.py

A weekly timetable you can say aloud, so "when is my class?" is answered from what you told ASH -- not from a guess.

    my NoSQL class is on Sunday and Tuesday from 10 to 11:30
    I have databases every monday at 2pm for 2 hours
    add a class called Algorithms on weekdays 9 to 10

    when is my next class?   when is my NoSQL class?   do I have class tomorrow?   how long until my class?
    what's my timetable?     remove my NoSQL class     clear my timetable

Nothing here reaches the network. Kept under state/timetable.json. Ten minutes before a class starts it says so once
(through the reminder queue, so a mute still holds it), and `in_class(now)` is what the runtime uses to stay quiet while
you are in one.

Times without am/pm are read as a student's day: 8-11 are morning, 12 is noon, 1-7 are afternoon.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.timetable")

_DAYNAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WD = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
       "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
_WD_ALT = "|".join(sorted(_WD, key=len, reverse=True))
_DAY_RX = re.compile(r"\b(" + _WD_ALT + r")s?\b", re.I)
_TIME = r"\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?"
_RANGE = re.compile(r"(?P<a>" + _TIME + r")\s*(?:to|until|till|-|–|and)\s*(?P<b>" + _TIME + r")", re.I)
_AT = re.compile(r"\bat\s+(?P<a>" + _TIME + r")", re.I)
_FOR = re.compile(r"\bfor\s+(?P<n>\d+(?:\.\d+)?|an?|one|two|three|half an)\s*(?P<u>hours?|hrs?|h|minutes?|mins?)\b", re.I)
_NUMW = {"a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "half an": 0.5}
_KINDS = r"(?:class|lecture|lab|tutorial|seminar|course|module|session)"
_NAME = r"(?P<name>[A-Za-z][\w .&'+/#-]{0,38}?)"
_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|(?:ok(?:ay)?|so)[,\s]+)?(?:remember\s+(?:that\s+)?|note\s+(?:that\s+)?)?"
_ADD1 = re.compile(_A + r"(?:my\s+|the\s+|our\s+)?" + _NAME + r"\s+" + _KINDS + r"\s+(?:is|are|runs?|meets?|happens?|starts?)\s+(?P<rest>.+?)[.!]*\s*$", re.I)
_ADD2 = re.compile(_A + r"i\s+(?:have|got)\s+(?:my\s+|a\s+|an\s+)?" + _NAME + r"(?:\s+" + _KINDS + r")?\s+(?P<rest>(?:on|every|at)\s+.+?)[.!]*\s*$", re.I)
_ADD3 = re.compile(_A + r"(?:add|put|save|schedule)\s+(?:a\s+|an\s+|my\s+)?(?:" + _KINDS + r"\s+)?(?:(?:called|named|for)\s+)?" + _NAME
                   + r"(?:\s+" + _KINDS + r")?\s*(?:,|:|\s)\s*(?P<rest>(?:on|every|at|from)?\s*(?:" + _WD_ALT + r"|weekdays?|\d).+?)"
                   r"(?:\s+(?:to|in|on)\s+(?:my\s+)?(?:timetable|schedule))?[.!]*\s*$", re.I)
_Z = r"\s*[.!?]*\s*$"
_NEXT = re.compile(_A + r"(?:when(?:'s|\s+is|\s+do\s+i\s+have|\s+are|\s+does)\s+(?:my\s+|the\s+)?(?:next\s+)?(?:(?P<n1>[\w .&'+/#-]{2,30}?)\s+)?"
                   + _KINDS + r"(?:\s+(?:next|start|begin))?|what(?:'s|\s+is)\s+my\s+next\s+" + _KINDS + r"|what\s+(?:time|day)\s+(?:is|does)\s+(?:my\s+)?(?:next\s+)?(?:(?P<n2>[\w .&'+/#-]{2,30}?)\s+)?"
                   + _KINDS + r"(?:\s+(?:start|begin))?)" + _Z, re.I)
_WHEN_I_HAVE = re.compile(_A + r"when\s+do\s+i\s+have\s+(?:class|lectures?|a\s+class)" + _Z, re.I)
_UNTIL = re.compile(_A + r"(?:how\s+(?:long|much\s+time)|when)\s+(?:is\s+it\s+)?(?:until|till|before|to)\s+(?:my\s+)?(?:next\s+)?(?:(?P<n>[\w .&'+/#-]{2,30}?)\s+)?"
                    + _KINDS + _Z, re.I)
_DAYQ = re.compile(_A + r"(?:do\s+i\s+have|have\s+i\s+got|is\s+there|what)\s+(?:any\s+)?(?:a\s+)?(?:classes|class|lectures?|labs?)\s*(?:do\s+i\s+have\s+|have\s+i\s+got\s+)?"
                   r"(?:on\s+)?(?P<day>today|tomorrow|tonight|" + _WD_ALT + r")?(?:\s+(?:today|tomorrow))?" + _Z, re.I)
_LIST = re.compile(_A + r"(?:what(?:'s|\s+is)\s+(?:on\s+)?my\s+(?:timetable|schedule|classes|class\s+schedule|class\s+timetable)|"
                   r"(?:show|read|tell\s+me|list)\s+(?:me\s+)?(?:my\s+)?(?:timetable|schedule|classes|class\s+schedule)|"
                   r"what\s+classes\s+do\s+i\s+have(?:\s+this\s+week)?|what(?:'s|\s+is)\s+my\s+week\s+like)" + _Z, re.I)
_REMOVE = re.compile(_A + r"(?:remove|delete|drop|cancel|forget)\s+(?:my\s+|the\s+)?(?:" + _KINDS + r"\s+)?(?P<name>[\w .&'+/#-]{2,30}?)(?:\s+" + _KINDS + r")?"
                     r"(?:\s+(?:from|on)\s+(?:my\s+)?(?:timetable|schedule))?" + _Z, re.I)
_CLEAR = re.compile(_A + r"(?:clear|delete|wipe|reset|forget)\s+(?:all\s+)?(?:my\s+)?(?:whole\s+)?(?:timetable|class\s+schedule|schedule|classes)" + _Z, re.I)
_NOT_NAMES = frozenset("""it this that there he she they what who which when my your the a an one class lecture lab exam exams test quiz
    interview appointment meeting dentist doctor deadline party trip flight date call game match birthday""".split())
_RECURRING = re.compile(r"\bevery\b|\beach\b|\bweekly\b|\b(?:" + _WD_ALT + r")s\b", re.I)

NO_TIMETABLE = ("I don't have your timetable yet. Tell me like this: \"my NoSQL class is on Sunday and Tuesday from 10 to 11:30\".")


def _clock_min(s: str, noon_rule: bool = True) -> Optional[int]:
    """"10", "10:30", "2pm", "14:00" -> minutes after midnight."""
    m = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*", s or "", re.I)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    ap = (m.group(3) or "").lower().replace(".", "")
    if mi > 59 or h > 24:
        return None
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif not ap and noon_rule and 1 <= h <= 7:
        h += 12
    if h > 23 and not (h == 24 and mi == 0):
        return None
    return h * 60 + mi


def _hm(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _days_of(text: str) -> List[int]:
    t = text.lower()
    out = set()
    if re.search(r"\bweekdays?\b", t):
        out |= {0, 1, 2, 3, 4}
    if re.search(r"\bweekends?\b", t):
        out |= {5, 6}
    if re.search(r"\bevery\s*day\b|\bdaily\b", t):
        out |= set(range(7))
    for m in _DAY_RX.finditer(t):
        out.add(_WD[m.group(1).lower()])
    return sorted(out)


def _span(rest: str) -> Optional[Tuple[int, int]]:
    """(start, end) minutes from "10 to 11:30", "at 2pm for 2 hours", "at 9" (an hour and a half when no end is said)."""
    r = _RANGE.search(rest)
    if r:
        a = _clock_min(r.group("a"))
        b_txt = r.group("b")
        b = _clock_min(b_txt)
        if a is None or b is None:
            return None
        a_txt = r.group("a").lower()
        if b <= a and not re.search(r"am|pm|a\.m|p\.m", b_txt.lower()):
            b += 12 * 60                                  # "12 to 1": the 1 is afternoon
        if b <= a and not re.search(r"am|pm|a\.m|p\.m", a_txt):
            return None
        return (a, b) if b > a else None
    at = _AT.search(rest)
    if at is None:
        at = re.search(r"(?:^|\s)(?:from\s+)?(?P<a>" + _TIME + r")(?:\s|$)", rest, re.I)
    if at is None:
        return None
    a = _clock_min(at.group("a"))
    if a is None:
        return None
    f = _FOR.search(rest)
    if f:
        n = f.group("n").lower()
        v = _NUMW.get(n)
        if v is None:
            try:
                v = float(n)
            except ValueError:
                return None
        mins = v * (60.0 if f.group("u").lower().startswith("h") else 1.0)
        if not 10 <= mins <= 600:
            return None
        return a, a + int(mins)
    return a, a + 90


def _title(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" .,:;-")
    return name if any(c.isupper() for c in name[1:]) or name.isupper() else (name[:1].upper() + name[1:])


def _ok_name(name: str) -> bool:
    w = [x for x in re.split(r"\s+", name.lower().strip()) if x]
    return bool(w) and not all(x in _NOT_NAMES for x in w) and len(name) <= 40


def _when_word(dt: datetime, now: datetime) -> str:
    d = (dt.date() - now.date()).days
    if d == 0:
        return "today"
    if d == 1:
        return "tomorrow"
    return dt.strftime("%A")


def _in_text(seconds: float) -> str:
    m = int(round(seconds / 60.0))
    if m < 1:
        return "now"
    if m < 60:
        return f"in {m} minute{'s' if m != 1 else ''}"
    h, mm = divmod(m, 60)
    if mm == 0:
        return f"in {h} hour{'s' if h != 1 else ''}"
    return f"in {h} hour{'s' if h != 1 else ''} and {mm} minutes"


class Timetable:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, lead_min: int = 10,
                 enabled: bool = True):
        self.path = path
        self.clock = clock
        self.lead_min = lead_min
        self.enabled = enabled
        self._lock = threading.RLock()
        self.classes: List[Dict[str, Any]] = []
        self._told: Dict[str, str] = {}                  # class id -> the date it was announced
        self.stats = {"added": 0, "asked": 0, "announced": 0}
        self._load()

    # -- persistence -------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            ok = []
            for c in raw.get("classes", []):
                if (isinstance(c, dict) and c.get("name") and isinstance(c.get("days"), list)
                        and all(isinstance(d, int) and 0 <= d <= 6 for d in c["days"]) and isinstance(c.get("start"), int)
                        and isinstance(c.get("end"), int) and 0 <= c["start"] < c["end"] <= 24 * 60):
                    ok.append({"id": str(c.get("id") or uuid.uuid4().hex[:6]), "name": str(c["name"])[:40], "days": sorted(set(c["days"])),
                               "start": c["start"], "end": c["end"]})
            self.classes = ok
        except Exception:
            logger.exception("Timetable unreadable; starting empty")
            self.classes = []

    def save(self):
        if not self.path:
            return
        try:
            from src.memory.atomic import write_json
            write_json(self.path, {"classes": self.classes}, indent=2)
        except Exception:
            logger.exception("Could not save the timetable")

    # -- the schedule --------------------------------------------------------------
    def occurrences(self, now: float, days: int = 8) -> List[Tuple[datetime, datetime, Dict[str, Any]]]:
        """Every class from `now` (or already begun) over the next `days` days, soonest first, on the wall clock."""
        base = datetime.fromtimestamp(now)
        out = []
        for k in range(-1, days + 1):
            day = (base + timedelta(days=k)).replace(hour=0, minute=0, second=0, microsecond=0)
            for c in self.classes:
                if day.weekday() in c["days"]:
                    s = day + timedelta(minutes=c["start"])
                    e = day + timedelta(minutes=c["end"])
                    if e > base:
                        out.append((s, e, c))
        out.sort(key=lambda x: x[0])
        return out

    def in_class(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now = self.clock() if now is None else now
        d = datetime.fromtimestamp(now)
        for s, e, c in self.occurrences(now, days=1):
            if s <= d < e:
                return c
        return None

    def next_class(self, now: float, name: str = "") -> Optional[Tuple[datetime, datetime, Dict[str, Any]]]:
        d = datetime.fromtimestamp(now)
        for s, e, c in self.occurrences(now):
            if s <= d:
                continue
            if name and name.lower() not in c["name"].lower():
                continue
            return s, e, c
        return None

    def _find(self, name: str) -> List[Dict[str, Any]]:
        n = re.sub(r"\s+", " ", name.lower()).strip()
        exact = [c for c in self.classes if c["name"].lower() == n]
        return exact or [c for c in self.classes if n and n in c["name"].lower()]

    @staticmethod
    def _describe(c: Dict[str, Any]) -> str:
        days = c["days"]
        if days == [0, 1, 2, 3, 4]:
            dtxt = "weekdays"
        elif len(days) == 1:
            dtxt = _DAYNAMES[days[0]]
        else:
            names = [_DAYNAMES[d] for d in days]
            dtxt = ", ".join(names[:-1]) + " and " + names[-1]
        return f"{c['name']}: {dtxt}, {_hm(c['start'])} to {_hm(c['end'])}"

    # -- hearing ---------------------------------------------------------------------
    @staticmethod
    def is_request(text: str) -> bool:
        """Is this about the timetable (for code that must refuse it to a guest)?"""
        t = (text or "").strip()
        return bool(t and len(t) <= 220 and (any(rx.match(t) for rx in (_CLEAR, _LIST, _UNTIL, _NEXT, _WHEN_I_HAVE, _DAYQ))
                                             or any(rx.match(t) and (rx is not _ADD2 or _RECURRING.search(rx.match(t).group("rest")))
                                                    and _days_of(rx.match(t).group("rest")) and _span(rx.match(t).group("rest")) is not None
                                                    for rx in (_ADD1, _ADD2, _ADD3))))

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self.enabled:
            return None
        t = (text or "").strip()
        if not t or len(t) > 220:
            return None
        now = self.clock() if now is None else now

        if _CLEAR.match(t):
            n = len(self.classes)
            if not n:
                return "Your timetable is already empty."
            with self._lock:
                self.classes = []
                self.save()
            return f"Cleared your timetable ({n} class{'es' if n != 1 else ''})."
        m = _LIST.match(t)
        if m:
            self.stats["asked"] += 1
            if not self.classes:
                return NO_TIMETABLE
            return "Your timetable: " + "; ".join(self._describe(c) for c in self.classes) + "."
        m = _UNTIL.match(t)
        if m:
            self.stats["asked"] += 1
            return self._until(now, (m.group("n") or "").strip())
        m = _NEXT.match(t) or _WHEN_I_HAVE.match(t)
        if m:
            self.stats["asked"] += 1
            name = ""
            if "n1" in m.re.groupindex:
                name = (m.group("n1") or m.group("n2") or "").strip()
            return self._next(now, name)
        m = _DAYQ.match(t)
        if m:
            self.stats["asked"] += 1
            return self._day(now, (m.group("day") or "today").lower())
        for rx in (_ADD1, _ADD2, _ADD3):
            m = rx.match(t)
            if m and (rx is not _ADD2 or _RECURRING.search(m.group("rest"))):      # "I have X on Monday" alone is one day, not weekly
                say = self._add(m.group("name"), m.group("rest"))
                if say:
                    return say
        m = _REMOVE.match(t)
        if m and self.classes:
            name = m.group("name").strip()
            if _ok_name(name):
                hit = self._find(name)
                if hit:
                    with self._lock:
                        self.classes = [c for c in self.classes if c not in hit]
                        self.save()
                    return f"Okay, I took {hit[0]['name']} off your timetable."
        return None

    def _add(self, name: str, rest: str) -> Optional[str]:
        name = _title(name)
        if not _ok_name(name):
            return None
        days = _days_of(rest)
        span = _span(rest)
        if not days or span is None:
            return None
        s, e = span
        if e > 24 * 60:
            e = 24 * 60 - 1
        with self._lock:
            same = [c for c in self.classes if c["name"].lower() == name.lower()]
            for c in same:
                self.classes.remove(c)
            self.classes.append({"id": uuid.uuid4().hex[:6], "name": name, "days": days, "start": s, "end": e})
            self.stats["added"] += 1
            self.save()
        return f"Got it -- {self._describe(self.classes[-1])}." + (" (I replaced the one I had.)" if same else "")

    def _next(self, now: float, name: str) -> str:
        if not self.classes:
            return NO_TIMETABLE
        if name and not self._find(name):
            return f"I don't have a {name} class on your timetable."
        cur = self.in_class(now)
        nx = self.next_class(now, name)
        pre = ""
        if cur is not None and (not name or name.lower() in cur["name"].lower()):
            pre = f"You're in {cur['name']} now, until {_hm(cur['end'])}. "
        if nx is None:
            return (pre + "I don't see another class on your timetable this week.").strip()
        s, e, c = nx
        d = datetime.fromtimestamp(now)
        w = _when_word(s, d)
        extra = f" ({_in_text((s - d).total_seconds())})" if w == "today" else ""
        if name:
            return f"{pre}Your next {c['name']} class is {w} at {_hm(c['start'])}{extra}."
        return f"{pre}Your next class is {c['name']} {w} at {_hm(c['start'])}{extra}."

    def _until(self, now: float, name: str) -> str:
        if not self.classes:
            return NO_TIMETABLE
        if name and not self._find(name):
            return f"I don't have a {name} class on your timetable."
        nx = self.next_class(now, name)
        if nx is None:
            return "I don't see another class on your timetable this week."
        s, _e, c = nx
        d = datetime.fromtimestamp(now)
        return f"{c['name']} starts {_in_text((s - d).total_seconds())}, at {_hm(c['start'])} {_when_word(s, d)}."

    def _day(self, now: float, day: str) -> str:
        if not self.classes:
            return NO_TIMETABLE
        d = datetime.fromtimestamp(now)
        if day in ("today", "tonight"):
            target = d.date()
        elif day == "tomorrow":
            target = (d + timedelta(days=1)).date()
        else:
            wd = _WD[day.rstrip("s")] if day.rstrip("s") in _WD else _WD.get(day, d.weekday())
            ahead = (wd - d.weekday()) % 7
            target = (d + timedelta(days=ahead)).date()
        rows = sorted((c for c in self.classes if target.weekday() in c["days"]), key=lambda c: c["start"])
        label = {"today": "today", "tonight": "tonight", "tomorrow": "tomorrow"}.get(day, _DAYNAMES[target.weekday()])
        if not rows:
            return f"No classes {label}."
        return f"{label.capitalize()}: " + ", ".join(f"{c['name']} at {_hm(c['start'])}" for c in rows) + "."

    # -- the clock ---------------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> List[str]:
        """Lines to say now: a class that starts within `lead_min` minutes, once per class per day."""
        if not self.enabled or not self.classes:
            return []
        now = self.clock() if now is None else now
        d = datetime.fromtimestamp(now)
        out = []
        for s, _e, c in self.occurrences(now, days=1):
            lead = (s - d).total_seconds() / 60.0
            key = f"{c['id']}"
            day = s.strftime("%Y-%m-%d")
            if 0 < lead <= self.lead_min and self._told.get(key) != day:
                self._told[key] = day
                self.stats["announced"] += 1
                out.append(f"{c['name']} starts {_in_text(lead * 60)}, at {_hm(c['start'])}.")
        return out

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "classes": len(self.classes), **self.stats}
