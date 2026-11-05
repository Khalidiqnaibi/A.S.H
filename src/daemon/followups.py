"""
src/daemon/followups.py

Follow-ups: asking how something went, after it happened.

"I have an interview on Thursday." A friend who heard that asks on Friday how
it went. ASH heard it too, and until now forgot it the moment the sentence was
answered. This module keeps it:

    detect     a future event in what the user says: something they have
               (an interview, an exam, the dentist), a trip, seeing someone,
               with a time it resolves to (tomorrow, on Thursday, at 3pm, in
               two days, next week, 12 March). No time, no follow-up: "I have
               an exam" says nothing about when to ask.
    keep       one follow-up per event, rescheduled when the user moves it,
               dropped when it is cancelled, closed without a question when
               the user brings it up themselves afterwards ("the interview
               went great").
    ask        once, after the event is over, at a free moment, as an
               activity competing with the others (see freemoment.py). An
               important event is worth more than a routine meeting; the
               value fades over the days after it, and after the window the
               question is let go.
    hear       the reply is read for how it went (good / bad / not yet /
               did not happen) by rule, acknowledged, and kept as a sentence
               the concept network learns from.

Rule-based throughout: no model, no network. Only the user's own utterances
are read. A follow-up keeps the event phrase ("your interview") and its time,
not the sentence it came from. The state file lives under state/.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .freemoment import Activity, Candidate, SentimentFn, read_with
from .wondering import _NON_ANSWER_START

logger = logging.getLogger("ash.daemon.followups")

HOUR = 3600.0
DAY = 86400.0

# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
# Events worth asking about, and how much. A job interview matters more than
# the weekly standup; both are kept, the big one wins a free moment.
BIG = {
    "interview": "interview", "exam": "exam", "exams": "exams", "final": "final", "finals": "finals",
    "midterm": "midterm", "midterms": "midterms", "presentation": "presentation", "demo": "demo",
    "surgery": "surgery", "operation": "operation", "appointment": "appointment",
    "dentist": "dentist", "doctor": "doctor", "checkup": "checkup", "check-up": "checkup",
    "wedding": "wedding", "trip": "trip", "flight": "flight", "launch": "launch", "release": "release",
    "deadline": "deadline", "audition": "audition", "defense": "defense", "defence": "defense",
    "viva": "viva", "pitch": "pitch", "competition": "competition", "match": "match",
    "tournament": "tournament", "race": "race", "concert": "concert", "gig": "gig",
    "party": "party", "date": "date", "conference": "conference", "hackathon": "hackathon",
    "recital": "recital", "graduation": "graduation", "move": "move", "results": "results",
    "driving test": "driving test", "job interview": "job interview",
}
ROUTINE = {
    "meeting": "meeting", "call": "call", "class": "class", "lecture": "lecture",
    "session": "session", "review": "review", "talk": "talk", "workshop": "workshop",
    "test": "test", "quiz": "quiz", "game": "game", "dinner": "dinner", "lunch": "lunch",
    "standup": "standup", "stand-up": "standup", "practice": "practice", "training": "training",
}
_NOUNS = {**ROUTINE, **BIG}
_NOUN_ALT = "|".join(sorted((re.escape(k) for k in _NOUNS), key=len, reverse=True))

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
            "saturday": 5, "sunday": 6, "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3,
            "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple of": 2, "a couple of": 2,
           "few": 3, "a few": 3}

_WD_ALT = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_MON_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
_NUM_ALT = r"\d{1,3}|" + "|".join(sorted((re.escape(k) for k in NUMBERS), key=len, reverse=True))

# Said about something that has already happened, or was called off.
_PAST = re.compile(r"\b(?:was|went|had|did|got through|passed|failed|aced|bombed|nailed|"
                   r"finished|yesterday|last night|earlier today|this morning)\b", re.I)
_CANCEL = re.compile(r"\b(?:cancel(?:l)?ed|called off|scrapped|not happening|isn'?t happening|"
                     r"off now|got cancel(?:l)?ed)\b", re.I)
_MOVED = re.compile(r"\b(?:postponed|moved|rescheduled|pushed(?: back)?|delayed|bumped)\b", re.I)
_NOT_YET = re.compile(r"\b(?:not yet|hasn'?t happened|haven'?t (?:had|done)|still waiting|"
                      r"waiting (?:for|on)|don'?t know yet|no idea yet|find out|tomorrow actually)\b", re.I)
_DIDNT = re.compile(r"\b(?:didn'?t happen|never happened|got cancel(?:l)?ed|was cancel(?:l)?ed|"
                    r"called off|didn'?t go|skipped it|missed it)\b", re.I)
_FILLER = frozenset("ok okay k hm hmm mm yes no yeah yep nope sure right".split())


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _clean(s: str, n: int = 60) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s or "")
    return re.sub(r"\s+", " ", s).strip(" \t\"'`")[:n].strip()


# ----------------------------------------------------------------------------
# Time
# ----------------------------------------------------------------------------
def _day_start(ts: float) -> float:
    t = time.localtime(ts)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def _at(day_ts: float, hour: int, minute: int = 0) -> float:
    t = time.localtime(day_ts)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, hour, minute, 0, 0, 0, -1))


def _add_days(ts: float, n: int) -> float:
    t = time.localtime(ts)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday + n, t.tm_hour, t.tm_min, 0, 0, 0, -1))


def _num(s: str) -> Optional[int]:
    s = (s or "").strip().lower()
    if s.isdigit():
        return int(s)
    return NUMBERS.get(s)


@dataclass
class When:
    start: float                # best estimate of when it starts
    clock: bool                 # a time of day was said
    span: str                   # the words that said it, for tests and status


def resolve_when(text: str, now: float) -> Optional[When]:
    """The time a sentence puts on something, if it says one.

    Only forward-looking forms are read; "yesterday" is a past marker, handled
    elsewhere. A weekday is the next one (today if it is today and the hour is
    still ahead); "next Monday" is the one in the following week.
    """
    t = text.lower()
    today = _day_start(now)
    day: Optional[float] = None
    span: List[str] = []
    rough_hour: Optional[int] = None

    m = re.search(r"\b(?:the\s+)?day after tomorrow\b", t)
    if m:
        day, span = _add_days(today, 2), [m.group(0)]
    if day is None:
        m = re.search(r"\btomorrow(?:\s+(morning|afternoon|evening|night))?\b", t)
        if m:
            day, span = _add_days(today, 1), [m.group(0)]
            rough_hour = {"morning": 9, "afternoon": 14, "evening": 19, "night": 20}.get(m.group(1) or "")
    if day is None:
        m = re.search(r"\b(tonight|this (?:morning|afternoon|evening)|later today|today)\b", t)
        if m:
            day, span = today, [m.group(0)]
            rough_hour = {"tonight": 20, "this morning": 9, "this afternoon": 14,
                          "this evening": 19}.get(m.group(1))
    if day is None:
        m = re.search(r"\b(?:(next|this|on|coming)\s+)?(" + _WD_ALT + r")\b", t)
        if m and not re.search(r"\blast\s+" + re.escape(m.group(2)) + r"\b", t):
            wd = WEEKDAYS[m.group(2)]
            cur = time.localtime(today).tm_wday
            ahead = (wd - cur) % 7
            if m.group(1) == "next":
                # "next Monday" said in the same week means the one after the
                # coming one; said at the weekend it means the coming one.
                if ahead == 0 or cur + ahead <= 6 and cur < 5:
                    ahead += 7
            day, span = _add_days(today, ahead), [m.group(0)]
            if ahead == 0:
                span.append("(today)")
    if day is None:
        m = re.search(r"\b(?:on\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MON_ALT + r")\b", t) \
            or re.search(r"\b(?:on\s+)?(" + _MON_ALT + r")\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\b", t)
        if m:
            a, b = m.group(1), m.group(2)
            d, mon = (int(a), MONTHS[b]) if a.isdigit() else (int(b), MONTHS[a])
            if 1 <= d <= 31:
                y = time.localtime(today).tm_year
                cand = time.mktime((y, mon, d, 0, 0, 0, 0, 0, -1))
                if cand < today:
                    cand = time.mktime((y + 1, mon, d, 0, 0, 0, 0, 0, -1))
                day, span = cand, [m.group(0)]
    if day is None:
        m = re.search(r"\bon\s+the\s+(\d{1,2})(?:st|nd|rd|th)\b", t)
        if m and 1 <= int(m.group(1)) <= 31:
            lt = time.localtime(today)
            cand = time.mktime((lt.tm_year, lt.tm_mon, int(m.group(1)), 0, 0, 0, 0, 0, -1))
            if cand < today:
                cand = time.mktime((lt.tm_year, lt.tm_mon + 1, int(m.group(1)), 0, 0, 0, 0, 0, -1))
            day, span = cand, [m.group(0)]
    if day is None:
        m = re.search(r"\bin\s+(" + _NUM_ALT + r")\s+(minute|min|hour|hr|day|week)s?\b", t)
        if m and _num(m.group(1)):
            n = _num(m.group(1))
            unit = m.group(2)
            if unit in ("minute", "min", "hour", "hr"):
                start = now + n * (60 if unit in ("minute", "min") else HOUR)
                return When(start, True, m.group(0))
            day = _add_days(today, n * (7 if unit == "week" else 1))
            span = [m.group(0)]
    if day is None:
        m = re.search(r"\b(next week|this weekend|next weekend|next month)\b", t)
        if m:
            cur = time.localtime(today).tm_wday
            if m.group(1) == "next week":
                day = _add_days(today, 7 - cur)              # the Monday after
                rough_hour = 12
            elif m.group(1) == "this weekend":
                day = _add_days(today, max(0, 5 - cur))
            elif m.group(1) == "next weekend":
                day = _add_days(today, 12 - cur)
            else:
                day = _add_days(today, 30)
            span = [m.group(0)]

    # the time of day
    hour = minute = None
    m = re.search(r"\b(?:at|by|around|from)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", t) \
        or re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", t)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2) or 0)
        ap = (m.group(3) or "").replace(".", "")
        if h <= 24 and mi < 60:
            if ap == "pm" and h < 12:
                h += 12
            elif ap == "am" and h == 12:
                h = 0
            elif not ap and 1 <= h <= 7:
                h += 12                                   # "at 3" is the afternoon
            hour, minute = h % 24, mi
            span.append(m.group(0))
    else:
        m = re.search(r"\bat\s+(noon|midday|midnight)\b", t)
        if m:
            hour, minute = (0 if m.group(1) == "midnight" else 12), 0
            span.append(m.group(0))

    if day is None and hour is None:
        return None
    if day is None:
        day = today if _at(today, hour, minute) > now else _add_days(today, 1)
    if hour is not None:
        return When(_at(day, hour, minute), True, " ".join(span))
    start = _at(day, rough_hour if rough_hour is not None else 12)
    if day == today and start < now:
        start = now                     # "today", said this evening: it is still ahead
    return When(start, False, " ".join(span))


# ----------------------------------------------------------------------------
# Follow-ups
# ----------------------------------------------------------------------------
@dataclass
class FollowUp:
    id: str
    key: str                       # dedupe identity: the event, normalised
    what: str                      # "your interview", "your trip to Paris", "seeing Sara"
    noun: str                      # interview / trip / seeing
    start: float
    clock: bool
    due: float                     # when it is over enough to ask
    created: float
    important: bool = False
    status: str = "pending"        # pending | asked | answered | told | cancelled | expired
    asked_at: float = 0.0
    closed_at: float = 0.0
    outcome: str = ""              # good | bad | mixed | not_yet | didnt_happen
    reschedules: int = 0

    def public(self) -> Dict[str, Any]:
        return asdict(self)


OPEN = ("pending", "asked")


@dataclass
class FollowUpsConfig:
    enabled: bool = True
    window_s: float = 3 * DAY          # after it is due, how long the question stays worth asking
    duration_s: float = 2 * HOUR       # assumed length of an event with a clock time
    max_open: int = 25
    max_ahead_s: float = 60 * DAY      # further out than this is a plan, not a follow-up
    routine: bool = True               # also follow up meetings, calls, classes (worth less)
    answer_window_s: float = 600.0
    sources: Tuple[str, ...] = ("console", "microphone", "mic")

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FollowUpsConfig":
        c = cls()
        for k, v in (d or {}).items():
            if not hasattr(c, k):
                continue
            cur = getattr(c, k)
            try:
                if isinstance(cur, bool):
                    setattr(c, k, bool(v))
                elif isinstance(cur, tuple):
                    setattr(c, k, tuple(str(x) for x in (v or ())))
                elif isinstance(cur, int):
                    setattr(c, k, int(v))
                else:
                    setattr(c, k, float(v))
            except (TypeError, ValueError):
                pass
        return c


@dataclass
class Event:
    """What one sentence said about one event."""
    noun: str
    what: str
    key: str
    when: Optional[When]
    important: bool
    past: bool = False
    cancelled: bool = False
    moved: bool = False


_ARTICLE = r"(?:my|the|a|an|our|this|that|his|her|their)"
_RE_NOUN = re.compile(r"\b(" + _ARTICLE + r")\s+((?:[a-z\-]+\s+){0,2}?)(" + _NOUN_ALT + r")\b", re.I)
_RE_TRIP = re.compile(r"\b(?:flying|travel(?:l)?ing|driving|heading|going|off|leaving)\s+(?:back\s+)?to\s+"
                      r"([A-Z][a-zA-Z]{2,20}(?:\s[A-Z][a-zA-Z]{2,20})?)")
_RE_SEEING = re.compile(r"\b(?:seeing|meeting(?: up with| with)?|visiting|having (?:dinner|lunch|coffee) with)\s+"
                        r"([A-Z][a-z]{1,19})\b")
_IMPORTANT = re.compile(r"\b(?:big|important|huge|final|first|real|key|scary|nervous)\b", re.I)
_NOT_NAMES = frozenset("I Monday Tuesday Wednesday Thursday Friday Saturday Sunday Today Tomorrow "
                       "The A An My Our This That Next Last".split())


def detect_events(sent: str, now: float, routine: bool = True) -> List[Event]:
    """Every event one sentence talks about."""
    when = resolve_when(sent, now)
    past = bool(_PAST.search(sent)) and (when is None or when.start <= now)
    cancelled = bool(_CANCEL.search(sent))
    moved = bool(_MOVED.search(sent))
    out: List[Event] = []
    seen = set()
    for m in _RE_NOUN.finditer(sent):
        art, mods, noun_raw = m.group(1).lower(), (m.group(2) or "").strip().lower(), m.group(3).lower()
        noun = _NOUNS[noun_raw]
        if noun in ROUTINE.values() and noun not in BIG.values() and not routine:
            continue
        # "the test suite", "a call to the API", "the date format": a following
        # word that makes the noun something else
        after = sent[m.end():m.end() + 14].lower()
        if re.match(r"\s*(?:suite|case|file|format|function|method|stack|log|data|cases|run)\b", after):
            continue
        if noun == "call" and re.match(r"\s*(?:to|into|from)\s+(?:the\s+|an?\s+)?(?:[A-Z]{2,}|api|function|"
                                       r"method|endpoint|server|service|database|db|backend|model)\b",
                                       sent[m.end():m.end() + 30], re.I):
            continue                                  # a function call, not a phone call
        if noun == "date" and art not in ("my", "a"):
            continue                                  # "the date" is a calendar date
        if noun == "move" and art != "my":
            continue
        mods_words = [w for w in mods.split() if w not in ("big", "important", "huge", "real", "scary")]
        phrase = " ".join(mods_words + [noun])
        important = noun in BIG.values() or bool(_IMPORTANT.search(mods))
        key = "event:" + norm(phrase)
        if key in seen:
            continue
        seen.add(key)
        own = "your" if art in ("my", "our", "a", "an") else "the"
        what = f"{own} {phrase}" if noun not in ("dentist", "doctor") else f"the {noun}"
        out.append(Event(noun, what, key, when, important, past, cancelled, moved))
    for m in _RE_TRIP.finditer(sent):
        place = m.group(1)
        if place.split()[0] in _NOT_NAMES:
            continue
        key = "trip:" + norm(place)
        if key not in seen:
            seen.add(key)
            out.append(Event("trip", f"your trip to {place}", key, when, True, past, cancelled, moved))
    for m in _RE_SEEING.finditer(sent):
        name = m.group(1)
        if name in _NOT_NAMES:
            continue
        key = "seeing:" + norm(name)
        if key not in seen:
            seen.add(key)
            out.append(Event("seeing", f"seeing {name}", key, when, False, past, cancelled, moved))
    return out


def _question(fu: FollowUp) -> str:
    if fu.noun == "trip":
        return f"How was {fu.what}?"
    if fu.noun == "seeing":
        return f"How was {fu.what}?"
    if fu.noun in ("dentist", "doctor"):
        return f"How did it go at {fu.what}?"
    if fu.noun == "deadline":
        return f"Did you make {fu.what}?"
    if fu.noun == "results":
        return f"Did you get {fu.what}?"
    return f"How did {fu.what} go?"


class FollowUps:
    """The events ASH heard about, and the questions it owes."""

    def __init__(self, cfg: Optional[FollowUpsConfig] = None, path: Optional[str] = None,
                 clock: Callable[[], float] = time.time, sentiment: Optional[SentimentFn] = None):
        self.cfg = cfg or FollowUpsConfig()
        self.sentiment = sentiment         # the brain's sentiment model reads how it went
        self.path = path
        self.clock = clock
        self._lock = threading.RLock()
        self.items: Dict[str, FollowUp] = {}
        self.stats = {"noted": 0, "rescheduled": 0, "cancelled": 0, "told": 0,
                      "asked": 0, "answered": 0, "expired": 0}
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for d in raw.get("items", []):
                try:
                    f = FollowUp(**{k: v for k, v in d.items() if k in FollowUp.__dataclass_fields__})
                    self.items[f.id] = f
                except Exception:
                    continue
            self.stats.update({k: int(v) for k, v in (raw.get("stats") or {}).items() if k in self.stats})
        except Exception:
            logger.exception("Follow-up state unreadable; starting empty")
            try:
                os.replace(self.path, self.path + ".bad")
            except Exception:
                pass

    def save(self):
        if not self.path:
            return
        with self._lock:
            closed = sorted((f for f in self.items.values() if f.status not in OPEN),
                            key=lambda f: f.closed_at)[-50:]
            keep = [f for f in self.items.values() if f.status in OPEN] + closed
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                tmp = f"{self.path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"version": 1, "items": [f.public() for f in keep],
                               "stats": self.stats}, fh, indent=1)
                os.replace(tmp, self.path)
            except Exception:
                logger.exception("Could not save follow-ups")

    # -- reading what the user said ------------------------------------------
    def hears(self, source: str) -> bool:
        s = (source or "").lower()
        if s.startswith("sim:"):
            s = s[4:]
        return s in {x.lower() for x in self.cfg.sources}

    def _open_for(self, key: str) -> Optional[FollowUp]:
        return next((f for f in self.items.values() if f.key == key and f.status in OPEN), None)

    def _due(self, w: When) -> float:
        if w.clock:
            due = w.start + self.cfg.duration_s
        else:                                           # the evening of the day, and after it
            due = max(_at(_day_start(w.start), 18), w.start + self.cfg.duration_s)
        h = time.localtime(due).tm_hour
        if h >= 22 or h < 8:                            # not asked in the night
            base = due if h < 8 else _add_days(due, 1)
            due = _at(_day_start(base), 9)
        return due

    def _close(self, f: FollowUp, status: str, now: float, outcome: str = ""):
        f.status = status
        f.closed_at = now
        if outcome:
            f.outcome = outcome
        if status in self.stats:
            self.stats[status] += 1

    def observe(self, text: str, now: Optional[float] = None) -> List[str]:
        """Note the events a sentence mentions. Returns sentences learned
        outright (an outcome the user told without being asked)."""
        now = self.clock() if now is None else now
        text = text if isinstance(text, str) else ""
        if not text or not self.cfg.enabled:
            return []
        learned: List[str] = []
        changed = False
        with self._lock:
            for sent in [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()][:10]:
                for ev in detect_events(sent, now, self.cfg.routine):
                    r = self._apply(ev, sent, now)
                    if r is None:
                        continue
                    changed = True
                    if r:
                        learned.append(r)
            if changed:
                self.save()
        return learned

    def _apply(self, ev: Event, sent: str, now: float) -> Optional[str]:
        """None: nothing changed. '': changed, nothing learned. Else a sentence."""
        cur = self._open_for(ev.key)
        if ev.cancelled:
            if cur is not None:
                self._close(cur, "cancelled", now)
                return ""
            return None
        if ev.past:
            # They told us how it went, or that it happened: nothing to ask.
            if cur is not None and cur.start <= now + HOUR:
                m = read_with(self.sentiment, sent)
                outcome = "good" if (m or 0) > 0.2 else "bad" if (m or 0) < -0.2 else ""
                self._close(cur, "told", now, outcome)
                return (f"The user's {cur.what.split(' ', 1)[1]} went {'well' if outcome == 'good' else 'badly'}."
                        if outcome else "")
            return None
        if ev.when is None or ev.when.start < now - HOUR:
            return None
        if ev.when.start - now > self.cfg.max_ahead_s:
            return None
        due = self._due(ev.when)
        if cur is not None:
            if abs(cur.start - ev.when.start) < 60 or (not ev.moved and not ev.when.clock
                                                       and _day_start(cur.start) == _day_start(ev.when.start)):
                return None                             # the same event, said again
            cur.start, cur.clock, cur.due = ev.when.start, ev.when.clock, due
            cur.reschedules += 1
            cur.status = "pending"
            cur.asked_at = 0.0
            cur.important = cur.important or ev.important
            self.stats["rescheduled"] += 1
            return ""
        if sum(1 for f in self.items.values() if f.status in OPEN) >= self.cfg.max_open:
            oldest = min((f for f in self.items.values() if f.status == "pending"),
                         key=lambda f: (f.important, f.due), default=None)
            if oldest is None:
                return None
            self._close(oldest, "expired", now)
        f = FollowUp(id=uuid.uuid4().hex[:8], key=ev.key, what=_clean(ev.what, 50), noun=ev.noun,
                     start=ev.when.start, clock=ev.when.clock, due=due, created=now,
                     important=ev.important)
        self.items[f.id] = f
        self.stats["noted"] += 1
        logger.debug("follow-up %s noted (%s)", f.id, f.noun)
        return ""

    # -- asking ---------------------------------------------------------------
    def housekeeping(self, now: Optional[float] = None):
        now = self.clock() if now is None else now
        with self._lock:
            changed = False
            for f in self.items.values():
                if f.status in OPEN and now > f.due + self.cfg.window_s:
                    self._close(f, "expired", now)
                    changed = True
            if changed:
                self.save()

    def due_now(self, now: Optional[float] = None) -> Optional[FollowUp]:
        """The follow-up most worth asking now, or None."""
        now = self.clock() if now is None else now
        self.housekeeping(now)
        with self._lock:
            cands = [f for f in self.items.values() if f.status == "pending" and f.due <= now]
            if not cands:
                return None
            return max(cands, key=lambda f: (self.value(f, now), -f.due))

    def value(self, f: FollowUp, now: float) -> float:
        """How worth asking it is: an important event, asked soon after, most."""
        base = 0.8 if f.important else 0.45
        age = max(0.0, now - f.due)
        return base * max(0.6, 1.0 - 0.4 * age / max(1.0, self.cfg.window_s))

    def question(self, f: FollowUp) -> str:
        return _question(f)

    def mark_asked(self, fid: str, now: Optional[float] = None) -> Optional[FollowUp]:
        now = self.clock() if now is None else now
        with self._lock:
            f = self.items.get(fid)
            if f is None or f.status != "pending":
                return None
            f.status = "asked"
            f.asked_at = now
            self.stats["asked"] += 1
            self.save()
            return f

    def hear(self, fid: str, text: str, now: Optional[float] = None) -> Optional[Tuple[str, str]]:
        """The reply to "how did X go?". Returns (what to say, sentence learned)
        or None if it does not read as an answer."""
        now = self.clock() if now is None else now
        text = text if isinstance(text, str) else ""
        with self._lock:
            f = self.items.get(fid)
            if f is None or not text.strip():
                return None
            t = text.strip()
            noun = f.what.split(" ", 1)[1] if " " in f.what else f.what
            if _DIDNT.search(t):
                self._finish(f, "didnt_happen", now)
                return "Ah, okay. Thanks for telling me.", f"The user's {noun} did not happen."
            if _NOT_YET.search(t):
                self._finish(f, "not_yet", now)
                return "Fingers crossed, then.", ""
            if len(t.split()) <= 2 and t.lower().strip(".!? ") in _FILLER:
                return None                             # a noise, not an answer
            if t.endswith("?") or _NON_ANSWER_START.match(t):
                return None                             # a request or a question of their own
            m = self._how_it_went(t)
            if m is None:
                return None
            if m > 0.2:
                self._finish(f, "good", now)
                return ("That's great to hear!" if f.important else "Good to hear."), \
                    f"The user's {noun} went well."
            if m < -0.2:
                self._finish(f, "bad", now)
                return ("I'm sorry it didn't go well." if f.important else "Sorry to hear that."), \
                    f"The user's {noun} went badly."
            self._finish(f, "mixed", now)
            return "Okay, thanks for telling me.", f"The user's {noun} went okay."

    def _how_it_went(self, text: str) -> Optional[float]:
        """The sentiment model's reading of a reply to "how did it go?". A short
        reply the model reads but is unsure of ("so-so", "it was alright") is
        an answer too: it went okay. No model, no reading."""
        m = read_with(self.sentiment, text)
        if m is not None or self.sentiment is None:
            return m
        try:
            r = self.sentiment(text) or {}
        except Exception:
            return None
        known = str(r.get("sentiment") or r.get("label") or "").upper() in ("POSITIVE", "NEGATIVE", "NEUTRAL")
        return 0.0 if known and len(text.split()) <= 6 else None

    def _finish(self, f: FollowUp, outcome: str, now: float):
        if f.status in OPEN or f.status == "told":
            if f.status != "told":
                self._close(f, "answered", now, outcome)
            else:
                f.outcome = outcome
            self.save()

    # -- inspection ------------------------------------------------------------
    def pending(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = sorted((f for f in self.items.values() if f.status in OPEN), key=lambda f: f.due)
            return [{"id": f.id, "what": f.what, "status": f.status, "start": f.start, "due": f.due,
                     "important": f.important, "question": _question(f)} for f in rows]

    def dismiss(self, fid: str, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        with self._lock:
            f = self.items.get(fid)
            if f is None or f.status not in OPEN:
                return False
            self._close(f, "cancelled", now)
            self.save()
            return True

    def report(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        rows = self.pending()
        lines = [f"Follow-ups: {'on' if self.cfg.enabled else 'off'}. {len(rows)} open."]
        for r in rows:
            when = time.strftime("%a %d %b %H:%M", time.localtime(r["start"]))
            due = "due" if r["due"] <= now else f"in {int((r['due'] - now) / 3600)}h"
            lines.append(f"  {r['id']}  {r['status']:<7} {when}  ({due})  {r['question']}")
        return "\n".join(lines)

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        with self._lock:
            return {"enabled": self.cfg.enabled,
                    "open": sum(1 for f in self.items.values() if f.status in OPEN),
                    "due": sum(1 for f in self.items.values() if f.status == "pending" and f.due <= now),
                    "stats": dict(self.stats)}


class FollowUpActivity(Activity):
    """Offer the follow-up most worth asking to the free-moment contest."""
    name = "follow_up"
    weight = 1.0
    cooldown_s = 2 * HOUR

    def __init__(self, followups: FollowUps, learn: Optional[Callable[[str], None]] = None,
                 mood: Optional[Callable[[float], None]] = None):
        self.fu = followups
        self.learn = learn
        self.set_mood = mood
        self._proposed: Optional[str] = None
        self._asked: Optional[Tuple[str, float]] = None

    def propose(self, env, now):
        f = self.fu.due_now(now)
        if f is None:
            return None
        self._proposed = f.id
        return Candidate(self.name, self.fu.question(f), self.fu.value(f, now), data={"followup": f.id})

    def spoken(self, now):
        if self._proposed:
            if self.fu.mark_asked(self._proposed, now) is not None:
                self._asked = (self._proposed, now)
            self._proposed = None

    def hear(self, text, now):
        if self._asked is None:
            return None
        fid, _t = self._asked
        r = self.fu.hear(fid, text, now)
        if r is None:
            return None
        self._asked = None
        say, learned = r
        if learned and self.learn:
            try:
                self.learn(learned)
            except Exception:
                logger.exception("learning a follow-up answer failed")
        if self.set_mood and ("badly" in learned):
            try:
                self.set_mood(-0.4)                       # go gently for a while
            except Exception:
                pass
        return say
