"""
src/daemon/situation.py

Putting together what ASH already notices, so it can say the one thing that matters right now -- by rule, no model.

Each part of ASH sees one thing: the calendar knows the 3 o'clock, the activity tracker knows you are gaming, the
battery knows it is at 9%. None of them alone knows that "your 3 o'clock is in ten minutes and you're still in a
game". This does, from a snapshot of all of them (World), and offers at most one line at a time through the attention
gate -- so muting, privacy, quiet hours, focus and the hourly budget all still apply.

    a meeting soon         "Heads up: Standup starts in 10 minutes -- and you're still gaming."     once per event
    a meeting started      "Standup started 2 minutes ago."  (only if you are at the computer and not working)
    battery low            "The battery's at 12% -- worth plugging in."  (urgent at 5%)             every 30 min at most
    memory nearly full     "Memory is 94% full, so things may slow down. Ask me why it's slow ..."   two readings in a row
    a birthday             "It's your sister's birthday today." / "Your sister's birthday is tomorrow."  from notes
    late at it             "It's past 1 -- the coding can wait till morning?"                        once a night

Long stretches of work are the care nudges' job (care.py), not this one's.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
_MON = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DAY = r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
_BDAY = re.compile(r"^(?P<who>.+?)(?:'s|’s)?\s+birthday\s+is\s+(?:on\s+)?(?:the\s+)?(?:" + _MON + r"\s+" + _DAY + r"|" +
                   _DAY.replace("day>", "day2>") + r"(?:\s+of)?\s+" + _MON.replace("mon>", "mon2>") + r")\b", re.I)

BUSY_LEISURE = {"gaming": "gaming", "video": "watching something", "leisure": "browsing"}
WORKING = {"working", "coding", "writing", "reading", "designing", "studying"}


@dataclass
class World:
    """What is true right now, gathered by the daemon each time."""
    now: float
    idle_s: Optional[float] = None
    activity: str = ""                                   # coding / gaming / leisure / "" (from occasion.py)
    events: List[Tuple[float, str]] = field(default_factory=list)      # (start, title) for today and tomorrow
    battery: Optional[Tuple[float, bool]] = None         # (percent, plugged in)
    mem_percent: Optional[float] = None
    birthdays: List[Tuple[str, int, int]] = field(default_factory=list)  # (whose, month, day): "your sister's", 5, 3


@dataclass
class Nudge:
    key: str
    say: str
    strength: float = 0.7
    urgency: float = 0.3


def birthday_of(note_back: str) -> Optional[Tuple[str, int, int]]:
    """("your sister's", 5, 3) from "your sister's birthday is on May 3rd", or None."""
    m = _BDAY.match((note_back or "").strip())
    if not m:
        return None
    mon = (m.group("mon") or m.group("mon2") or "").lower()
    day = int(m.group("day") or m.group("day2") or 0)
    month = MONTHS.get(mon) or MONTHS.get(mon[:3])
    if not month or not 1 <= day <= 31:
        return None
    who = m.group("who").strip()
    who = who if who.lower() in ("your", "my") else (who if who.endswith(("'s", "’s")) else who + "'s")
    who = "your" if who.lower() in ("your", "my") else who
    return who, month, day


def _mins(s: float) -> int:
    return max(1, int(round(s / 60.0)))


class Situation:
    def __init__(self, clock: Callable[[], float] = time.time, cooldowns: Optional[Dict[str, float]] = None):
        self.clock = clock
        self.cool = {"battery": 1800.0, "memory": 7200.0, "late": 6 * 3600.0}
        self.cool.update(cooldowns or {})
        self.said: Dict[str, float] = {}                 # key -> when it was last offered
        self._mem_high = 0                               # consecutive high readings
        self.stats = {"offered": 0}

    # -- the rules ------------------------------------------------------------------------------------------
    def candidates(self, w: World) -> List[Nudge]:
        out: List[Nudge] = []
        here = w.idle_s is not None and w.idle_s < 300   # at the computer
        doing = BUSY_LEISURE.get(w.activity, "")
        for start, title in w.events:
            ahead = start - w.now
            day = datetime.fromtimestamp(start).strftime("%Y%m%d%H%M")
            if 4 * 60 < ahead <= 15 * 60:
                tail = f" -- and you're still {doing}." if doing and here else "."
                out.append(Nudge(f"event-soon:{day}:{title}", f"Heads up: {title} starts in {_mins(ahead)} minutes{tail}",
                                 0.9 if doing else 0.8, 0.5))
            elif -3 * 60 <= ahead <= -30 and here and w.activity not in WORKING:
                out.append(Nudge(f"event-now:{day}:{title}", f"{title} started {_mins(-ahead)} minute"
                                 f"{'s' if _mins(-ahead) != 1 else ''} ago.", 0.92, 0.6))
        if w.battery is not None and not w.battery[1]:
            p = w.battery[0]
            if p <= 5:
                out.append(Nudge("battery", f"The battery's at {p:.0f}% -- plug in now or it'll shut down.", 0.99, 0.95))
            elif p <= 15:
                out.append(Nudge("battery", f"The battery's at {p:.0f}% -- worth plugging in.", 0.85, 0.5))
        if w.mem_percent is not None:
            self._mem_high = self._mem_high + 1 if w.mem_percent >= 93 else 0
            if self._mem_high >= 2:
                out.append(Nudge("memory", f"Memory is {w.mem_percent:.0f}% full, so things may slow down. Ask me "
                                           "\"why is my computer slow\" and I'll tell you what's using it.", 0.7, 0.3))
        today = datetime.fromtimestamp(w.now)
        tomorrow = today + timedelta(days=1)
        for who, month, day in w.birthdays:
            name = "your" if who == "your" else who
            if (month, day) == (today.month, today.day) and 7 <= today.hour < 21 and here:
                out.append(Nudge(f"bday:{today:%Y%m%d}:{who}", f"It's {name} birthday today.", 0.85, 0.4))
            elif (month, day) == (tomorrow.month, tomorrow.day) and 17 <= today.hour < 22 and here:
                cap = name[:1].upper() + name[1:]
                out.append(Nudge(f"bday-eve:{today:%Y%m%d}:{who}", f"{cap} birthday is tomorrow.", 0.8, 0.3))
        if 1 <= today.hour < 5 and here and w.activity in ("coding", "gaming", "working", "writing"):
            what = {"gaming": "the game", "coding": "the coding", "working": "the work", "writing": "the writing"}[w.activity]
            out.append(Nudge("late", f"It's past {today.hour} -- {what} can wait till morning?", 0.6, 0.2))
        return out

    def due(self, n: Nudge, now: float) -> bool:
        last = self.said.get(n.key)
        if last is None:
            return True
        cool = self.cool.get(n.key.split(":")[0])
        return cool is not None and now - last >= cool       # one-off keys (an event, a birthday) never repeat

    def tick(self, w: World) -> Optional[Nudge]:
        """The one line worth offering now, or None. Marks it as said."""
        best = None
        for n in self.candidates(w):
            if self.due(n, w.now) and (best is None or n.strength > best.strength):
                best = n
        if best is not None:
            self.said[best.key] = w.now
            self.stats["offered"] += 1
            for k in [k for k, t in self.said.items() if w.now - t > 3 * 86400]:
                del self.said[k]
        return best

    def status(self) -> Dict[str, int]:
        return dict(self.stats)


def birthdays_from(notes: Iterable) -> List[Tuple[str, int, int]]:
    out = []
    for n in notes or ():
        b = birthday_of(getattr(n, "back", "") or "")
        if b is not None:
            out.append(b)
    return out
