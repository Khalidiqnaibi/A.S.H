"""
src/daemon/care.py

Caring nudges: water, a stretch, "you've been at it for three hours".

ASH adds up how long the user has been at the machine without a real break
(ten minutes away resets it), and at a free moment -- never mid-flow, since
free moments only come at pauses -- may say:

    after 90 minutes   "You've been at it an hour and a half. Maybe stand up
                        and stretch for a minute?"
    every 2 hours      "Quick one: have you had some water lately?"
    after 3 hours      "You've been at it for three hours straight. A short
                        break might help."

Only when the mood is fine (known, and not low), at most once an hour, and
never the same nudge twice in one stretch. "I'm fine" / "stop reminding me"
backs it off like any other free-moment remark (backoff.py); "no more
nudges" turns it off until the daemon restarts.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, Optional, Set

from .freemoment import Activity, Candidate

BREAK_S = 600.0                  # this long away is a break
_OFF = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:no\s+more\s+(?:nudges|reminders\s+to\s+(?:stretch|drink))|"
                  r"stop\s+(?:the\s+)?(?:nudges|nudging\s+me|telling\s+me\s+to\s+(?:stretch|drink(?:\s+water)?)))"
                  r"\s*[.!]*\s*$", re.I)


def _span(seconds: float) -> str:
    h = seconds / 3600.0
    if h >= 2.75:
        return f"{int(round(h))} hours"
    if h >= 1.75:
        return "two hours"
    if h >= 1.25:
        return "an hour and a half"
    return "an hour"


class Care(Activity):
    name = "care"
    weight = 0.8
    cooldown_s = 3600.0

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.since: Optional[float] = None       # when this stretch at the machine began
        self.last_seen: Optional[float] = None
        self.said: Set[str] = set()              # nudges already given this stretch
        self.off = False
        self._proposed: Optional[str] = None
        self.stats: Dict[str, int] = {"nudges": 0, "breaks": 0}

    def tick(self, now: float, idle_s: Optional[float]):
        if idle_s is None:
            return
        if idle_s >= BREAK_S:
            if self.since is not None:
                self.stats["breaks"] += 1
            self.since, self.said = None, set()
            return
        if idle_s < 60.0:
            if self.since is None:
                self.since = now
            self.last_seen = now

    def at_it(self, now: float) -> float:
        return 0.0 if self.since is None else max(0.0, now - self.since)

    def propose(self, env, now):
        if self.off or self.since is None:
            return None
        if env.mood is None or env.mood < 0.0:
            return None                          # only when things are fine
        t = self.at_it(now)
        if t >= 3 * 3600 and "long" not in self.said:
            self._proposed = "long"
            return Candidate(self.name, f"You've been at it for {_span(t)} straight. A short break might help.", 0.55)
        if t >= 2 * 3600 and "water" not in self.said:
            self._proposed = "water"
            return Candidate(self.name, "Quick one: have you had some water lately?", 0.45)
        if t >= 90 * 60 and "stretch" not in self.said:
            self._proposed = "stretch"
            return Candidate(self.name, f"You've been at it {_span(t)}. Maybe stand up and stretch for a minute?", 0.45)
        return None

    def spoken(self, now):
        if self._proposed:
            self.said.add(self._proposed)
            self.stats["nudges"] += 1
            self._proposed = None

    def hear(self, text: str, now: float) -> Optional[str]:
        if _OFF.match(text or ""):
            self.off = True
            return "Okay, no more nudges."
        return None

    def status(self) -> Dict[str, object]:
        return {"off": self.off, "at_it_min": int(self.at_it(self.clock()) / 60), **self.stats}
