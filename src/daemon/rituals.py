"""
src/daemon/rituals.py

Hello and goodbye, said the way a friend would.

    "good morning"     "Good morning, Khalid." and, the first time today, the
                       morning briefing (what's on, what you did yesterday)
    "good evening" /   named by the real hour, not the word: "good morning" at
    "good afternoon"   3 pm gets "It's 3:04 pm already -- hello, Khalid."
    "good night"       "Good night, Khalid. Sleep well." with a one-line
                       recap of the day if ASH has one, and tomorrow's first thing

By rule, from the clock and what ASH already keeps; no language model, and
the face's own reaction (sleepy, happy) is the face's business (moods.py).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Optional

_A = r"^\s*(?:(?:hey|hi|hello|oh)[,\s]+)?(?:ash[,\s]+)?"
_Z = r"(?:[,\s]+(?:ash|everyone|buddy|friend))?\s*[.!]*\s*$"
_MORNING = re.compile(_A + r"(?:good\s+morning|morning|mornin'?)" + _Z, re.I)
_AFTERNOON = re.compile(_A + r"(?:good\s+afternoon|afternoon)" + _Z, re.I)
_EVENING = re.compile(_A + r"(?:good\s+evening|evening)" + _Z, re.I)
_NIGHT = re.compile(_A + r"(?:good\s*night|night\s*night|nighty\s*night|nite|g'?night|sleep\s+well|i'?m\s+(?:going\s+to|off\s+to)\s+(?:bed|sleep)|"
                    r"i'?m\s+heading\s+to\s+bed)" + _Z, re.I)


def kind_of(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or len(t) > 50:
        return None
    for name, rx in (("night", _NIGHT), ("morning", _MORNING), ("afternoon", _AFTERNOON), ("evening", _EVENING)):
        if rx.match(t):
            return name
    return None


def _clock(now: float) -> str:
    d = datetime.fromtimestamp(now)
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'}"


def part_of_day(hour: int) -> str:
    return "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 22 else "night"


def greet(kind: str, now: float, user: str = "",
          briefing: Optional[Callable[[str, float], str]] = None, briefed_today: bool = False) -> str:
    """The reply to a greeting. `briefing(kind, now)` gives a briefing paragraph or ''."""
    hour = datetime.fromtimestamp(now).hour
    who = f", {user}" if user else ""
    actual = part_of_day(hour)
    if kind == "night":
        recap = ""
        if briefing is not None:
            try:
                recap = (briefing("evening", now) or "").replace("Here's your day.", "").strip()
            except Exception:
                recap = ""
        tail = f" {recap}" if recap else ""
        if hour < 5:
            return f"It's {_clock(now)} -- good night{who}. Get some sleep.{tail}"
        return f"Good night{who}. Sleep well.{tail}"
    if kind == "morning" and actual == "morning":
        out = f"Good morning{who}."
        if briefing is not None and not briefed_today:
            try:
                b = (briefing("morning", now) or "").replace("Good morning.", "").strip()
            except Exception:
                b = ""
            if b:
                out += " " + b
        return out
    if kind == actual:
        return f"Good {actual}{who}."
    # The word and the clock disagree: say the real time, kindly.
    if actual == "night" and hour >= 22:
        return f"It's {_clock(now)} already -- hello{who}."
    return f"It's {_clock(now)} already -- good {actual}{who}."
