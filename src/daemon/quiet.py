"""
src/daemon/quiet.py

"Quiet till 9", "do not disturb for 2 hours", "be quiet until 3:30 pm".

ASH stops volunteering (the attention gate is muted) and comes back by itself
at the time, so a person doesn't have to remember to say "wake up". Answers to
direct questions are unaffected, exactly as with "go to sleep".

parse(text, now) -> (until_timestamp, phrase) or None.
    "quiet for 2 hours" / "for 30 minutes" / "for an hour"   a length
    "quiet till 9" / "until 9pm" / "till 3:30" / "until noon"   a clock time
        (the next time that clock reads that: "till 9" at 10pm is tomorrow 9am;
         without am/pm, the next 9 on the clock -- am or pm -- that is ahead)
"""

from __future__ import annotations

import re
import time
from typing import Optional, Tuple

_LEAD = (r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:(?:be|go|stay)\s+quiet|quiet(?:\s+mode)?|"
         r"do\s+not\s+disturb|don'?t\s+disturb\s+me|silence|mute\s+(?:yourself|ash)|no\s+interruptions|"
         r"leave\s+me\s+alone|stop\s+talking)\s*")
_FOR = re.compile(_LEAD + r"for\s+(?:(?:an?|one)\s+(?P<unit1>hour|minute|min)|(?P<n>\d+(?:\.\d+)?)\s*"
                  r"(?P<unit>hours?|hrs?|h|minutes?|mins?|m))\s*[.!]*\s*$", re.I)
_UNTIL = re.compile(_LEAD + r"(?:till|until|til|to)\s+(?:(?P<word>noon|midnight)|(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*"
                    r"(?P<ap>a\.?m\.?|p\.?m\.?)?)\s*(?:o'?clock)?\s*[.!]*\s*$", re.I)


def parse(text: str, now: float) -> Optional[Tuple[float, str]]:
    t = (text or "").strip()
    if not t or len(t) > 80:
        return None
    m = _FOR.match(t)
    if m:
        if m.group("unit1"):
            mins = 60.0 if m.group("unit1").lower() == "hour" else 1.0
        else:
            n = float(m.group("n"))
            mins = n * (60.0 if m.group("unit").lower().startswith("h") else 1.0)
        if not 1.0 <= mins <= 24 * 60:
            return None
        until = now + mins * 60.0
        return until, _describe(until)
    m = _UNTIL.match(t)
    if m:
        lt = time.localtime(now)
        if m.group("word"):
            hour, minute, ap = (12 if m.group("word").lower() == "noon" else 0), 0, "24"
        else:
            hour, minute = int(m.group("h")), int(m.group("m") or 0)
            ap = (m.group("ap") or "").lower().replace(".", "")
            if hour > 23 or minute > 59 or (ap and hour > 12) or (ap and hour == 0):
                return None
            if ap == "pm" and hour < 12:
                hour += 12
            elif ap == "am" and hour == 12:
                hour = 0
        cands = []
        for add_day in (0, 1, 2):
            hours = [hour] if (ap or m.group("word") or hour > 12 or hour == 0) else [hour % 12, hour % 12 + 12]
            for h in hours:
                cands.append(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + add_day, h, minute, 0, 0, 0, -1)))
        ahead = sorted(c for c in cands if c > now + 60)
        if not ahead:
            return None
        until = ahead[0]
        return until, _describe(until)
    return None


def _describe(until: float) -> str:
    lt = time.localtime(until)
    h12 = lt.tm_hour % 12 or 12
    return f"{h12}:{lt.tm_min:02d} {'am' if lt.tm_hour < 12 else 'pm'}"
