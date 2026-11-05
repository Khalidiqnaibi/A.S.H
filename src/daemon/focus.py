"""
src/daemon/focus.py

Focus mode: ASH keeps to itself while the user concentrates.

    "I'm focusing for an hour", "focus mode for 45 minutes", "do not disturb
    until 3", "let me focus" (an hour)          -> on, until then
    "I'm done focusing", "focus off", "end focus" -> off now

While it is on, nothing is volunteered: no questions, no free-moment remarks,
no check-ins, no briefings, no unprompted observations. What the user asked
for still happens -- a reminder, a timer, a watch goes off as set, and a
question put to ASH is answered. When focus ends (on time or on request),
ASH says in one line what it held back ("While you focused: 2 questions are
waiting and your research on Raft is ready.").

ASH cannot silence other apps' notifications without changing system
settings, and does not try: this keeps ASH itself quiet. No model is used.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .followups import resolve_when

_ON = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:i'?m\s+(?:going to\s+)?focus(?:ing)?|i\s+need\s+to\s+focus|"
                 r"let\s+me\s+focus|focus\s+mode|(?:turn\s+on\s+|start\s+)?(?:focus|do\s+not\s+disturb|dnd)(?:\s+mode)?|"
                 r"don'?t\s+disturb\s+me)\b(?P<rest>.*)$", re.I)
_OFF = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:i'?m\s+done\s+focusing|done\s+focusing|focus\s+(?:mode\s+)?off|"
                  r"end\s+(?:the\s+)?focus|stop\s+focus(?:ing)?|turn\s+off\s+(?:focus|do\s+not\s+disturb|dnd)|"
                  r"(?:focus|dnd)\s+over|you\s+can\s+talk\s+(?:again|now))\b", re.I)


class Focus:
    def __init__(self, clock: Callable[[], float] = time.time, default_s: float = 3600.0,
                 max_s: float = 8 * 3600.0, enabled: bool = True):
        self.clock = clock
        self.default_s = default_s
        self.max_s = max_s
        self.enabled = enabled
        self.until = 0.0
        self.started = 0.0
        self.held: Counter = Counter()
        self.stats = {"sessions": 0, "held": 0}

    def active(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self.enabled and now < self.until

    def hold(self, kind: str):
        """Something ASH would have volunteered and did not."""
        self.held[kind or "remark"] += 1
        self.stats["held"] += 1

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self.enabled:
            return None
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t or len(t) > 100:
            return None
        if _OFF.match(t):
            if not self.active(now):
                return "Focus wasn't on."
            self.until = now
            return None                      # the caller says the summary
        m = _ON.match(t)
        if not m:
            return None
        rest = m.group("rest") or ""
        until = None
        mm = re.search(r"\bfor\s+(.+)$", rest, re.I)
        if mm:
            w = resolve_when("in " + mm.group(1).replace("half an hour", "30 minutes")
                             .replace("an hour and a half", "90 minutes"), now)
            if w and w.clock and w.start > now:
                until = w.start
        mm = re.search(r"\buntil\s+(.+)$", rest, re.I)
        if until is None and mm:
            w = resolve_when("at " + mm.group(1) if not re.match(r"\s*(?:at|tomorrow)\b", mm.group(1)) else mm.group(1),
                             now)
            if w and w.clock and w.start > now:
                until = w.start
        if until is None:
            if rest.strip(" .!") and not re.fullmatch(r"\s*(?:mode|now|please|for a while|for a bit)\s*[.!]?\s*",
                                                       rest, re.I):
                return None                  # "I'm focusing on the report" is not a request
            until = now + self.default_s
        until = min(until, now + self.max_s)
        self.until, self.started = until, now
        self.held = Counter()
        self.stats["sessions"] += 1
        return (f"Focus mode until {datetime.fromtimestamp(until).strftime('%H:%M')}. I'll keep to myself; "
                f"reminders still go off.")

    def summary(self, extra: List[str]) -> str:
        """The one line said when focus ends."""
        mins = max(1, int(round((self.clock() - self.started) / 60.0))) if self.started else 0
        parts = [p for p in extra if p]
        n = sum(self.held.values())
        if not parts and not n:
            return f"Focus done ({mins} min). Nothing came up."
        held = f"I held back {n} thing{'s' if n != 1 else ''}" if n else ""
        body = "; ".join(([held] if held else []) + parts)
        return f"Focus done ({mins} min). While you focused: {body}."

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        return {"enabled": self.enabled, "active": self.active(now),
                "until": self.until if self.active(now) else None, **self.stats}
