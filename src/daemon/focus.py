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
from .numwords import digits

_ON = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:i'?m\s+(?:going to\s+)?focus(?:ing)?|i\s+need\s+to\s+focus|"
                 r"let\s+me\s+focus|focus\s+mode|(?:turn\s+on\s+|start\s+)?(?:focus|do\s+not\s+disturb|dnd)(?:\s+mode)?|"
                 r"don'?t\s+disturb\s+me)\b(?P<rest>.*)$", re.I)
_OFF = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:i'?m\s+done\s+focusing|done\s+focusing|focus\s+(?:mode\s+)?off|"
                  r"end\s+(?:the\s+)?focus|stop\s+focus(?:ing)?|turn\s+off\s+(?:focus|do\s+not\s+disturb|dnd)|"
                  r"(?:focus|dnd)\s+over|you\s+can\s+talk\s+(?:again|now))\b", re.I)
# Saying where you are is asking to be left alone: "I'm in lecture", "I have an exam", "I'm driving", "stop interrupting me".
# value = how long it lasts when no time is given (minutes).
_PLACE = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|right)[,\s]+)?"
    r"(?:(?:i'?m|i\s+am|we'?re|i'?m\s+currently|i'?m\s+now)\s+(?:in|at|starting|going\s+(?:in)?to)\s+(?:a\s+|an\s+|the\s+|my\s+|our\s+)?"
    r"(?P<what>lectures?|class(?:room)?|meeting|exam|test|quiz|interview|seminar|lab|tutorial|presentation|webinar|"
    r"stand-?up|call|lesson|session|study\s+session)"
    r"|i\s+have\s+(?:a\s+|an\s+|my\s+)?(?P<what2>exam|test|quiz|interview|lecture|class|presentation)\s+(?:now|right\s+now|in\s+a\s+(?:bit|minute|moment))"
    r"|i'?m\s+(?P<what3>driving|presenting|praying|sleeping|studying|in\s+(?:lecture|class|a\s+meeting)))"
    r"(?P<rest>(?:\s+(?:now|right\s+now|today|for\s+a\s+(?:while|bit)|for\s+.+|until\s+.+|till\s+.+|at\s+.+))?)\s*[.!]*\s*$", re.I)
_STOP_BOTHERING = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:stop\s+(?:interrupting|bothering|disturbing|pestering)(?:\s+me)?|"
    r"stop\s+with\s+(?:all\s+)?the\s+(?:interrupts?|interruptions?|interventions?)|"
    r"(?:minimum|minimal|fewer|no)\s+(?:interventions?|interruptions?)(?:\s+mode)?|"
    r"(?:i\s+want|give\s+me)\s+(?:minimum|minimal|no)\s+(?:interventions?|interruptions?))"
    r"(?:\s+(?:please|now|right\s+now))?\s*[.!]*\s*$", re.I)
_DONE = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:i'?m\s+(?:done|out|finished|back)(?:\s+now)?|(?:the\s+|my\s+)?"
                   r"(?:lecture|class|meeting|exam|test|interview|seminar|call|session)\s+(?:is\s+)?(?:over|done|finished|ended)|"
                   r"(?:lecture|class|meeting|exam)\s+ended|we'?re\s+done)\s*[.!]*\s*$", re.I)
_PLACE_MIN = {"lecture": 90, "lectures": 90, "class": 90, "classroom": 90, "lesson": 60, "seminar": 90, "lab": 120, "tutorial": 60,
              "exam": 120, "test": 90, "quiz": 45, "interview": 60, "meeting": 60, "presentation": 45, "webinar": 60,
              "stand-up": 20, "standup": 20, "call": 45, "driving": 60, "presenting": 45, "praying": 20, "sleeping": 480,
              "studying": 90, "session": 60, "study session": 90}


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
        t = digits((text or "").strip())
        if not t or len(t) > 100:
            return None
        if _OFF.match(t):
            if not self.active(now):
                return "Focus wasn't on."
            self.until = now
            return None                      # the caller says the summary
        if _DONE.match(t):                   # "I'm done", "the lecture is over": ends it, but only if it is on
            if not self.active(now):
                return None
            self.until = now
            return None
        place = _PLACE.match(t)
        bother = None if place else _STOP_BOTHERING.match(t)
        m = _ON.match(t)
        if not m and not place and not bother:
            return None
        rest = ""
        if m:
            rest = m.group("rest") or ""
        elif place:
            rest = place.group("rest") or ""
        what = ""
        if place:
            what = re.sub(r"^in\s+(?:a\s+)?", "", (place.group("what") or place.group("what2") or place.group("what3") or "").lower()).strip()
            what = {"lectures": "lecture", "classroom": "class"}.get(what, what)
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
            if rest.strip(" .!") and not re.fullmatch(r"\s*(?:mode|now|right now|today|please|for a while|for a bit)\s*[.!]?\s*",
                                                       rest, re.I):
                return None                  # "I'm focusing on the report" is not a request
            until = now + (_PLACE_MIN.get(what, self.default_s / 60.0) * 60.0 if place else self.default_s)
        until = min(until, now + self.max_s)
        self.until, self.started = until, now
        self.held = Counter()
        self.stats["sessions"] += 1
        hm = datetime.fromtimestamp(until).strftime('%H:%M')
        if place or bother:
            for_what = ("while you're " + what if what in ("driving", "presenting", "praying", "sleeping", "studying") else f"for your {what}") if what else ""
            return (f"Okay -- quiet {for_what + ', ' if for_what else ''}till {hm}. Say \"I'm done\" when it's over. "
                    f"Reminders still go off.")
        return (f"Focus mode until {hm}. I'll keep to myself; "
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
