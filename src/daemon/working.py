"""
src/daemon/working.py

"What am I working on?", "what have I been working on today / this week?", "what was I doing?": from what ASH keeps about the user's own windows.

It went to the model, which answered "Khalid, who you mentioned is Immortal" -- it never read the window and activity history ASH already
keeps (`activity.py`: the stretches on a file, a page or a project; `dwell.py`: how long on the window in front). The answer is a fact about
the last hours, so it is read off those, by rule: the stretch in progress, today's top things, this week's. Private windows are never named
(activity already leaves them out), and with nothing recorded it says what it needs rather than guessing.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .dwell import _span

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|ok(?:ay)?|um|hey)[,\s]+)?"
_NOW = re.compile(_LEAD + r"(?:what(?:'s|s|\s+is)?\s+(?:am\s+i|i'?m)\s+(?:working\s+on|doing)(?:\s+(?:right\s+)?now)?|what\s+project\s+(?:am\s+i|i'?m)\s+(?:on|in)|"
                  r"what\s+(?:am\s+i|i'?m)\s+(?:on|in)\s+(?:right\s+)?now|remind\s+me\s+what\s+i'?m\s+(?:working\s+on|doing))\s*[?.!]*\s*$", re.I)
_WAS = re.compile(_LEAD + r"(?:what\s+did\s+i\s+(?:just\s+)?work\s+on(?:\s+(?:earlier|today|lately|recently|this\s+(?:morning|afternoon|evening)|last\s+night))?|what\s+(?:was|were)\s+i\s+(?:just\s+)?(?:working\s+on|doing)(?:\s+(?:earlier|before|a\s+(?:bit|while)\s+ago|this\s+morning|last\s+night))?|"
                  r"what\s+(?:have\s+i\s+been|did\s+i\s+(?:just\s+)?(?:do|work\s+on))\s+(?:working\s+on|doing)?(?:\s+(?:today|lately|recently|so\s+far|this\s+(?:morning|afternoon|evening)))?|"
                  r"what\s+have\s+i\s+been\s+(?:working\s+on|up\s+to)(?:\s+(?:today|lately|recently|so\s+far|this\s+(?:morning|afternoon|evening)))?)\s*[?.!]*\s*$", re.I)
_WEEK = re.compile(_LEAD + r"(?:what\s+(?:have\s+i\s+been|did\s+i)\s+(?:working\s+on|work\s+on|doing|do)\s+(?:this\s+week|over\s+the\s+(?:last|past)\s+(?:few\s+)?days|the\s+(?:last|past)\s+week)"
                   r"|what\s+(?:projects|things)\s+(?:have\s+i\s+been|am\s+i)\s+(?:working\s+on|on)\s+(?:this\s+week|lately))\s*[?.!]*\s*$", re.I)


def parse(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or len(t) > 90:
        return None
    if _WEEK.match(t):
        return "week"
    if _NOW.match(t):
        return "now"
    if _WAS.match(t):
        return "today"
    return None


def _days(now: float, n: int) -> List[str]:
    return [time.strftime("%Y-%m-%d", time.localtime(now - i * 86400)) for i in range(n)]


def top(activity: Any, now: float, days: int, limit: int = 3, min_s: float = 300.0) -> List[Tuple[str, float]]:
    """[(name, seconds)] over the last `days` days: projects if any were seen, else files/pages."""
    if activity is None:
        return []
    want = set(_days(now, days))
    try:
        with activity._lock:
            topics = list(activity.topics.values())
    except Exception:
        return []
    tot: Dict[Tuple[str, str], float] = {}
    for t in topics:
        s = sum(v for d, v in (t.by_day or {}).items() if d in want)
        if s >= min_s:
            tot[(t.kind, t.name)] = s
    have_projects = any(k == "project" for k, _ in tot)
    rows = [(n, s) for (k, n), s in tot.items() if (k == "project" or not have_projects)]
    rows.sort(key=lambda r: -r[1])
    return rows[:limit]


def _list(rows: List[Tuple[str, float]]) -> str:
    return ", ".join(f"{n} ({_span(s)})" for n, s in rows)


def answer(rt: Any, kind: str, now: Optional[float] = None, fallback: bool = True) -> str:
    now = time.time() if now is None else now
    a = getattr(rt, "activity", None)
    d = getattr(rt, "dwell", None)
    parts: List[str] = []
    cur = getattr(a, "_cur", None) if a is not None else None
    if kind == "now":
        if cur is not None and getattr(cur, "subject", ""):
            what = cur.subject + (f" in {cur.project}" if getattr(cur, "project", "") else "")
            mins = max(0.0, now - cur.start)
            parts.append(f"Right now: {what}" + (f", for {_span(mins)}" if mins >= 60 else "") + ".")
        elif d is not None and getattr(d, "process", "") and not getattr(d, "private", False):
            try:
                parts.append(d.answer("here", now))
            except Exception:
                pass
        tdy = top(a, now, 1)
        if tdy:
            parts.append(f"Today so far: {_list(tdy)}.")
    elif kind == "week":
        wk = top(a, now, 7, 4)
        if wk:
            parts.append(f"This week: {_list(wk)}.")
    else:                                              # "what have I been working on" / "what was I doing"
        tdy = top(a, now, 1, 4)
        if tdy:
            parts.append(f"Today: {_list(tdy)}.")
        if cur is not None and getattr(cur, "subject", ""):
            parts.append(f"And right now, {cur.subject}" + (f" in {cur.project}" if getattr(cur, "project", "") else "") + ".")
        elif not tdy:
            wk = top(a, now, 7, 3)
            if wk:
                parts.append(f"Nothing long today, but this week: {_list(wk)}.")
    if parts:
        return " ".join(parts)
    if not fallback:
        return ""
    return ("I haven't got enough yet: I only note a stretch once you've been on one file, page or project for a few minutes. "
            "Tell me what it is -- say “remember that I'm working on ...” -- and I'll keep it.")
