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
from .activity import chat_phrase, chat_site

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|ok(?:ay)?|um|hey)[,\s]+)?"
# ("what project am I ACTIVELY doing" went to the model for 54 s: the adverbs and "at the moment" are part of the question)
_ADV = r"(?:(?:actively|actually|currently|presently|even|really|exactly|still|just)\s+)?"
_NOWT = r"(?:\s+(?:(?:right\s+)?now|at\s+the\s+moment|at\s+present|these\s+days|currently|today))?"
_NOW = re.compile(_LEAD + r"(?:what(?:'s|s|\s+is)?\s+(?:am\s+i|i'?m)\s+" + _ADV + r"(?:working\s+on|doing|busy\s+with|up\s+to)" + _NOWT + r"|"
                  r"what\s+(?:project|thing|task|stuff)s?\s+(?:am\s+i|i'?m)\s+" + _ADV + r"(?:on|in|doing|working\s+on|busy\s+with)" + _NOWT + r"|"
                  r"what\s+(?:am\s+i|i'?m)\s+(?:on|in)\s+(?:right\s+)?now|remind\s+me\s+what\s+i'?m\s+" + _ADV + r"(?:working\s+on|doing))\s*[?.!]*\s*$", re.I)
_WAS = re.compile(_LEAD + r"(?:what\s+did\s+i\s+(?:just\s+)?work\s+on(?:\s+(?:earlier|today|lately|recently|this\s+(?:morning|afternoon|evening)|last\s+night))?|what\s+(?:was|were)\s+i\s+(?:just\s+)?(?:working\s+on|doing)(?:\s+(?:earlier|before|a\s+(?:bit|while)\s+ago|this\s+morning|last\s+night))?|"
                  r"what\s+(?:have\s+i\s+been|did\s+i\s+(?:just\s+)?(?:do|work\s+on))\s+(?:working\s+on|doing)?(?:\s+(?:today|lately|recently|so\s+far|this\s+(?:morning|afternoon|evening)))?|"
                  r"what\s+have\s+i\s+been\s+(?:working\s+on|up\s+to)(?:\s+(?:today|lately|recently|so\s+far|this\s+(?:morning|afternoon|evening)))?)\s*[?.!]*\s*$", re.I)
_WEEK = re.compile(_LEAD + r"(?:what\s+(?:have\s+i\s+been|did\s+i)\s+(?:working\s+on|work\s+on|doing|do)\s+(?:this\s+week|over\s+the\s+(?:last|past)\s+(?:few\s+)?days|the\s+(?:last|past)\s+week)"
                   r"|what\s+(?:projects|things)\s+(?:have\s+i\s+been|am\s+i)\s+(?:working\s+on|on)\s+(?:this\s+week|lately))\s*[?.!]*\s*$", re.I)


# "How much did I work on A.S.H this week?" / "how long was I on YouTube today?" / "how much time did I spend on NoSQL?": the same per-day seconds the
# weekly review adds up, for the one thing asked about. Nothing for "how long have I been on this" (that is the window in front: dwell.py).
_HOW = re.compile(_LEAD + r"how\s+(?:much|long)(?:\s+time)?\s+(?:did\s+i\s+(?:work|spend|spent)|have\s+i\s+(?:worked|spent|been)|was\s+i|do\s+i\s+spend)"
                  r"(?:\s+(?:working|time|spent))?\s+(?:on|in|with|at)\s+(?P<name>[\w.' &-]{2,40}?)"
                  r"(?:\s+(?P<span>today|yesterday|this\s+week|all\s+week|so\s+far|lately|this\s+month|over\s+the\s+(?:last|past)\s+(?:week|few\s+days)))?\s*[?.!]*\s*$", re.I)
_PRONOUN = re.compile(r"^(?:this|that|it|here|there|them|these|those|the\s+(?:screen|window|page|tab))$", re.I)


def parse_time(text: str) -> Optional[Tuple[str, int, str]]:
    """(thing, days, how it was put) for "how much did I work on X this week"; None for anything else."""
    t = (text or "").strip()
    if not t or len(t) > 100:
        return None
    m = _HOW.match(t)
    if not m:
        return None
    name = " ".join(m.group("name").split()).strip(" .")
    if not name or _PRONOUN.match(name):
        return None
    span = re.sub(r"\s+", " ", (m.group("span") or "").lower())
    if span in ("this week", "all week", "lately", "over the last week", "over the past week", "over the last few days", "over the past few days"):
        return name, 7, "this week" if span in ("this week", "all week") else "over the last week"
    if span == "this month":
        return name, 14, "in the last two weeks (that is as far back as I keep it)"
    if span == "yesterday":
        return name, -1, "yesterday"
    return name, 1, "today"


def time_on(activity: Any, name: str, days: int, now: float) -> Tuple[float, List[str]]:
    """(seconds, the topics that matched) for what the name is part of (or is part of); days -1 is yesterday alone."""
    from .activity import norm
    if activity is None:
        return 0.0, []
    want = set(_days(now, 2)[1:2]) if days == -1 else set(_days(now, days))
    q = norm(name)
    total, hit = 0.0, []
    try:
        with activity._lock:
            topics = list(activity.topics.values())
    except Exception:
        return 0.0, []
    for t in topics:
        n = norm(t.name)
        if not q or not n or not (q in n or (len(n) >= 3 and n in q)):
            continue
        s = sum(v for d, v in (t.by_day or {}).items() if d in want)
        if s > 0:
            total += s
            hit.append(t.name)
    return total, hit


def answer_time(rt: Any, name: str, days: int, when: str, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    secs, hit = time_on(getattr(rt, "activity", None), name, days, now)
    if secs < 60:
        return f"I haven't tracked anything on {name} {when} -- I only note a stretch once you've been on it for a few minutes."
    return f"About {_span(secs)} on {name} {when}."


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


def taught(kb: Any, name: str) -> str:
    """'your first novel' for a name the person explained ("Branch is im working on my first novel"), short enough to say beside it; else ''."""
    if kb is None or not name:
        return ""
    try:
        from .activity import norm
        from .wondering import _yours
        t = (kb.terms or {}).get(norm(name)) or {}
        d = str(t.get("definition") or "")
        if not d or d.startswith("ASH itself"):
            return ""
        s = _yours(d)
        return s if 1 <= len(s.split()) <= 6 else ""
    except Exception:
        return ""


def top(activity: Any, now: float, days: int, limit: int = 3, min_s: float = 300.0, kb: Any = None) -> List[Tuple[str, float]]:
    """[(name, seconds)] over the last `days` days: projects if any were seen, else files/pages. With `kb`, a name the person explained is said
    with what it is ("Branch, your first novel")."""
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
    chats = {t.name: t.chat for t in topics if getattr(t, "chat", "")}
    have_projects = any(k == "project" for k, _ in tot)
    rows = [(chat_phrase(chats[n], n) if n in chats else n, s) for (k, n), s in tot.items() if (k == "project" or not have_projects)]
    rows.sort(key=lambda r: -r[1])
    out = []
    for n, s in rows[:limit]:
        what = taught(kb, n)
        out.append((f"{n}, {what}" if what else n, s))
    return out


# "How's the novel going?" / "how is my thesis going": the thing the person explained ("Branch is my first novel"), found by what it is, and the
# time on it this week with the days in a row.
_GOING = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:so[,\s]+)?how(?:'s|\s+is)\s+(?:my|the)\s+(?P<x>[a-z][\w' -]{2,30}?)\s+(?:going|coming\s+along|doing)\s*[?.!]*\s*$",
                    re.I)


def going(rt: Any, text: str, now: Optional[float] = None) -> Optional[str]:
    m = _GOING.match((text or "").strip())
    if not m:
        return None
    now = time.time() if now is None else now
    x = m.group("x").lower()
    kb = getattr(getattr(rt, "wonder", None), "kb", None)
    a = getattr(rt, "activity", None)
    name = ""
    try:
        for t in (kb.terms or {}).values() if kb is not None else []:
            d, n = str(t.get("definition") or "").lower(), str(t.get("term") or "")
            if re.search(r"\b" + re.escape(x) + r"\b", d) or n.lower() == x:
                name = n
                break
    except Exception:
        name = ""
    if not name:
        return None
    week, hit = time_on(a, name, 7, now)
    if week < 60:
        return f"I haven't seen you on {name} this week."
    days = 0
    try:
        from datetime import date, timedelta
        from .activity import norm
        with a._lock:
            by: Dict[str, float] = {}
            for t in a.topics.values():
                if norm(t.name) == norm(name) or t.name in hit:
                    for d, v in (t.by_day or {}).items():
                        by[d] = by.get(d, 0.0) + v
        d = date.fromtimestamp(now)
        while by.get(d.isoformat(), 0.0) >= 600:
            days, d = days + 1, d - timedelta(days=1)
    except Exception:
        pass
    return (f"{_span(week)} on {name} this week" + (f", {days} days in a row" if days >= 2 else "") + ".")


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
            site = chat_site(getattr(cur, "project", "") or "")
            what = chat_phrase(site, cur.subject) if site else cur.subject + (f" in {cur.project}" if getattr(cur, "project", "") else "")
            mins = max(0.0, now - cur.start)
            parts.append(f"Right now: {what}" + (f", for {_span(mins)}" if mins >= 60 else "") + ".")
        elif d is not None and getattr(d, "process", "") and not getattr(d, "private", False):
            try:
                parts.append(d.answer("here", now))
            except Exception:
                pass
        tdy = top(a, now, 1, kb=getattr(getattr(rt, "wonder", None), "kb", None))
        if tdy:
            parts.append(f"Today so far: {_list(tdy)}.")
    elif kind == "week":
        wk = top(a, now, 7, 4, kb=getattr(getattr(rt, "wonder", None), "kb", None))
        if wk:
            parts.append(f"This week: {_list(wk)}.")
    else:                                              # "what have I been working on" / "what was I doing"
        tdy = top(a, now, 1, 4, kb=getattr(getattr(rt, "wonder", None), "kb", None))
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
