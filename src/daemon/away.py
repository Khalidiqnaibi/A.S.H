"""
src/daemon/away.py

"What's new?", "what did I miss?", "catch me up" -- and what is said on "Welcome back": what actually happened while the user was away.

On 22 Nov at 16:45 ASH said "Welcome back. You were away about 1 hour 8 minutes." Then "whats new while i was away" went to the model, which
answered "away about an hour and a half" -- made up. Everything it needed was already kept, in different places: the reminders that fired, the
questions it saved up, a shutdown the crash watch found, the window the user left open. Nothing joined them. This does, from those, by rule --
and says "nothing came up" when nothing did, instead of inventing a morning.
"""

from __future__ import annotations

import re
import time
from typing import Any, List, Optional

from .dwell import _span

_ASK = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|ok(?:ay)?|um)[,\s]+)?(?:what(?:'s|s|\s+is|\s+has)?\s+(?:new|happened|been\s+going\s+on|have\s+i\s+missed|i\s+(?:miss(?:ed)?|"
    r"(?:have|'ve)\s+missed))(?:\s+(?:while|when)\s+i\s+(?:was|were)\s+(?:away|gone|out|off))?(?:\s+(?:since\s+i\s+left|today|so\s+far))?|"
    r"what\s+did\s+i\s+miss(?:\s+while\s+i\s+was\s+(?:away|gone|out))?|"
    r"(?:anything|something)\s+(?:happen|come\s+up|new)\s+(?:while\s+i\s+was\s+(?:away|gone|out)|since\s+i\s+left)|"
    r"catch\s+me\s+up|(?:give\s+me\s+)?(?:a\s+)?(?:quick\s+)?(?:catch[\s-]?up|update|rundown|summary)(?:\s+(?:please|on\s+what\s+i\s+missed))?|"
    r"what\s+happened\s+while\s+i\s+was\s+(?:away|gone|out|off))\s*[?.!]*\s*$", re.I)


def asks_digest(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and len(t) <= 80 and bool(_ASK.match(t))


def _hm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def items(rt: Any, since: float, now: float) -> List[str]:
    """What happened since `since`, as short clauses (nothing for what did not)."""
    out: List[str] = []
    r = getattr(rt, "reminders", None)
    if r is not None:
        try:
            with r._lock:
                fired = [x for x in r.items.values() if x.status == "fired" and x.fired_at >= since]
            for x in sorted(fired, key=lambda x: x.fired_at)[:3]:
                if x.kind == "timer":
                    out.append(f"your {x.label} timer finished at {_hm(x.fired_at)}")
                elif x.kind == "alarm":
                    out.append(f"your {_hm(x.due)} alarm went off")
                else:
                    out.append(f"a reminder went off at {_hm(x.fired_at)}: {x.text}")
            if len(fired) > 3:
                out.append(f"{len(fired) - 3} more reminders went off")
        except Exception:
            pass
    c = getattr(rt, "crashwatch", None)
    if c is not None:
        try:
            crashed = [e for e in c.unreported(now) if e["t"] >= since - 300]
            if crashed:
                out.append(f"the computer shut down unexpectedly at {_hm(crashed[0]['t'])}")
        except Exception:
            pass
    w = getattr(rt, "wonder", None)
    if w is not None:
        try:
            n = len(w.pending())
            if n:
                out.append(f"I saved {n} question{'s' if n != 1 else ''} to ask you when it's a good time")
        except Exception:
            pass
    return out


def left_on(rt: Any, since: float) -> str:
    """What was on screen when they left, if it is known (never a private window)."""
    d = getattr(rt, "dwell", None)
    if d is None:
        return ""
    try:
        from .social import friendly_app
        with d._lock:
            rows = [x for x in d.recent if x[1] <= since + 90]
        if rows:
            start, end, app, subj = rows[-1]
            return f"you left {subj} ({friendly_app(app)}) open" if subj else f"you left {friendly_app(app)} open"
    except Exception:
        pass
    return ""


def digest(rt: Any, since: float, now: Optional[float] = None, welcome: bool = False) -> str:
    """The digest. For the welcome (`welcome=True`): only when something is worth saying, else ''; asked for: always an answer."""
    now = time.time() if now is None else now
    got = items(rt, since, now)
    if not got:
        return "" if welcome else f"Nothing came up while you were away ({_span(max(0.0, now - since))}) that I know of."
    head = "While you were away: " if not welcome else ""
    s = head + "; ".join(got) + "."
    s = s[0].upper() + s[1:]
    gone = left_on(rt, since)
    if gone and not welcome:
        s += f" Before that, {gone}."
    return s
