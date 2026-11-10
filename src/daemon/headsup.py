"""
src/daemon/headsup.py

Heads-up cards: what is coming up soon, shown in the console without being
asked, and never spoken (speaking is the reminders' job). Built fresh from
what ASH already keeps each time the console asks:

    reminder / timer / alarm   due within the next 30 minutes
    calendar event             starting within the next 60 minutes (or on now)
    home                       what is waiting for you when you get home

Each card is {"kind", "title", "at" (epoch seconds, 0 if none), "in" (a short
"in 12 min" / "now" / "" ), "sub"}, soonest first, at most five. The console
draws them as small cards; the idle clock screen shows them too.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _in(delta_s: float) -> str:
    if delta_s <= 60:
        return "now"
    m = int(round(delta_s / 60.0))
    if m < 60:
        return f"in {m} min"
    h, mm = divmod(m, 60)
    return f"in {h} h" if mm == 0 else f"in {h} h {mm} min"


def cards(runtime, now: float, reminder_window_s: float = 1800.0, event_window_s: float = 3600.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    r = getattr(runtime, "reminders", None)
    if r is not None:
        try:
            for it in list(r.items.values()):
                if it.status != "set" or it.due - now > reminder_window_s:
                    continue
                if it.kind == "timer":
                    title, sub = f"{it.label} timer", ""
                elif it.kind == "alarm":
                    title, sub = "Alarm", datetime.fromtimestamp(it.due).strftime("%H:%M")
                else:
                    title, sub = it.text, "Reminder"
                out.append({"kind": it.kind, "title": title, "at": it.due, "in": _in(it.due - now), "sub": sub})
        except Exception:
            pass
    cal = getattr(runtime, "calendar", None)
    if cal is not None:
        try:
            day = datetime.fromtimestamp(now).date()
            for e in cal.events_on(day):
                if e.allday:
                    continue
                ts = e.start.timestamp()
                end = e.end.timestamp() if e.end else ts + 1800
                if ts - now > event_window_s or end < now:
                    continue
                out.append({"kind": "event", "title": e.summary, "at": ts, "in": _in(ts - now),
                            "sub": (e.location or e.start.strftime("%H:%M"))})
        except Exception:
            pass
    a = getattr(runtime, "arrivals", None)
    if a is not None:
        try:
            p = a.pending()
            if p:
                out.append({"kind": "home", "title": str(p[0]["what"]), "at": 0, "in": "at home",
                            "sub": f"+{len(p) - 1} more" if len(p) > 1 else "Waiting for you"})
        except Exception:
            pass
    out.sort(key=lambda c: (c["at"] == 0, c["at"]))
    return out[:5]
