"""
src/daemon/learning.py

Everything ASH is learning about the user, in one place they can look at.

A profile that cannot be inspected is a liability. This gathers, for the
console's "What I'm learning" panel, what each part of ASH holds right now:

    likes and dislikes   the strongest tastes (taste.py), said or seen
    browsing             whether tab tracking is on, paused, and how much it holds
    interests, focus     the ledger's reading of it (browsing.py)
    reactions            how often what it volunteers lands (social.py)

Lines, not objects: each is a sentence the panel prints as it is, with the same
words the voice answers use. Nothing here is computed by a model. Cached for 30
seconds, because the console asks often and the ledger is a database.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

_CACHE: Dict[int, tuple] = {}
TTL_S = 30.0


def _lines(rt, now: float) -> List[str]:
    out: List[str] = []
    t = getattr(rt, "taste", None)
    if t is not None:
        try:
            rows = t.items(t.user_key, 0, "", 6)
            if rows:
                out.append("Tastes: " + "; ".join(f"{'likes' if r['score'] > 0 else 'dislikes'} {r['item']}" for r in rows[:6]))
        except Exception:
            pass
    b = getattr(rt, "browsing", None)
    sensor = b._s() if b is not None else None
    if sensor is not None:
        st = sensor.status()
        state = "paused" if st.get("paused") else "watching"
        led = getattr(sensor, "ledger", None)
        if led is not None:
            c = led.counts()
            out.append(f"Browser tabs: {state}; holding {c['episodes']} recent stretches and {c['sites']} older site totals "
                       f"(site and category only; {int(led.retention_days)} days of detail).")
            try:
                goals = b.goal_progress(now)
                if goals:
                    out.append("Goals: " + "; ".join(b.goal_line(g) for g in goals))
                ignored = sorted(getattr(sensor.cls, "ignored", ()) or ())
                if ignored:
                    out.append("Keeping out: " + ", ".join(ignored) + " (plus banks, mail, chat, health and similar, always)")
            except Exception:
                pass
            try:
                ints = b.interests(now, 4)
                if ints:
                    out.append("Interests: " + ", ".join(f"{s} ({cat})" for s, cat, _w, _d in ints))
                f = b.focus_profile(now)
                if f and f.get("enough"):
                    out.append(f"Focus: mostly {f['from']:02d}:00-{f['to']:02d}:00 (over {f['days']} days)")
            except Exception:
                pass
    lr = getattr(rt, "learner", None)
    if lr is not None and int(getattr(lr, "updates", 0)) >= 5:
        out.append(f"Reactions: about {int(round(100 * lr.p_general(now)))}% of what I volunteer lands well "
                   f"({int(lr.updates)} reactions).")
    return out


def snapshot(rt, now: Optional[float] = None) -> Dict[str, Any]:
    """{"lines": [...], "forget": "..."} for the panel, or {} when there is nothing to show."""
    now = time.time() if now is None else now
    hit = _CACHE.get(id(rt))
    if hit is not None and now - hit[0] < TTL_S:
        return hit[1]
    try:
        lines = _lines(rt, now)
    except Exception:
        lines = []
    val: Dict[str, Any] = ({"lines": lines, "forget": "Say \"forget my browsing\", \"forget Interstellar\" or "
                            "\"forget how I react\" to delete any of it."} if lines else {})
    _CACHE[id(rt)] = (now, val)
    return val
