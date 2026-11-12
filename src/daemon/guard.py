"""
src/daemon/guard.py

Guard mode: while you are away, the camera watches, and your phone hears about it.

    guard mode on / watch the house / keep an eye on the place / I'm leaving, guard the room
    guard mode off / stop guarding / I'm back                    -- or your own face coming back (camera presence)

While it is on, every couple of seconds it looks at the camera sensor's own latest frame:
  * someone the camera doesn't know (presence, when vision.presence is on)  -> "Someone I don't know is in view (14:32)."
  * movement -- the picture differing from the room as it usually looks, two looks in a row (a slowly updated
    background, so a one-look flicker or camera noise is not movement, and a lamp left on stops counting after a while)
                                                                              -> "Movement in front of the camera (14:32)."
Each kind at most once every five minutes. The alert goes to your phone (phone.py: KDE Connect, or the ntfy address you
set); when you come back it tells you what happened. No picture is kept or sent -- only those words and the time.

It needs the camera: it turns the camera on when asked (not while privacy mode is on) and back to how it was after.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|ok(?:ay)?)[,\s]+)*"
_END = r"\s*(?:please|now|for\s+me|while\s+i'?m\s+(?:out|away|gone))?\s*[.!]*\s*$"
_ON = re.compile(_LEAD + r"(?:(?:i'?m\s+(?:leaving|heading\s+out|going\s+out)[,.\s]+(?:can\s+you\s+)?)?"
                 r"(?:guard\s+mode\s+on|turn\s+on\s+guard\s+mode|start\s+guard(?:ing|\s+mode)?|guard\s+(?:the\s+)?(?:house|room|place|flat|apartment|desk)|"
                 r"watch\s+(?:the\s+)?(?:house|room|place|flat|apartment|door|desk)|keep\s+an\s+eye\s+on\s+(?:the\s+)?(?:house|room|place|flat|apartment|things)))" + _END, re.I)
_OFF = re.compile(_LEAD + r"(?:guard\s+mode\s+off|turn\s+off\s+guard\s+mode|stop\s+guard(?:ing|\s+mode)?|stop\s+watching(?:\s+the\s+(?:house|room|place))?|"
                  r"i'?m\s+(?:back|home)(?:\s+now)?)" + _END, re.I)


def _small_gray(frame, size: int = 48):
    import numpy as np
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a[:, :, :3].mean(axis=2)
    h, w = a.shape[:2]
    rows = np.linspace(0, h - 1, size).astype(int)
    cols = np.linspace(0, w - 1, size).astype(int)
    return a[np.ix_(rows, cols)].astype(np.float32)


class Guard:
    def __init__(self, notify: Callable[[str], bool], can_alert: Callable[[], bool] = lambda: True,
                 clock: Callable[[], float] = time.time, cooldown_s: float = 300.0, look_every_s: float = 2.0,
                 pixel_diff: float = 25.0, changed_fraction: float = 0.04):
        self.notify = notify
        self.can_alert = can_alert
        self.clock = clock
        self.cooldown_s, self.look_every_s = cooldown_s, look_every_s
        self.pixel_diff, self.changed_fraction = pixel_diff, changed_fraction
        self.active = False
        self.since = 0.0
        self.events: List[Tuple[float, str, bool]] = []            # (when, what, phone told)
        self._prev = None
        self._moving = 0
        self._last_kind: Dict[str, float] = {}
        self._looked = 0.0
        self.camera_was_on: Optional[bool] = None
        self.stats = {"sessions": 0, "alerts": 0, "unsent": 0}

    @staticmethod
    def parse(text: str) -> Optional[str]:
        t = (text or "").strip()
        if len(t) > 120:
            return None
        if _ON.match(t):
            return "on"
        if _OFF.match(t):
            return "off"
        return None

    def start(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        if self.active:
            return "Guard mode is already on."
        self.active, self.since = True, now
        self.events, self._prev, self._moving, self._last_kind = [], None, 0, {}
        self.stats["sessions"] += 1
        reach = ("I'll send anything I see to your phone." if self.can_alert() else
                 "I have no way to reach your phone (set up KDE Connect or phone.ntfy_url), so I'll tell you when you're back.")
        return "Guard mode on -- I'm watching. " + reach

    def stop(self, now: Optional[float] = None, welcome: bool = False) -> str:
        now = self.clock() if now is None else now
        if not self.active:
            return "Guard mode isn't on."
        self.active = False
        mins = max(1, int((now - self.since) // 60))
        head = "Welcome back -- guard mode off." if welcome else "Guard mode off."
        if not self.events:
            return f"{head} Nothing happened in the {mins} minute{'s' if mins != 1 else ''} you were away."
        items = "; ".join(f"{what.rstrip('.')} at {datetime.fromtimestamp(t).strftime('%H:%M')}" for t, what, _ in self.events[:5])
        more = f" and {len(self.events) - 5} more" if len(self.events) > 5 else ""
        return f"{head} While you were away: {items}{more}."

    def _alert(self, kind: str, what: str, now: float) -> Optional[str]:
        last = self._last_kind.get(kind)
        if last is not None and now - last < self.cooldown_s:
            return None
        self._last_kind[kind] = now
        text = f"{what} ({datetime.fromtimestamp(now).strftime('%H:%M')})"
        sent = False
        try:
            sent = bool(self.notify("ASH guard: " + text))
        except Exception:
            sent = False
        self.events.append((now, what, sent))
        self.stats["alerts" if sent else "unsent"] += 1
        return text

    def look(self, frame, now: Optional[float] = None, stranger: bool = False, user_here: bool = False) -> Optional[str]:
        """One look while on. Returns "welcome" if the user came back, an alert text if one was raised, else None."""
        now = self.clock() if now is None else now
        if not self.active or now - self._looked < self.look_every_s:
            return None
        self._looked = now
        if user_here and now - self.since > 60:
            return "welcome"
        if stranger:
            return self._alert("stranger", "Someone I don't know is in view", now)
        if frame is None:
            return None
        try:
            import numpy as np
            g = _small_gray(frame)
        except Exception:
            return None
        bg = self._prev                                   # the room as it usually looks, updated slowly
        if bg is None or bg.shape != g.shape:
            self._prev = g
            return None
        changed = float((np.abs(g - bg) > self.pixel_diff).mean())
        self._prev = bg * 0.9 + g * 0.1 if changed >= self.changed_fraction else bg * 0.7 + g * 0.3
        self._moving = self._moving + 1 if changed >= self.changed_fraction else 0
        if self._moving >= 2:
            self._moving = 0
            return self._alert("motion", "Movement in front of the camera", now)
        return None

    def status(self) -> Dict[str, Any]:
        return {"active": self.active, "events": len(self.events), **self.stats}
