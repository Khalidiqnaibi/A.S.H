"""
src/vision/presence.py

Who is here -- as a state, not a recording.

Fed, every few seconds, with the people the camera sees (a name, or None for a face nobody taught it), this keeps one
thing: who was last seen when. From that it says when someone ARRIVED (seen after being gone for a while), when they
LEFT (not seen for `away_after_s`), and when a STRANGER came into view (an unknown face seen on `confirm` readings in a
row, so one bad angle is not a stranger). Nothing about a frame is kept: no picture, no face, no numbers -- names and times.

A stranger "holds" for `hold_s` after the last sighting: during that time ASH does not read private things aloud
(mail, calendar, clipboard, the day's briefing).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class Presence:
    def __init__(self, away_after_s: float = 90.0, hold_s: float = 120.0, confirm: int = 2):
        self.away_after_s = float(away_after_s)
        self.hold_s = float(hold_s)
        self.confirm = max(1, int(confirm))
        self.present: Dict[str, float] = {}          # name -> when last seen
        self.unknown_until = 0.0                     # a stranger counts as here until then
        self._run = 0                                # consecutive readings with an unknown face
        self._last_update = 0.0
        self._stranger_on = False
        self.stats = {"updates": 0, "arrived": 0, "left": 0, "strangers": 0}

    def update(self, people: List[Optional[str]], now: float) -> List[Tuple[str, str]]:
        """`people`: for each face seen, the name or None. Returns [("arrived"|"left"|"stranger", name)]."""
        events: List[Tuple[str, str]] = []
        names = {p for p in people if p}
        unknown = any(p is None for p in people)
        for n in sorted(names):
            was = self.present.get(n)
            if was is None or now - was > self.away_after_s:
                events.append(("arrived", n))
                self.stats["arrived"] += 1
            self.present[n] = now
        for n in sorted(self.present):
            if n not in names and now - self.present[n] > self.away_after_s:
                events.append(("left", n))
                self.stats["left"] += 1
                del self.present[n]
        if unknown:
            self._run += 1
            if self._run >= self.confirm:
                if not self._stranger_on or now > self.unknown_until:
                    events.append(("stranger", ""))
                    self.stats["strangers"] += 1
                self._stranger_on = True
                self.unknown_until = now + self.hold_s
        else:
            self._run = 0
        if self._stranger_on and now > self.unknown_until:
            self._stranger_on = False
        self._last_update = now
        self.stats["updates"] += 1
        return events

    def here(self, now: float) -> List[str]:
        """Who has been seen within the away window, most recent first."""
        return [n for n, t in sorted(self.present.items(), key=lambda kv: -kv[1]) if now - t <= self.away_after_s]

    def stranger(self, now: float) -> bool:
        return now <= self.unknown_until

    def fresh(self, now: float, within_s: float) -> bool:
        """Whether there has been a reading recently enough to answer from."""
        return self._last_update > 0 and now - self._last_update <= within_s

    def clear(self):
        self.present.clear()
        self.unknown_until, self._run, self._stranger_on = 0.0, 0, False
