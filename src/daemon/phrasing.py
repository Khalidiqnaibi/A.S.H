"""
src/daemon/phrasing.py

The same acknowledgement twice in a row sounds like a machine: "Opening Chrome." "Opening Word." "Opening Excel." A rule's
answer is kept word for word the first time (so what is said stays predictable, and what is tested stays exact); only when
the same kind of line comes round again within a couple of minutes does its opening word change.

    fresh = Fresh()
    fresh.apply("Okay, a 5-minute timer, starting now.", now)   # unchanged
    fresh.apply("Okay, I'll remind you to stretch at 15:00.", now + 30)   # "Got it -- I'll remind you to ..."
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# opening of a line -> what else it can open with
_ALTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Okay, ", ("Got it -- ", "Sure, ", "Done -- ")),
    ("Opening ", ("Launching ", "Starting ", "Bringing up ")),
    ("Switching to ", ("Bringing up ", "Jumping to ", "Going to ")),
    ("Closing ", ("Shutting down ", "Quitting ")),
    ("Cancelled ", ("Dropped ", "Cleared ")),
]


class Fresh:
    def __init__(self, window_s: float = 120.0):
        self.window_s = window_s
        self._last: Dict[str, Tuple[float, int]] = {}       # opening -> (when last said, which alternative)

    def apply(self, line: str, now: float) -> str:
        for head, alts in _ALTS:
            if not line.startswith(head):
                continue
            last = self._last.get(head)
            if last is None or now - last[0] > self.window_s:
                self._last[head] = (now, -1)                  # the first time: as written
                return line
            i = (last[1] + 1) % len(alts)                     # again soon: the next opening in turn
            self._last[head] = (now, i)
            return alts[i] + line[len(head):]
        return line
