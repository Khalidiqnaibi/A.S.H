"""
src/daemon/backoff.py

Talking less, by learning what is not wanted.

The attention gate learns per broad kind of remark ("a free-moment remark",
"a question"), so waving off jokes made ASH shyer about everything it says
at a pause. This keeps a score per *specific* thing -- the joke activity,
the mood check, the story offer, "stale" questions, guesses about people --
from how the user reacted to it:

    rejected, dismissed   1     ("no thanks", swiped away)
    deferred              0.5   ("not now")
    accepted, engaged     clears the score: it is welcome

Three points within a week benches that thing for three days. Then it may
come back, once; welcomed, it stays; waved off again, it is benched again.
Nothing is ever turned off for good by this -- that is daemon.json's job.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

DAY = 86400.0
WEIGHT = {"rejected": 1.0, "dismissed": 1.0, "deferred": 0.5}
WELCOME = ("accepted", "engaged")


class Backoff:
    def __init__(self, clock: Callable[[], float] = time.time, threshold: float = 3.0,
                 window_s: float = 7 * DAY, bench_s: float = 3 * DAY):
        self.clock = clock
        self.threshold = threshold
        self.window_s = window_s
        self.bench_s = bench_s
        self._hits: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)
        self.until: Dict[str, float] = {}
        self.stats = {"times_benched": 0, "welcomed": 0}

    def note(self, name: str, outcome: str, now: Optional[float] = None) -> bool:
        """Record a reaction to `name`. True if this benched it."""
        if not name:
            return False
        now = self.clock() if now is None else now
        if outcome in WELCOME:
            self._hits.pop(name, None)
            self.stats["welcomed"] += 1
            return False
        w = WEIGHT.get(outcome, 0.0)
        if w <= 0:
            return False
        q = self._hits[name]
        q.append((now, w))
        while q and now - q[0][0] > self.window_s:
            q.popleft()
        if sum(x for _, x in q) >= self.threshold:
            self.until[name] = now + self.bench_s
            q.clear()
            self.stats["times_benched"] += 1
            return True
        return False

    def benched(self, name: str, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self.until.get(name, 0.0) > now

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        return {"benched": sorted(k for k, t in self.until.items() if t > now), **self.stats}

    # -- persistence (the wonderer keeps it in its own file) -------------------
    def dump(self) -> Dict[str, Any]:
        return {"until": dict(self.until),
                "hits": {k: list(v) for k, v in self._hits.items() if v}}

    def load(self, d: Optional[Dict[str, Any]]):
        d = d or {}
        self.until = {str(k): float(v) for k, v in (d.get("until") or {}).items()}
        for k, v in (d.get("hits") or {}).items():
            self._hits[str(k)] = deque((float(a), float(b)) for a, b in v)
