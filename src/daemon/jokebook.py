"""
src/daemon/jokebook.py

Which jokes ASH has told, kept across restarts, so a joke is not told twice until the rest have been.

The free-moment joke was picked by a counter held in memory: it started at the first joke every time ASH started (the programmer
and the loaf of bread), and only moved when the person answered. Told again and again, it became "you said the same joke 6 times",
and when the person said so the model replied "I haven't told it six times yet", with nothing to check. Here every telling is
written down; a joke is chosen from the ones told least (the oldest telling first); and "you told me that one already" is answered
from the record and retires that joke for good.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional


class JokeBook:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, rng: Optional[random.Random] = None):
        self.path = path
        self.clock = clock
        self.rng = rng or random.Random()
        self._lock = threading.RLock()
        self.told: Dict[str, List[float]] = {}          # joke -> when it was told
        self.retired: List[str] = []                    # told too often, said so: never again
        self.last: Optional[str] = None                 # the last one told
        self._load()

    # -- storage ----------------------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.told = {str(k): [float(t) for t in v] for k, v in (d.get("told") or {}).items()}
            self.retired = [str(x) for x in (d.get("retired") or [])]
            self.last = d.get("last") or None
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"told": self.told, "retired": self.retired, "last": self.last}, fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # -- the record ---------------------------------------------------------------------------------
    def record(self, joke: str, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        with self._lock:
            self.told.setdefault(joke, []).append(float(now))
            del self.told[joke][:-20]
            self.last = joke
            self.save()

    def times(self, joke: str) -> int:
        with self._lock:
            return len(self.told.get(joke, ()))

    def pick(self, pool: Iterable[str]) -> Optional[str]:
        """The joke told least, the oldest telling first; ties are broken at random. Never a retired one."""
        with self._lock:
            jokes = [j for j in dict.fromkeys(pool) if j not in self.retired]
            if not jokes:
                return None
            best = min((len(self.told.get(j, ())), (self.told[j][-1] if self.told.get(j) else 0.0)) for j in jokes)
            return self.rng.choice([j for j in jokes if (len(self.told.get(j, ())), (self.told[j][-1] if self.told.get(j) else 0.0)) == best])

    def complaint(self, said_times: Optional[int] = None) -> str:
        """"You told me that one already": from the record. The last joke told is retired, so it does not come back."""
        with self._lock:
            j = self.last
            if not j:
                return "I don't have a joke on record to check, but I'll keep from repeating myself."
            n = self.times(j)
            if j not in self.retired:
                self.retired.append(j)
            self.save()
            short = j if len(j) <= 60 else j[:57].rstrip() + "..."
            if n >= 2:
                return f"You're right -- I've told “{short}” {n} times. That one's retired."
            return f"Fair -- that was “{short}”, and I won't tell it again."
