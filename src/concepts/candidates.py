"""
src/concepts/candidates.py

How a concept comes to exist, the way the brain does it (complementary learning systems).

The hippocampus takes in a new thing at once and keeps it as an episode. The neocortex does not: a concept is what is left after the
same kind of thing has come back on separate occasions and replay has averaged the episodes into their gist. One strange moment is
an episode; it never becomes a category by itself, however novel it was.

So when nothing known explains an input, the graph no longer grows a concept on the spot (it did, and one-off lines, window
switches and readings became "concepts"). It writes a candidate here: a running average of what looks alike, how many times it came,
and on how many separate occasions. At sleep a candidate that came back often enough, spread over separate occasions, becomes a
concept whose prototype is the gist; one that never came back fades, like a trace nobody recalled.

An input made mostly of specifics (a duration, a clock time, a path, a link, a run of numbers) is a moment, not a kind of thing, so
it is not even a candidate: it stays an episode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

_SPECIFIC = re.compile(
    r"\bfor \d+(?:\.\d+)?\s*(?:s|sec|secs|seconds?|m|min|mins|minutes?|h|hrs?|hours?)\b|\b\d{1,2}:\d{2}(?::\d{2})?\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|https?://|\bwww\.|[A-Za-z]:\\|(?:^|\s)/(?:[\w.-]+/)+[\w.-]+|"
    r"\b(?:was on|switched to|screen changed)\b", re.I)


def is_episode_like(text: str) -> bool:
    """A moment, not a kind of thing: it names a duration, a clock time, a date, a link, a path, or is mostly numbers."""
    t = str(text or "")
    if not t.strip():
        return False
    if _SPECIFIC.search(t):
        return True
    words = t.split()
    digits = sum(1 for w in words if any(ch.isdigit() for ch in w))
    return len(words) >= 3 and digits >= 2 and digits / len(words) >= 0.3


@dataclass
class Candidate:
    vec: np.ndarray
    n: int = 1                       # times it came
    occasions: int = 1               # on how many separate occasions
    first: float = 0.0
    last: float = 0.0
    texts: List[str] = field(default_factory=list)   # a few of the actual sentences, for a name


class CandidateBuffer:
    def __init__(self, dim: int, min_seen: int = 3, min_occasions: int = 2, gap_s: float = 1200.0,
                 ttl_s: float = 14 * 86400.0, cap: int = 300, join: float = 0.70, per_night: int = 8):
        self.dim = dim
        self.min_seen = min_seen
        self.min_occasions = min_occasions
        self.gap_s = gap_s           # two arrivals closer than this are one occasion
        self.ttl_s = ttl_s
        self.cap = cap
        self.join = join             # how alike two arrivals must be to count as the same kind of thing
        self.per_night = per_night   # the cortex learns slowly: only so many new concepts a night
        self.items: List[Candidate] = []

    def note(self, vec: np.ndarray, text: str, now: float) -> Candidate:
        v = np.asarray(vec, dtype=np.float32)
        best, bi = -1.0, -1
        if self.items:
            sims = np.stack([c.vec for c in self.items]) @ v
            bi = int(np.argmax(sims))
            best = float(sims[bi])
        if best >= self.join:
            c = self.items[bi]
            m = c.vec * c.n + v
            norm = float(np.linalg.norm(m))
            c.vec = (m / norm).astype(np.float32) if norm > 1e-9 else c.vec
            if now - c.last >= self.gap_s:
                c.occasions += 1
            c.n += 1
            c.last = now
        else:
            c = Candidate(vec=v.copy(), first=now, last=now)
            self.items.append(c)
            if len(self.items) > self.cap:
                self.remove(min(self.items, key=lambda x: (x.n, x.last)))
        t = str(text or "").strip()[:200]
        if t and t not in c.texts and len(c.texts) < 3:
            c.texts.append(t)
        return c

    def ripe(self) -> List[Candidate]:
        r = [c for c in self.items if c.n >= self.min_seen and c.occasions >= self.min_occasions]
        return sorted(r, key=lambda c: (-c.occasions, -c.n))[: self.per_night]

    def remove(self, c: Candidate) -> None:
        self.items = [x for x in self.items if x is not c]      # by identity: a candidate holds an array, == would not do

    def forget(self, now: float) -> int:
        keep = [c for c in self.items if now - c.last < self.ttl_s]
        gone = len(self.items) - len(keep)
        self.items = keep
        return gone

    # ---- persistence: the vectors go in the prototypes file, the rest beside the graph
    def matrix(self) -> np.ndarray:
        return np.stack([c.vec for c in self.items]).astype(np.float32) if self.items else np.zeros((0, self.dim), np.float32)

    def meta(self) -> List[Dict[str, Any]]:
        return [{"n": c.n, "occ": c.occasions, "first": c.first, "last": c.last, "texts": c.texts} for c in self.items]

    def restore(self, mat: Optional[np.ndarray], meta: List[Dict[str, Any]]) -> None:
        self.items = []
        if mat is None or len(meta) != len(mat):
            return
        for v, m in zip(mat, meta):
            self.items.append(Candidate(vec=np.asarray(v, np.float32), n=int(m.get("n", 1)), occasions=int(m.get("occ", 1)),
                                        first=float(m.get("first", 0.0)), last=float(m.get("last", 0.0)),
                                        texts=[str(t) for t in (m.get("texts") or [])][:3]))
