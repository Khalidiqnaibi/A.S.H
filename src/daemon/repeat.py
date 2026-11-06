"""
src/daemon/repeat.py

ASH saying the same thing twice.

In one session the model said "I didn't go to privacy mode because there was
no trigger to activate it" three times in a row, and gave the same
"three interactions" list twice -- the second time to a line that was barely
a question. A model answer is checked against what ASH itself said in the
last ten minutes: any sentence it already said is dropped. If nothing new is
left, it says so in a few words instead.

Only for the model's free-form answers. A rule's answer ("Privacy mode on.")
is meant to be the same every time.
"""

from __future__ import annotations

import re
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Callable, Deque, List, Optional, Tuple

WINDOW_S = 600.0
SAME = 0.8                    # share of words two sentences have in common to count as the same
KEEP = 40                     # sentences remembered
NOTHING_NEW = "That's what I just said."

_SENT = re.compile(r"(?<=[.!?])\s+")
_STOP = {"the", "a", "an", "and", "or", "to", "of", "is", "it", "i", "you", "in", "on", "for", "that", "this", "my",
         "me", "be", "as", "so"}


def _words(s: str) -> tuple:
    return tuple(w for w in re.findall(r"[a-z0-9']+", (s or "").lower()) if w not in _STOP)


def _same(a: tuple, b: tuple) -> bool:
    """Same sentence, allowing for a word or two. In order: "the microphone is
    off and the camera is on" is not "the camera is off and the microphone is
    on", though they use the same words."""
    if len(a) < 3 or len(b) < 3:
        return False                      # "Okay." / "Sure." may be said twice
    return SequenceMatcher(None, a, b).ratio() >= SAME


class RepeatGuard:
    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self._said: Deque[Tuple[float, tuple]] = deque(maxlen=KEEP)
        self.stats = {"trimmed": 0, "replaced": 0}

    def _recent(self, now: float) -> List[tuple]:
        return [w for t, w in self._said if now - t <= WINDOW_S]

    def check(self, text: str, now: Optional[float] = None) -> str:
        """`text` without the sentences ASH already said lately; remembers
        what is left."""
        now = self.clock() if now is None else now
        sents = [s.strip() for s in _SENT.split((text or "").strip()) if s.strip()]
        if not sents:
            return text
        recent = self._recent(now)
        fresh, dropped = [], 0
        for s in sents:
            w = _words(s)
            if any(_same(w, r) for r in recent):
                dropped += 1
            else:
                fresh.append(s)
        if not dropped:
            for s in sents:
                self._said.append((now, _words(s)))
            return text
        if not fresh:
            self.stats["replaced"] += 1
            return NOTHING_NEW
        self.stats["trimmed"] += dropped
        for s in fresh:
            self._said.append((now, _words(s)))
        return " ".join(fresh)

    def note(self, text: str, now: Optional[float] = None):
        """Something ASH said that is not a model answer (a rule's line): it
        counts as said, so the model is not allowed to say it again either,
        but it is never trimmed itself."""
        now = self.clock() if now is None else now
        for s in _SENT.split((text or "").strip()):
            if s.strip():
                self._said.append((now, _words(s)))
