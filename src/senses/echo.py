"""
src/senses/echo.py

ASH hearing itself.

With speakers instead of headphones, the microphone picks up what ASH says
and transcribes it as if the user had said it -- and ASH then answers its own
words ("My current status is warm and upbeat." heard back, scored as
addressed, answered). This keeps what ASH said and when, and the microphone
drops an utterance that

  * was heard while ASH was speaking or just after (the room's echo, and
    the time transcription takes), and
  * is mostly ASH's own words.

Something the user says over ASH ("stop", "no, the other one") is not made
of ASH's words, so it still gets through: talking over it keeps working.
Whisper mishears, so the match is by words in common, not the exact text.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

TAIL_S = 2.5            # still "just after" this long after ASH stopped
LATE_S = 25.0           # a very close match counts this long after (slow transcription)
OVERLAP = 0.6           # share of heard words found in what ASH said
STRONG = 0.8

_STOP = {"the", "a", "an", "and", "or", "to", "of", "is", "it", "i", "you", "in", "on", "for", "that", "this"}


def words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'"))


class EchoGuard:
    def __init__(self, clock: Callable[[], float] = time.time, keep: int = 6):
        self.clock = clock
        self._said: Deque[Dict] = deque(maxlen=keep)     # {"words": set, "start", "end"}
        self._lock = threading.Lock()
        self.dropped = 0

    def said(self, text: str, start: Optional[float] = None, seconds: Optional[float] = None):
        """ASH started saying `text`; `seconds` is a guess at how long."""
        start = self.clock() if start is None else start
        ws = set(words(text))
        if not ws:
            return
        dur = seconds if seconds is not None else max(1.5, len(text) / 14.0)
        with self._lock:
            self._said.append({"words": ws, "start": start, "end": start + dur})

    def done(self, at: Optional[float] = None):
        """ASH stopped speaking (finished, or was cut off)."""
        at = self.clock() if at is None else at
        with self._lock:
            if self._said:
                self._said[-1]["end"] = at

    def is_echo(self, text: str, heard_at: Optional[float] = None, duration: float = 0.0) -> bool:
        """Was this utterance ASH's own voice coming back?"""
        heard_at = self.clock() if heard_at is None else heard_at
        began = heard_at - max(0.0, duration)
        ws = [w for w in words(text) if w not in _STOP] or words(text)
        if not ws:
            return False
        with self._lock:
            said = list(self._said)
        for s in said:
            share = sum(1 for w in ws if w in s["words"]) / len(ws)
            overlaps = began <= s["end"] + TAIL_S and heard_at >= s["start"] - 0.5
            late = heard_at - s["end"] <= LATE_S and heard_at >= s["start"]
            if (overlaps and share >= OVERLAP) or (late and share >= STRONG and len(ws) >= 3):
                self.dropped += 1
                return True
        return False


# One per process: the speaker notes what it says, the microphone checks.
guard = EchoGuard()
