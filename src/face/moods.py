"""
src/face/moods.py

When the face reacts on its own, using only the animations that already exist
(animations/builtin). Two things the face did not do before:

  WHAT YOU SAY    "thank you" -> love, "haha" -> very_happy, "good night" ->
                  sleepy, "I'm sorry" -> acknowledge, "you're wrong" -> sad,
                  "shut up" -> annoyed, "wow" -> surprised, "good morning" ->
                  happy, "hello" -> wink. The face answers before the voice.

  IDLE LIFE       a quiet room is not a still face. After a while with nobody
                  talking ASH looks around, double-blinks, now and then winks
                  or perks up curious; late at night it gets sleepy. Rare, and
                  never twice in a row, so it reads as life and not a screensaver.

Both are rule-based: a regex and a clock, no language model. A user can ask for
any animation directly ("wink", "show me a big smile"); that is voicectl's job.
"""

from __future__ import annotations

import random
import re
from typing import Optional

_W = r"(?:\s|^|[,.!?-])"

# (animation, pattern) -- first match wins, so the stronger feelings come first.
_SAYINGS = [
    ("love", re.compile(r"\b(?:thank\s*you(?:\s+so\s+much)?|thanks(?:\s+a\s+lot)?|thx|i\s+love\s+you|love\s+you|you(?:'re|\s+are)\s+(?:the\s+best|awesome|amazing|great|wonderful)|good\s+(?:job|bot)|well\s+done|nice\s+work)\b", re.I)),
    ("very_happy", re.compile(r"\b(?:ha(?:ha)+|lol|lmao|rofl|hehe+|that'?s\s+(?:funny|hilarious)|we\s+did\s+it|i\s+(?:won|passed|got\s+the\s+job)|yay+|woo+hoo+)\b", re.I)),
    ("sad", re.compile(r"\b(?:you(?:'re|\s+are)\s+(?:wrong|useless|dumb|stupid|bad)|that'?s\s+wrong|i\s+hate\s+you|you\s+(?:suck|failed)|bad\s+(?:job|bot))\b", re.I)),
    ("annoyed", re.compile(r"\b(?:shut\s+up|be\s+quiet|stop\s+talking|go\s+away|you'?re\s+annoying|enough)\b", re.I)),
    ("surprised", re.compile(r"\b(?:wow+|whoa+|no\s+way|oh\s+my\s+god|omg|seriously\?|unbelievable|holy\s+\w+)\b", re.I)),
    ("sleepy", re.compile(r"\b(?:good\s*night|night\s+night|i'?m\s+(?:so\s+)?(?:tired|exhausted|sleepy)|i'?m\s+going\s+to\s+(?:bed|sleep)|sleep\s+well)\b", re.I)),
    ("happy", re.compile(r"\b(?:good\s+morning|morning\s+ash|i'?m\s+(?:so\s+)?(?:happy|excited|glad)|great\s+news|it\s+worked|that\s+worked|perfect|awesome|nice|cool)\b", re.I)),
    ("acknowledge", re.compile(r"\b(?:i'?m\s+sorry|sorry\s+about\s+that|my\s+bad|oops|apologi[sz]e)\b", re.I)),
    ("curious", re.compile(r"\b(?:guess\s+what|i\s+have\s+(?:a\s+)?(?:question|idea|news)|can\s+i\s+tell\s+you|you\s+won'?t\s+believe)\b", re.I)),
    ("wink", re.compile(r"^\s*(?:hey|hi|hello|yo|sup|hiya|howdy)\b(?:\s+(?:there|ash))?\s*[.!]*\s*$", re.I)),
    ("sad", re.compile(r"\b(?:i'?m\s+(?:so\s+)?(?:sad|upset|depressed|lonely|stressed|anxious)|bad\s+day|i\s+(?:failed|lost)\b|rough\s+day)\b", re.I)),
]


def reaction_to(text: str) -> Optional[str]:
    """The animation a sentence from the user calls for, or None."""
    t = (text or "").strip()
    if not t or len(t) > 160:
        return None
    for anim, rx in _SAYINGS:
        if rx.search(t):
            return anim
    return None


# ------------------------------------------------------------------ idle life
IDLE_FIRST_S = 45.0          # quiet this long before the first bit of life
IDLE_GAP_S = (60.0, 180.0)   # then one every one to three minutes
_ACTS = (("look_around", 5), ("double_blink", 4), ("curious", 2), ("wink", 1))


class IdleLife:
    """Picks the next small thing to do while nobody is talking."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self._next_at: Optional[float] = None
        self._last: Optional[str] = None

    def reset(self):
        self._next_at = None

    def step(self, now: float, idle_s: float, busy: bool = False, hour: Optional[int] = None) -> Optional[str]:
        """The animation to play now, or None. `idle_s` = seconds since anyone
        spoke; `busy` = ASH is answering or speaking."""
        if busy or idle_s < IDLE_FIRST_S:
            self._next_at = None
            return None
        if self._next_at is None:
            self._next_at = now
        if now < self._next_at:
            return None
        self._next_at = now + self.rng.uniform(*IDLE_GAP_S)
        if hour is not None and (hour >= 23 or hour < 5):
            pick = "sleepy"
        else:
            names = [n for n, _ in _ACTS if n != self._last]
            weights = [w for n, w in _ACTS if n != self._last]
            pick = self.rng.choices(names, weights)[0]
        self._last = pick
        return pick
