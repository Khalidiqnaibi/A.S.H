"""
src/daemon/courtesy.py

"Thanks", "will do thanks", "later", "good night", "kk": answered at once, by rule, with a short line -- no language model.

Today these went to the model: "Later" took 27 seconds and came back as "You just switched back to Discord... I told you to stand up
after an hour and a half" (ASH had said nothing of the kind), and "will do thanks" took 23 seconds and came back as a telling-off.
A person who says goodbye does not want a thought; they want "later". It is the last rule in the chain, so anything that is waiting
for an answer (an offer, a question ASH asked, a game) has the line first: this only takes what nothing else wanted.

Deliberately narrow: only short lines made of nothing but courtesy words. "Thanks, now what's the weather" is not one.
Not "yes", "ok", "sure" or "alright": those answer a question, and which one is not for this module to guess.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

_PUNCT = re.compile(r"[^\w\s']")

_THANKS_WORDS = frozenset("thanks thank thx ty tysm cheers appreciated appreciate".split())
_THANKS_EXTRA = frozenset("""you a lot so much very many it man bro mate dude ash buddy ok okay alright cool great nice awesome
    perfect sweet yeah yep will do sure thing bet got that again for everything the help""".split())
_ACK_LINES = frozenset({"k", "kk", "kay", "okie", "okey", "got it", "gotcha", "noted", "makes sense", "i see", "cool", "nice", "aight",
                        "will do", "on it", "sure thing", "no worries", "np", "all good", "fair enough", "understood"})
_BYE = re.compile(r"^(?:ok(?:ay)?\s+|alright\s+|well\s+|cool\s+|ash\s+)*(?:later|laters|see\s+(?:you|ya)(?:\s+(?:later|soon|tomorrow))?|"
                  r"cya|bye(?:\s+bye)?|goodbye|talk\s+(?:to\s+you\s+)?later|ttyl|gotta\s+go|i(?:'m|\s+am)\s+(?:off|out|heading\s+out|"
                  r"going)|heading\s+out|catch\s+you\s+later|peace|peace\s+out|take\s+care)(?:\s+(?:ash|man|bro|mate))*$")
_NIGHT = re.compile(r"^(?:ok(?:ay)?\s+)?(?:good\s*night|goodnight|gn|night(?:\s+night)?|nighty\s+night|sleep\s+well)(?:\s+(?:ash|man|bro))*$")
_BRB = re.compile(r"^(?:brb|be\s+right\s+back|back\s+in\s+a\s+(?:bit|sec|minute|moment)|one\s+sec(?:ond)?|give\s+me\s+a\s+(?:sec|minute|moment))$")

_LINES: Dict[str, List[str]] = {
    "thanks": ["Anytime.", "You got it.", "No problem.", "Happy to."],
    "bye": ["Later.", "See you.", "Catch you later.", "Alright, later."],
    "night": ["Night. Sleep well.", "Good night.", "Night."],
    "brb": ["I'll be here.", "Take your time.", "Sure."],
    "ack": ["Okay.", "Got it.", "Mm-hm."],
}


def _norm(text: str) -> str:
    t = _PUNCT.sub(" ", str(text or "").lower().replace("’", "'"))
    t = re.sub(r"^\s*(?:hey\s+)?ash[\s,]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


class Courtesy:
    def __init__(self, clock: Optional[Callable[[], float]] = None):
        self.clock = clock
        self._recent: Dict[str, Deque[str]] = {k: deque(maxlen=2) for k in _LINES}
        self.stats = {"answered": 0}

    def kind(self, text: str) -> Optional[str]:
        t = _norm(text)
        words = t.split()
        if not words or len(words) > 6 or len(t) > 40:
            return None
        if _NIGHT.match(t):
            return "night"
        if _BRB.match(t):
            return "brb"
        if _BYE.match(t):
            return "bye"
        if any(w in _THANKS_WORDS for w in words) and all(w in _THANKS_WORDS or w in _THANKS_EXTRA for w in words):
            return "thanks"
        if t in _ACK_LINES:
            return "ack"
        return None

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        k = self.kind(text)
        if k is None:
            return None
        options = [x for x in _LINES[k] if x not in self._recent[k]] or _LINES[k]
        line = options[self.stats["answered"] % len(options)]
        if k == "night" and now is not None and 5 <= datetime.fromtimestamp(now).hour < 18:
            line = "Alright, later."                      # "night" at noon is a goodbye, not a bedtime
        self._recent[k].append(line)
        self.stats["answered"] += 1
        return line
