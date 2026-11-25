"""
src/daemon/repair.py

When the person is frustrated, ASH changes how it behaves -- instead of improvising in the same way, with a filler first, and a question back.

25 Nov, 12:49-13:06: "Bruh." x5, "Never mind." x6, "That's wrong, you do have that information", "You are really frustrating me today Ash." ASH went on
exactly as before: "Let me see." ... 20 s ... a confident wrong answer, then "How else might I assist?". Nothing about the frustration reached the
behaviour.

`Repair.hear(text, now)` weighs what the person says (and how often they cancel or rephrase); past a threshold repair mode is on for ten minutes:
one short apology, no thinking fillers, the model is told to answer in a sentence and to say "I don't know" rather than guess, and a turn that
produced nothing gets a plain "I don't know that one, here is what I can do". A thanks, or ten quiet minutes, ends it. The lines that led
to it are kept in state/failures.jsonl, so what went wrong is a list, not a memory of an afternoon.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

THRESHOLD = 3.0
WINDOW_S = 300.0                 # signals count for this long
HOLD_S = 600.0                   # repair mode lasts this long after the last signal

APOLOGY = "Sorry -- that's been a mess. I'll keep it short, and if I don't know something I'll say so instead of guessing."
NOTE = ("[The person is frustrated with you. Answer in one short sentence. If you do not know, say \"I don't know\". "
        "Do not guess, do not ask a question back, do not offer more help, and do not apologise again.]")
NOTHING = "I don't know that one."

# (pattern, weight)
_SIGNALS: List[Tuple[re.Pattern, float]] = [(re.compile(p, re.I), w) for p, w in (
    (r"^\s*(?:ash[,\s]+)?(?:bruh+|ugh+|argh+|wtf|come\s+on|seriously|ffs)\s*[.!?]*\s*$", 1.0),
    (r"\b(?:never\s*mind|forget\s+it|drop\s+it)\b", 1.0),
    (r"\b(?:frustrat\w+|annoy\w+|irritat\w+|fed\s+up|had\s+enough)\b", 3.0),
    (r"\b(?:you(?:'re|\s+are)\s+(?:useless|stupid|dumb|hopeless|terrible|bad|so\s+slow|not\s+listening)|you\s+(?:suck|never\s+listen|don'?t\s+listen))\b", 3.0),
    (r"\b(?:that(?:'s|\s+is)\s+(?:wrong|not\s+(?:right|correct|what\s+i)|unrelated|nonsense)|not\s+what\s+i\s+(?:asked|meant|said)|unrelated\s+to\s+what\s+i\s+ask)", 2.0),
    (r"\byou\s+(?:do|did)\s+(?:have|know|say)\s+(?:that|it|this)\b", 2.0),
    (r"\b(?:stop\s+(?:guessing|making\s+(?:stuff|things)\s+up|lying)|you(?:'re|\s+are)\s+(?:making|lying|hallucinat)\w*)", 2.0),
    (r"\b(?:i\s+(?:just\s+)?(?:said|asked)|i\s+told\s+you)\b[^.?!]{0,40}\b(?:already|before|again|that)\b", 1.5),
    (r"\bwhy\s+(?:are|is)\s+(?:you|this)\s+so\s+(?:slow|dumb|bad)\b", 2.0),
)]
_ENDS = re.compile(r"^\s*(?:ash[,\s]+)?(?:thanks?(?:\s+you)?|thank\s+you|that'?s\s+(?:better|good|right|great|perfect)|perfect|nice\s+one|good\s+(?:job|one)|great|cool|"
                   r"awesome|ok(?:ay)?\s+(?:thanks|cool|good)|i'?m\s+(?:ok|fine|good|not\s+(?:frustrated|angry|upset)))\b", re.I)
_WORKED = re.compile(r"^\s*(?:ash[,\s]+)?(?:thanks?|thank\s+you|that'?s\s+(?:better|good|right|great|perfect)|perfect|nice\s+one|good\s+(?:job|one))\b", re.I)


class Repair:
    def __init__(self, clock: Callable[[], float] = time.time, path: Optional[str] = None):
        self.clock = clock
        self.path = path
        self._signals: Deque[Tuple[float, float, str]] = deque(maxlen=40)
        self.until = 0.0
        self.entered_at = 0.0
        self.recent: Deque[str] = deque(maxlen=6)
        self.stats = {"entered": 0, "ended": 0, "signals": 0, "nothing": 0}

    # ---- hearing ------------------------------------------------------------------------------------------------
    def score(self, text: str, last: str = "") -> Tuple[float, str]:
        t = (text or "").strip()
        if not t or len(t) > 160:
            return 0.0, ""
        best, why = 0.0, ""
        for rx, w in _SIGNALS:
            if rx.search(t) and w > best:
                best, why = w, rx.pattern[:30]
        if last and best == 0.0 and _same(t, last):
            best, why = 1.0, "the same thing again"
        return best, why

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """Take in one user line. Returns the apology the first time repair mode starts, else None."""
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t:
            return None
        last = self.recent[-1] if self.recent else ""
        self.recent.append(t)
        if self.active(now) and _ENDS.match(t):
            self.until = 0.0
            self._signals.clear()
            self.stats["ended"] += 1
            return None
        w, why = self.score(t, last)
        if w <= 0.0:
            return None
        self.stats["signals"] += 1
        self._signals.append((now, w, t))
        total = sum(x[1] for x in self._signals if now - x[0] <= WINDOW_S)
        if total < THRESHOLD:
            return None
        was = self.active(now)
        self.until = now + HOLD_S
        if was:
            return None
        self.entered_at = now
        self.stats["entered"] += 1
        self._log(now)
        return APOLOGY

    # ---- state --------------------------------------------------------------------------------------------------
    def active(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return now < self.until

    def note(self, now: Optional[float] = None) -> str:
        """What the model is told while repair mode is on."""
        return NOTE if self.active(now) else ""

    def end(self):
        self.until = 0.0
        self._signals.clear()

    def _log(self, now: float):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(now), "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
                                     "lines": [x[2] for x in list(self._signals)[-6:]], "said": list(self.recent)[-6:]}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def status(self) -> Dict:
        return {"active": self.active(), **self.stats}


# What a small model says to a frustrated person despite being told not to (25 Nov 13:00-15:10): a comment on the feeling, a different approach,
# a question back, an offer. Each sentence like that is cut; the first real sentence stays, and nothing else.
_FLUFF = re.compile(r"(?:\byou\s+(?:seem|sound|appear|look)\s+(?:frustrated|upset|annoyed|angry)|\blet'?s\s+(?:try|take|look|see|do)\b|\bhow\s+about\b|"
                    r"\bwould\s+you\s+like\b|\bhow\s+(?:can|may|might)\s+i\b|\bfeel\s+free\b|\bi'?m\s+(?:here|happy|glad)\s+to\s+help\b|"
                    r"\bis\s+there\s+anything\b|\bdifferent\s+approach\b|\bi\s+understand\b|\bi\s+apologi[sz]e\b|\bsorry\s+for\b|\bmy\s+apologies\b|"
                    r"\banything\s+else\b|\bwhat\s+(?:else|would)\b|\bdo\s+you\s+want\b)", re.I)


def shorten(text: str) -> str:
    """The reply while repair mode is on: the first sentence that is an answer -- no feelings talk, no offers, no question back. Nothing left: NOTHING."""
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if x.strip()]
    keep = [x for x in parts if not x.endswith("?") and not _FLUFF.search(x)]
    return keep[0] if keep else NOTHING


def _same(a: str, b: str) -> bool:
    from difflib import SequenceMatcher
    wa = re.findall(r"[a-z0-9']+", a.lower())
    wb = re.findall(r"[a-z0-9']+", b.lower())
    return len(wa) >= 3 and len(wb) >= 3 and SequenceMatcher(None, wa, wb).ratio() >= 0.8
