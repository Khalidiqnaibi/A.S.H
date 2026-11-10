"""
src/daemon/voiceid.py

Knowing the user's voice from a guest's, roughly.

Off until the user asks: "learn my voice" starts it, and the next few
things they say into the microphone are averaged into an enrolled print
(src/senses/voiceprint.py says what that is and is not). After that, each
utterance heard is compared with it:

    user      close to the enrolled print
    guest     clearly not: someone else is talking to ASH
    unknown   in between, too short to tell, no print, or not enrolled --
              and unknown is treated as the user: being refused your own
              request by a misheard voice is worse than a guest getting an
              answer, so only a clear mismatch makes someone a guest

A guest still gets answers, but not actions taken on the user's behalf --
opening or closing apps, watching things, the house, research jobs -- and
nothing a guest says is learned as being about the user (who someone is,
what their events are). "Forget my voice" deletes the enrolled print.
Typed console input is always the user.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ash.daemon.voiceid")

_LEARN = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:learn|remember|get to know)\s+(?:my|what my)\s+"
                    r"voice(?:\s+sounds like)?\s*[.!]?\s*$", re.I)
_FORGET = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:forget|delete|stop recogni[sz]ing)\s+"
                     r"my voice\s*[.!]?\s*$", re.I)


def is_request(text: str) -> bool:
    """"learn my voice" / "forget my voice"."""
    t = (text or "").strip()
    return bool(_LEARN.match(t) or _FORGET.match(t))


@dataclass
class VoiceIDConfig:
    enabled: bool = True
    samples: int = 6                 # utterances averaged when learning
    min_threshold: float = 0.80      # a match is never accepted below this similarity
    margin: float = 3.0              # standard deviations below the enrolment spread
    guest_band: float = 0.10         # this far below the match threshold before it is someone else

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "VoiceIDConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                cur = getattr(c, k)
                try:
                    setattr(c, k, bool(v) if isinstance(cur, bool) else type(cur)(v))
                except (TypeError, ValueError):
                    pass
        return c


def _cos(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / n) if n else 0.0


class VoiceID:
    def __init__(self, cfg: Optional[VoiceIDConfig] = None, path: Optional[str] = None):
        self.cfg = cfg or VoiceIDConfig()
        self.path = path
        self.centroid: Optional[List[float]] = None
        self.threshold: float = self.cfg.min_threshold
        self._learning: Optional[List[List[float]]] = None
        self.stats = {"user": 0, "guest": 0, "unknown": 0}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            c = raw.get("centroid")
            if isinstance(c, list) and c:
                self.centroid = [float(x) for x in c]
                self.threshold = max(self.cfg.min_threshold, float(raw.get("threshold", self.cfg.min_threshold)))
        except Exception:
            logger.exception("Voice print unreadable; not enrolled")

    def _save(self):
        if not self.path:
            return
        try:
            if self.centroid is None:
                if os.path.exists(self.path):
                    os.remove(self.path)
                return
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"centroid": self.centroid, "threshold": self.threshold}, fh)
        except Exception:
            logger.exception("Could not save the voice print")

    @property
    def enrolled(self) -> bool:
        return self.centroid is not None

    @property
    def learning(self) -> bool:
        return self._learning is not None

    # -- requests --------------------------------------------------------------
    def hear_request(self, text: str) -> Optional[str]:
        if not self.cfg.enabled:
            return None
        t = (text or "").strip()
        if _LEARN.match(t):
            self._learning = []
            return (f"Okay. Talk to me as you normally would -- the next {self.cfg.samples} things you say into "
                    f"the microphone teach me your voice.")
        if _FORGET.match(t):
            had = self.enrolled
            self.centroid, self._learning = None, None
            self._save()
            return "Done, I've forgotten your voice." if had else "I didn't have your voice saved."
        return None

    def learn(self, vec: Optional[List[float]]) -> Optional[str]:
        """While learning, take one utterance's print. Returns what to say when
        done (or if the samples disagree too much to trust)."""
        if self._learning is None or not vec:
            return None
        self._learning.append(list(vec))
        if len(self._learning) < self.cfg.samples:
            return None
        s = np.asarray(self._learning, dtype=float)
        self._learning = None
        c = s.mean(axis=0)
        c = c / (np.linalg.norm(c) or 1.0)
        sims = np.array([_cos(x, c) for x in s])
        # Two people's samples averaged still sit near their own middle; it is
        # how far the samples are from EACH OTHER that shows the mix.
        pair_min = min(_cos(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s)))
        if pair_min < 0.7:
            return "Those didn't sound like one voice to me. Say \"learn my voice\" to try again, on your own."
        self.centroid = [float(x) for x in c]
        self.threshold = max(self.cfg.min_threshold, float(sims.mean() - self.cfg.margin * sims.std()))
        self._save()
        return "Got it, I know your voice now. Say \"forget my voice\" any time."

    def who(self, vec: Optional[List[float]]) -> str:
        if not self.cfg.enabled or self.centroid is None or not vec:
            self.stats["unknown"] += 1
            return "unknown"
        sim = _cos(vec, self.centroid)
        r = ("user" if sim >= self.threshold else
             "guest" if sim < self.threshold - self.cfg.guest_band else "unknown")
        self.stats[r] += 1
        return r

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "enrolled": self.enrolled, "learning": self.learning,
                "threshold": round(self.threshold, 3), **self.stats}
