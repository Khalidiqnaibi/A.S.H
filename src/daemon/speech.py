"""
src/daemon/speech.py

Interruptible speech: "stop" or "shh" cuts ASH off mid-sentence, "the short
version" asks for less, and "keep it short" makes that the rule.

    stop, shh, hush, quiet, enough,       while ASH is talking (or just
    that's enough, stop talking, ...      after: transcription lags) -- the
                                          sound stops at once, nothing is
                                          said back
    the short version, tl;dr, the gist,   the last answer again, cut to its
    too long, in short                    first sentence or two
    keep it short / be brief              from now on, long answers are
                                          spoken short (the console still
                                          shows all of it)
    you can talk more / full answers      back to normal

All rule-based: the phrases are anchored, whole-utterance matches, so ASH's
own "I'll stop asking about that", heard back through the microphone, is not
a "stop". A bare "stop" while ASH is quiet is left alone -- it may be for a
game, a timer or the music.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("ash.daemon.speech")

_LEAD = r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|alright|please)[,\s]+)?"
_END = r"(?:[,\s]+(?:please|ash|now|thanks|thank you))*[\s.!]*$"

_STOP = re.compile(_LEAD + r"(?:stop(?:\s+(?:talking|it|there|speaking))?|sh+|shush|hush|"
                   r"(?:be\s+)?quiet|enough|that'?s\s+enough|shut\s+up|pause|"
                   r"(?:ok(?:ay)?|alright)(?:[,\s]+(?:ok(?:ay)?|alright))+|i\s+get\s+it|got\s+it)" + _END, re.I)
# Words that only ever mean "stop talking" -- the rest ("stop", "pause",
# "got it") are left to the game, the music and the timer when ASH is quiet.
_ONLY_SPEECH = re.compile(_LEAD + r"(?:stop\s+(?:talking|speaking)|sh+|shush|hush|shut\s+up)" + _END, re.I)

_SHORT = re.compile(_LEAD + r"(?:(?:give\s+me\s+|what'?s\s+|just\s+)?(?:the\s+)?(?:short|quick)\s+version|"
                    r"tl;?\s*dr|(?:give\s+me\s+|just\s+)?the\s+gist|too\s+long|in\s+short|"
                    r"(?:can\s+you\s+)?(?:summari[sz]e|shorten)\s+(?:that|it)|shorter|briefly|"
                    r"(?:can\s+you\s+)?say\s+(?:that|it)\s+(?:shorter|briefly|in\s+short))" + _END, re.I)

_BRIEF_ON = re.compile(_LEAD + r"(?:(?:please\s+)?(?:keep|make)\s+(?:it|them|your\s+answers|things|answers)\s+"
                       r"(?:short|brief)(?:er)?|be\s+brief(?:er)?|(?:give\s+me\s+)?short(?:er)?\s+answers|"
                       r"(?:talk|say)\s+less|less\s+talking)"
                       r"(?:\s+(?:from\s+now\s+on|please|for\s+now))*" + _END, re.I)
_BRIEF_OFF = re.compile(_LEAD + r"(?:you\s+can\s+(?:talk|say)\s+more|(?:give\s+me\s+)?(?:longer|full|normal)\s+answers|"
                        r"no\s+need\s+to\s+keep\s+it\s+short|(?:talk|say)\s+more|"
                        r"(?:go\s+)?back\s+to\s+(?:long|full|normal)\s+answers)"
                        r"(?:\s+(?:from\s+now\s+on|please|again))*" + _END, re.I)

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def wants_stop(text: str, speaking: bool) -> bool:
    """A "stop talking". While ASH is quiet, only the words that can mean
    nothing else count."""
    t = (text or "").strip()
    if not t or len(t) > 60:
        return False
    return bool(_STOP.match(t)) if speaking else bool(_ONLY_SPEECH.match(t))


def wants_short(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and len(t) <= 60 and bool(_SHORT.match(t))


def wants_brief(text: str) -> Optional[bool]:
    """True: keep answers short from now on. False: back to full. None: neither."""
    t = (text or "").strip()
    if not t or len(t) > 80:
        return None
    if _BRIEF_OFF.match(t):
        return False
    if _BRIEF_ON.match(t):
        return True
    return None


def words(text: str) -> int:
    return len((text or "").split())


def shorten(text: str, max_words: int = 25, min_words: int = 8) -> str:
    """The first sentence -- or two, if the first is very short -- cut at
    `max_words`. Lists and code blocks after the opening are dropped."""
    t = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    t = t.split("\n\n", 1)[0] if words(t.split("\n\n", 1)[0]) >= min_words else t
    t = re.sub(r"\s*\n\s*[-*•]\s+", "; ", t)       # a bullet list reads as one sentence
    t = re.sub(r":\s*;\s*", ": ", t)
    t = re.sub(r"\s+", " ", t).strip()
    sents = _SENT.split(t)
    out = sents[0] if sents else t
    if words(out) < min_words and len(sents) > 1:
        out = out + " " + sents[1]
    w = out.split()
    if len(w) > max_words:
        out = " ".join(w[:max_words]).rstrip(",;:-") + "…"
    return out.strip()


class SpeechControl:
    """Holds the brief-answers preference (kept in a small file) and the last
    answer, for "the short version"."""

    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 brief_over_words: int = 40, short_window_s: float = 300.0,
                 stop_grace_s: float = 4.0):
        self.path = path
        self.clock = clock
        self.brief = False
        self.brief_over_words = brief_over_words
        self.short_window_s = short_window_s
        self.stop_grace_s = stop_grace_s
        self.last: Optional[Dict[str, Any]] = None        # {"text", "t"}: the last answer
        self.stats = {"stopped": 0, "shortened": 0, "brief_spoken": 0}
        self._lock = threading.Lock()
        self._load()

    # -- the preference --------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self.brief = bool(json.load(fh).get("brief", False))
        except Exception:
            logger.warning("Could not read %s", self.path, exc_info=True)

    def set_brief(self, on: bool):
        self.brief = bool(on)
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"brief": self.brief}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("Could not save %s", self.path, exc_info=True)

    # -- what is spoken --------------------------------------------------------
    def said(self, text: str, meta: Dict[str, Any], now: Optional[float] = None):
        """Every line ASH says. Answers are kept for "the short version"."""
        if meta.get("addressed") and text and not meta.get("short_version"):
            self.last = {"text": text, "t": self.clock() if now is None else now}

    def spoken_form(self, text: str, meta: Dict[str, Any]) -> Optional[str]:
        """What to say out loud instead of `text`, in brief mode; None to say
        it all. Only long answers are cut; a remark is already short."""
        if not self.brief or not meta.get("addressed") or meta.get("short_version"):
            return None
        if words(text) <= self.brief_over_words:
            return None
        s = shorten(text)
        if s == text:
            return None
        self.stats["brief_spoken"] += 1
        return s

    def short_version(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        last = self.last
        if last is None or now - last["t"] > self.short_window_s:
            return "I haven't said anything long just now."
        s = shorten(last["text"])
        if s.rstrip("…") == re.sub(r"\s+", " ", last["text"]).strip():
            return "That was the short version."
        self.stats["shortened"] += 1
        return s

    def status(self) -> Dict[str, Any]:
        return {"brief": self.brief, **self.stats}
