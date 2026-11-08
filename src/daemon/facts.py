"""
src/daemon/facts.py

Short factual questions answered from Wikipedia, with the source named.

    "Who was Ada Lovelace?"   ->  "Ada Lovelace was an English mathematician ... (Wikipedia)"

A language model asked "who is X" will say something fluent whether or not it
knows; a made-up CV in memory is how this started. For the narrow case of a
short "who/what is X" question, a Wikipedia summary is a better answer than a
guess, and it is quoted, not paraphrased.

It is OFF by default and only does anything when switched on (Settings, or
facts.enabled in daemon.json), because it sends the topic -- and only the
topic -- to en.wikipedia.org. It runs only for something said TO ASH, never for
talk overheard in the room.

  * a topic with a page: its first sentences, with "(Wikipedia)"
  * a topic with several meanings: asks which one
  * a topic with no page: says so (a 404 from Wikipedia)
  * no connection, or too slow: says nothing and lets ASH answer as usual,
    rather than claiming it looked
  * anything that is not clearly "who/what is <a thing>" is left alone: times,
    dates, weather, sums, "what is your name", "what is this", questions about
    the user.
"""

from __future__ import annotations

import logging
import re
import threading
import urllib.error
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .lookup import _wiki_fetch

logger = logging.getLogger("ash.daemon.facts")


@dataclass
class FactsConfig:
    enabled: bool = False
    timeout_s: float = 3.0
    max_chars: int = 320

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FactsConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                try:
                    setattr(c, k, type(getattr(c, k))(v))
                except (TypeError, ValueError):
                    pass
        return c


_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|quick(?:ly)?)[,\s]+)?(?:(?:can|could)\s+you\s+tell\s+me\s+)?"
_Q = re.compile(_LEAD + r"(?:who(?:'s|\s+is|\s+was|\s+were|\s+are)|what(?:'s|\s+is|\s+are|\s+was|\s+were)|"
                r"where(?:'s|\s+is|\s+was)|tell\s+me\s+about|define|what\s+does\s+(?P<mean>.+?)\s+mean\b)\s*"
                r"(?P<topic>.*?)[\s?.!]*$", re.I)

# Things "what is ..." is asked about that are not Wikipedia topics.
_NOT_TOPICS = {
    "time", "the time", "date", "the date", "day", "the day", "today", "the weather", "weather", "up", "new",
    "wrong", "going on", "happening", "that", "this", "it", "there", "here", "next", "left", "the matter",
    "the problem", "the plan", "the point", "your name", "your job", "your purpose", "your favorite", "your favourite",
    "my name", "my plan", "my schedule", "the news", "the score", "the answer", "the difference", "the weather like",
    "the status", "your status", "the capital", "a good idea", "a good time", "happening today", "your mood",
}
_PERSONAL = re.compile(r"\b(?:my|your|our|me|i|you|we|this|that|these|those|it|its|here|there|today|tonight|tomorrow|"
                       r"yesterday|now|current(?:ly)?|latest|weather|time|date|battery|disk|memory|cpu|screen)\b", re.I)


def topic_of(text: str) -> Optional[str]:
    """The Wikipedia topic of a short "who/what is X" question, or None."""
    t = (text or "").strip()
    if not t or len(t) > 70 or len(t.split()) > 9:
        return None
    m = _Q.match(t)
    if not m:
        return None
    topic = (m.group("mean") or m.group("topic") or "").strip(" ,'\"")
    topic = re.sub(r"^(?:the\s+word\s+)", "", topic, flags=re.I)
    topic = re.sub(r"^(?:a|an|the)\s+", "", topic, flags=re.I).strip()
    low = topic.lower()
    if (re.match(r"(?:capital|population|meaning|price|cost|difference|way|reason|point|purpose|name|age|height|"
                 r"distance|temperature|number|score|answer|plan|problem|status|rate|value|weather)\s+"
                 r"(?:of|between|for|in|to|at)\b", low) or re.match(r"(?:best|weather)\b", low)):
        return None                                  # a question about a quantity, not a topic
    if not topic or low in _NOT_TOPICS or ("the " + low) in _NOT_TOPICS or len(topic) > 48 or len(topic.split()) > 5:
        return None
    if re.search(r"\d", topic) or _PERSONAL.search(topic):
        return None
    if re.search(r"[^\w\s.'&-]", topic):
        return None
    return topic


def _tidy(extract: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", str(extract)).strip()
    s = re.sub(r"\s*\([^)]*\)", "", s)                      # pronunciations and dates in brackets
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", s)
    out = parts[0]
    for p in parts[1:2]:
        if len(out) + 1 + len(p) <= limit:
            out += " " + p
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return out


class Facts:
    def __init__(self, cfg: Optional[FactsConfig] = None,
                 fetch: Optional[Callable[[str, float], Optional[Dict[str, Any]]]] = None):
        self.cfg = cfg or FactsConfig()
        self.fetch = fetch or _wiki_fetch
        self.stats = {"asked": 0, "answered": 0, "missing": 0, "ambiguous": 0, "offline": 0}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def _get(self, title: str) -> Optional[Dict[str, Any]]:
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(title.replace(" ", "_"), safe=""))
        return self.fetch(url, self.cfg.timeout_s)

    def hear(self, text: str) -> Optional[str]:
        """The answer, or None to let ASH answer as it always did."""
        if not self.enabled:
            return None
        topic = topic_of(text)
        if topic is None:
            return None
        self.stats["asked"] += 1
        tries = [topic[:1].upper() + topic[1:]]
        cap = " ".join(w[:1].upper() + w[1:] for w in topic.split())
        if cap not in tries:
            tries.append(cap)
        data: Optional[Dict[str, Any]] = None
        not_found = False
        for title in tries:
            try:
                data = self._get(title)
                not_found = False
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    not_found = True
                    continue
                self.stats["offline"] += 1
                logger.info("Wikipedia answered %s", e.code)
                return None
            except Exception as e:                           # offline, slow, blocked
                self.stats["offline"] += 1
                logger.info("Wikipedia lookup failed: %s", type(e).__name__)
                return None
            if isinstance(data, dict):
                break
        if not isinstance(data, dict):
            if not_found:
                self.stats["missing"] += 1
                return f"I couldn't find a Wikipedia page for “{topic}”."
            self.stats["offline"] += 1
            return None
        if data.get("type") == "disambiguation":
            self.stats["ambiguous"] += 1
            return f"“{topic}” can mean several things on Wikipedia -- which one do you mean?"
        extract = _tidy(data.get("extract") or "", self.cfg.max_chars)
        if not extract:
            self.stats["missing"] += 1
            return f"I couldn't find a Wikipedia page for “{topic}”."
        self.stats["answered"] += 1
        return f"{extract} (Wikipedia)"

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, **self.stats}
