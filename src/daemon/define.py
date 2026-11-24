"""
src/daemon/define.py

"Define X", "what's the definition of segue", "what does ubiquitous mean": a dictionary answer, with the source named.

    "What's the definition of segue?"  ->  "segue (noun): a smooth transition from one thing to another. (verb): to move without interruption. (Wiktionary)"

It was answered "I couldn't find a Wikipedia page for 'definition of segue'" -- the facts rule took a definition for an article title -- or went to the
model, which defines words fluently whether or not it knows them. Wiktionary's own definition endpoint is the source: only the WORD leaves the
machine (to en.wiktionary.org), only when something is said TO ASH, and offline or slow it says nothing and the model answers as before.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .lookup import _wiki_fetch

logger = logging.getLogger("ash.daemon.define")

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|quick(?:ly)?|can\s+you|could\s+you)[,\s]+)*"
def _w(n: int) -> str:
    return r"\"?(?:the\s+word\s+)?(?P<w%d>[A-Za-z][A-Za-z'\- ]{1,28}?)\"?" % n


_DEFINE = re.compile(_LEAD + r"(?:define\s+" + _w(1) + r"|(?:what(?:'s|\s+is|\s+are)|tell\s+me|give\s+me|get\s+me)\s+(?:the\s+)?(?:definition|meaning)\s+of\s+" + _w(2) +
                     r"|(?:definition|meaning)\s+of\s+" + _w(3) + r"|what\s+does\s+" + _w(4) + r"\s+mean|what\s+do\s+you\s+mean\s+by\s+" + _w(5) + r")"
                     r"\s*[?.!]*\s*$", re.I)
_NOT_WORDS = frozenset("this that it these those here there me you him her them us what who why how when where which life love".split())
_PERSONAL = re.compile(r"\b(?:my|your|our|me|i|you|we|this|that|these|those|it|its|here|there|today|now)\b", re.I)


@dataclass
class DefineConfig:
    enabled: bool = False
    timeout_s: float = 3.0
    max_chars: int = 300

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DefineConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                try:
                    setattr(c, k, type(getattr(c, k))(v))
                except (TypeError, ValueError):
                    pass
        return c


def parse(text: str) -> Optional[str]:
    """The word asked about, or None. One word, or two ("ice cream", "post hoc"); never "what does this mean" or a sentence."""
    t = (text or "").strip()
    if not t or len(t) > 70:
        return None
    m = _DEFINE.match(t)
    if not m:
        return None
    w = next((m.group(k) for k in ("w1", "w2", "w3", "w4", "w5") if m.group(k)), "")
    w = re.sub(r"\s+", " ", w).strip(" '\"-")
    if not w or len(w.split()) > 2 or w.lower() in _NOT_WORDS or _PERSONAL.search(w):
        return None
    return w


def _plain(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", str(s or ""))
    return re.sub(r"\s+", " ", html.unescape(s)).strip(" ;,")


def render(word: str, data: Any, limit: int = 300) -> Optional[str]:
    """Wiktionary's definitions for `word` as one or two short sentences, or None when there is no English entry."""
    if not isinstance(data, dict):
        return None
    entries = data.get("en") or []
    parts: List[str] = []
    for e in entries[:3]:
        pos = str(e.get("partOfSpeech") or "").lower()
        defs = [d for d in (_plain(x.get("definition")) for x in (e.get("definitions") or [])[:3]) if d]
        if not defs:
            continue
        parts.append((f"({pos}) " if pos else "") + defs[0].rstrip("."))
    if not parts:
        return None
    s = f"{word}: " + "; ".join(parts) + "."
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return s + " (Wiktionary)"


class Define:
    def __init__(self, cfg: Optional[DefineConfig] = None, fetch: Optional[Callable[[str, float], Optional[Dict[str, Any]]]] = None):
        self.cfg = cfg or DefineConfig()
        self.fetch = fetch or _wiki_fetch
        self.stats = {"asked": 0, "answered": 0, "missing": 0, "offline": 0}

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def hear(self, text: str) -> Optional[str]:
        if not self.enabled:
            return None
        word = parse(text)
        if word is None:
            return None
        self.stats["asked"] += 1
        url = "https://en.wiktionary.org/api/rest_v1/page/definition/" + urllib.parse.quote(word.lower().replace(" ", "_"), safe="")
        try:
            data = self.fetch(url, self.cfg.timeout_s)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.stats["missing"] += 1
                return f"I couldn't find “{word}” in Wiktionary -- check the spelling, or say “spell” and I'll go letter by letter."
            self.stats["offline"] += 1
            return None
        except Exception as e:                                   # offline, slow, blocked: say nothing, let ASH answer as it did
            self.stats["offline"] += 1
            logger.info("Wiktionary lookup failed: %s", type(e).__name__)
            return None
        say = render(word, data, self.cfg.max_chars)
        if say is None:
            self.stats["missing"] += 1
            return f"I couldn't find an English definition of “{word}” in Wiktionary."
        self.stats["answered"] += 1
        return say

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, **self.stats}
