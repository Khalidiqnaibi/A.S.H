"""
src/daemon/misheard.py

"I said Opera, not Obra." -- learning the words the recogniser keeps getting wrong, from the person correcting it, and putting them right
in every later transcript.

Speech recognition mishears the same words the same way: "Opera" as "Obra" ("Open Obra Ash"), "Khalid" as "Khaled", "Ash" as "Aloe"/"Dash". Each one
is a line that fails or goes to the model. The user already says what they meant -- "I said Opera, not Obra" -- and nothing was listening for it.
Now a pair like that is kept (state/misheard.json), applied to transcripts from then on (the whole word, case kept), and the word they meant goes
into the hint the recogniser is given, so it leans the right way.

Narrow on purpose: a short phrase (a word or three) on each side, said as a correction of what was heard, never a longer sentence. A word the
user's own correction maps cannot map to itself, and an "undo" is one phrase away ("forget that correction").
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.misheard")

_PHRASE = r"[A-Za-z0-9][A-Za-z0-9'.\- ]{0,28}?"
_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:no|nope|no no|wait|actually|ok(?:ay)?|sorry)[,\s]+)*"
# Only wordings that are plainly about what was SAID: "it's fine, not great" is a sentence about fine and great, not a correction of the recogniser.
_SAID_NOT = re.compile(_LEAD + r"(?:i\s+(?:said|meant|was\s+saying)|the\s+word\s+(?:is|was))\s+\"?(?P<meant>" + _PHRASE + r")\"?"
                       r"[,\s]+(?:and\s+)?not\s+\"?(?P<heard>" + _PHRASE + r")\"?\s*[.!]*\s*$", re.I)
_NOT_SAID = re.compile(_LEAD + r"(?:i\s+(?:didn'?t|did\s+not)\s+say|not)\s+\"?(?P<heard>" + _PHRASE + r")\"?"
                       r"[,\s]+(?:i\s+(?:said|meant)|the\s+word\s+(?:is|was))\s+\"?(?P<meant>" + _PHRASE + r")\"?\s*[.!]*\s*$", re.I)
# "its called Co.Te.De. actually" / "its called Co.Te.De., not CODT": a NAME the recogniser will not know. Kept as a hint word (it leans towards it), and when
# "not Y" is said too, Y becomes a mishearing of it. "it's called a day" is not a name: nothing starting with an article or a function word.
_CALLED = re.compile(_LEAD + r"(?:(?:it(?:'s|\s+is|\s+was)|that(?:'s|\s+is)|they(?:'re|\s+are)|its)\s+(?:called|spelled|spelt|written|named|pronounced)|the\s+(?:name|word)\s+is)\s+\"?(?P<term>" + _PHRASE + r")\"?"
                     r"(?:[,\s]+(?:and\s+)?not\s+\"?(?P<heard>" + _PHRASE + r")\"?)?(?:\s+(?:actually|though|by\s+the\s+way|btw))?\s*[.!]*\s*$", re.I)
_FORGET = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:forget|undo|remove|drop|ignore)\s+(?:that|the\s+last)\s+(?:correction|mishearing|spelling)\s*[.!]*\s*$", re.I)


# Never a pair that would change ordinary speech: a function word, or the start of a clause ("it was fine", "that is") -- "I said yes, not no".
_FUNCTION = re.compile(r"^(?:yes|yeah|no|nope|not|the|and|or|but|is|was|are|it|it's|its|that|this|those|these|he|she|they|we|you|i|there|here|to|of|in|on|at|for|with|so|if|do|does|did|can|will|would)(?![A-Za-z'])", re.I)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip(" .,!?\"'")).strip()


class Misheard:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.pairs: Dict[str, str] = {}               # heard (lowercase) -> meant (as said)
        self.terms: Dict[str, str] = {}               # names and terms the user spelled out (lowercase -> as written): hint words
        self.last: Optional[str] = None
        self._lock = threading.RLock()
        self._rx: Optional[re.Pattern] = None
        self._load()

    # -- keeping -------------------------------------------------------------------
    def _load(self):
        if not self.path:
            return
        try:
            from src.memory.atomic import load_json
            d = load_json(self.path, default=None, logger=logger)
            if isinstance(d, dict):
                self.pairs = {str(k).lower(): str(v) for k, v in (d.get("pairs") or {}).items()}
                self.terms = {str(k).lower(): str(v) for k, v in (d.get("terms") or {}).items()}
                self._rx = None
        except Exception:
            logger.debug("misheard pairs not loaded", exc_info=True)

    def _save(self):
        if not self.path:
            return
        try:
            from src.memory.atomic import write_json
            write_json(self.path, {"pairs": self.pairs, "terms": self.terms}, indent=1, ensure_ascii=False)
        except Exception:
            logger.debug("misheard pairs not saved", exc_info=True)

    # -- learning ------------------------------------------------------------------
    @staticmethod
    def parse(text: str) -> Optional[Tuple[str, str]]:
        """(heard, meant) from "I said Opera, not Obra" / "not Obra, Opera" / "I didn't say Obra, I said Opera"; else None."""
        t = (text or "").strip()
        if not t or len(t) > 90:
            return None
        for rx in (_SAID_NOT, _NOT_SAID):
            m = rx.match(t)
            if m:
                heard, meant = _clean(m.group("heard")), _clean(m.group("meant"))
                if (heard and meant and heard.lower() != meant.lower() and len(heard.split()) <= 3 and len(meant.split()) <= 3
                        and len(heard) >= 3 and len(meant) >= 3 and not _FUNCTION.match(heard) and not _FUNCTION.match(meant)):
                    return heard, meant
        return None

    @staticmethod
    def parse_term(text: str) -> Optional[Tuple[str, Optional[str]]]:
        """(term, heard-or-None) from "its called Co.Te.De., not CODT" / "its called NoSQL actually"; else None."""
        t = (text or "").strip()
        if not t or len(t) > 90:
            return None
        m = _CALLED.match(t)
        if not m:
            return None
        term = re.sub(r"\s+", " ", m.group("term")).strip(" ,\"'")
        if "." not in term.rstrip("."):
            term = term.rstrip(".")                          # a sentence's full stop; but "Co.Te.De." is an abbreviation and keeps its last dot
        heard = _clean(m.group("heard") or "")
        if (not term or len(term) < 3 or len(term.split()) > 3 or _FUNCTION.match(term) or re.match(r"^(?:an?|some|one)\b", term, re.I)
                or re.fullmatch(r"(?:day|night|it|that|this|time|bed|off)", term, re.I)):
            return None
        if heard and (heard.lower() == term.lower() or len(heard) < 3 or len(heard.split()) > 3 or _FUNCTION.match(heard)):
            heard = ""
        return term, (heard or None)

    def hear(self, text: str) -> Optional[str]:
        """A correction (kept, and said back) or 'forget that correction'; None for anything else."""
        if _FORGET.match(text or ""):
            with self._lock:
                if self.last and self.last in self.pairs:
                    gone = self.pairs.pop(self.last)
                    self._rx = None
                    self._save()
                    last, self.last = self.last, None
                    return f"Okay, I won't change “{last}” to “{gone}” any more."
            return "I haven't learned a correction I can take back."
        got = self.parse(text)
        if got is None:
            named = self.parse_term(text)
            if named is None:
                return None
            term, heard = named
            with self._lock:
                self.terms[term.lower()] = term
                if heard:
                    self.pairs[heard.lower()] = term
                    self.last = heard.lower()
                    self._rx = None
                self._save()
            return (f"Got it -- \u201c{term}\u201d, and when I hear \u201c{heard}\u201d I'll take it as that." if heard
                    else f"Got it -- \u201c{term}\u201d. I'll listen for it.")
        heard, meant = got
        with self._lock:
            self.pairs[heard.lower()] = meant
            self.last = heard.lower()
            self._rx = None
            self._save()
        return f"Got it -- when I hear “{heard}” I'll take it as “{meant}”."

    # -- applying ------------------------------------------------------------------
    def _pattern(self) -> Optional[re.Pattern]:
        if self._rx is None and self.pairs:
            keys = sorted(self.pairs, key=len, reverse=True)
            self._rx = re.compile(r"(?<![\w'])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![\w'])", re.I)
        return self._rx

    def fix(self, text: str) -> str:
        """`text` with every learned mishearing put right (whole words, the first letter's case kept)."""
        with self._lock:
            rx = self._pattern()
            if rx is None or not text:
                return text

            def one(m: "re.Match[str]") -> str:
                meant = self.pairs.get(m.group(0).lower(), m.group(0))
                return meant[:1].upper() + meant[1:] if m.group(0)[:1].isupper() and not meant[:1].isupper() else meant
            return rx.sub(one, text)

    def words(self) -> List[str]:
        """What the recogniser should lean towards: the words the user said they meant."""
        with self._lock:
            return list(dict.fromkeys(list(self.terms.values()) + list(self.pairs.values())))[:30]

    def status(self) -> Dict[str, int]:
        return {"pairs": len(self.pairs), "terms": len(self.terms)}
