"""
src/daemon/wordtools.py

Small things about words, by rule -- no model, nothing leaves the machine unless a dictionary is asked for (see define.py).

    spell Segway / how do you spell segue / how is "Khalid" spelled     "S, E, G, W, A, Y."
    spell it again / say it again slowly / spell that                    the last word, again
    spell it out loud                                                    the same

"Spell Segway" went to the model, which took 28 seconds and answered 'To spell Segway, it would be spelled as "Segway."'. A word spelled
back is a letter at a time, and that is all a person asking wants.
"""

from __future__ import annotations

import re
from typing import Optional

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|ok(?:ay)?|um|can\s+you|could\s+you|please)[,\s]+)*"
def _w(n: int) -> str:
    """One word (or a short phrase) the user wants spelled, as its own named group."""
    return r"\"?(?P<w%d>[A-Za-z0-9][A-Za-z0-9'.\- ]{0,28}?)\"?" % n


_SPELL = re.compile(_LEAD + r"(?:spell(?:\s+out)?(?:\s+the\s+word)?\s+" + _w(1) + r"|how\s+(?:do\s+(?:you|i)|would\s+you|to)\s+spell\s+(?:the\s+word\s+)?" + _w(2) +
                    r"|how\s+(?:is|do\s+you\s+spell)\s+(?:the\s+word\s+)?" + _w(3) + r"\s+spelled|what'?s\s+the\s+spelling\s+of\s+" + _w(4) + r")"
                    r"(?:\s+for\s+me|\s+please|\s+out\s+loud)*\s*[?.!]*\s*$", re.I)
_AGAIN = re.compile(_LEAD + r"(?:spell\s+(?:it|that|the\s+word)(?:\s+(?:again|out|out\s+loud|slowly))?|(?:say|spell)\s+(?:it|that)\s+(?:again|slowly|letter\s+by\s+letter)|"
                    r"letter\s+by\s+letter|one\s+more\s+time\s+(?:spelled|slowly))\s*[?.!]*\s*$", re.I)
_PRONOUN = frozenset("it that this".split())
_NAMES = {" ": "space", "-": "hyphen", "'": "apostrophe", ".": "dot"}


def letters(word: str) -> str:
    """'Segway' -> 'S, E, G, W, A, Y'."""
    out = []
    for ch in word:
        out.append(_NAMES.get(ch, ch.upper()))
    return ", ".join(out)


class WordTools:
    def __init__(self):
        self.last: Optional[str] = None             # the last word spelled, so "spell it again" has one

    def spell(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if not t or len(t) > 80:
            return None
        m = None if _AGAIN.match(t) else _SPELL.match(t)           # "spell it again" is the last word again, not a word called "it again"
        word = ""
        if m:
            word = next((m.group(k) for k in ("w1", "w2", "w3", "w4") if m.group(k)), "")
            word = " ".join(word.split()).strip(" .\"'")
            if word.lower() in _PRONOUN:
                word = self.last or ""
                if not word:
                    return "I haven't spelled anything yet -- say “spell” and the word."
        elif _AGAIN.match(t):
            if not self.last:
                return "I haven't spelled anything yet -- say “spell” and the word."
            word = self.last
        else:
            return None
        if not word:
            return None
        self.last = word
        return f"{word}: {letters(word)}."
