"""
src/daemon/identity.py

What ASH calls the person. It is a setting (console settings -> "Your name", saved as daemon.json identity.user_name), so anyone who
runs ASH can set their own; until they do, ASH uses the developer's DEV_NAME if one is set, else a plain "User", which the prompt
reads as "the person you work with". The name used before is kept as an alias: what was said under it is still found.
"""

import os


def default_user() -> str:
    n = (os.getenv("ASH_USER_NAME") or os.getenv("DEV_NAME") or "").strip().strip("\"'")
    return n.title() if n else "User"


class NameMixin:
    user: str
    user_aliases: set

    def set_user_name(self, name: str) -> str:
        """Change what ASH calls the person, everywhere it is held (the brain too). Returns the name now in use."""
        name = " ".join(str(name or "").split())[:40]
        if not name:
            return self.user
        old = self.user
        if old and old.lower() != name.lower() and old.lower() not in ("user", "unknown"):
            self.user_aliases.add(old)
        self.user = name
        b = getattr(self, "brain", None)
        if b is not None:
            b.user = name
            b.user_aliases = set(self.user_aliases)
        return name


# "khalid (me)", "khalid is me", "my real name is Khalid", "I'm also called Khalid": the person tells ASH another name they go by.
# Today "who is khalid" went to Wikipedia because the name ASH calls them is "Immortal" and nothing said Khalid was them.
import re as _re

_NM = r"([A-Za-z][A-Za-z'\-]{1,19}(?:\s+[A-Za-z][A-Za-z'\-]{1,19})?)"
_ALIAS_PAREN = _re.compile(r"([A-Za-z][A-Za-z'\-]{1,19})\s*\(\s*(?:me|myself|that'?s\s+me|it'?s\s+me)\s*\)", _re.I)
_ALIAS_IS_ME = _re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?" + _NM + r"\s+is\s+(?:also\s+)?(?:me|my\s+name|my\s+real\s+name)\s*[.!]*\s*$", _re.I)
_ALIAS_SAY = _re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:and|but|also|so)[,\s]+)?(?:my\s+(?:real|actual|first|full|other)\s+name\s+is|"
                         r"my\s+name\s+is\s+also|i(?:'m|\s+am)\s+also\s+(?:called|known\s+as)|i\s+also\s+go\s+by|you\s+can\s+also\s+call\s+me)\s+"
                         + _NM + r"\s*[.!]*\s*$", _re.I)
_NOT_ALIAS = frozenset("me my you he she it they this that there who what ash user someone somebody the a an i we".split())


# "khalid and immortal are the same person", "khalid just likes to be called immortal", "Immortal is my nickname": two names for one person.
# Today this was told twice, the model answered "the records show Immortal and Khalid are separate entries", and nothing was kept.
_W = r"([A-Za-z][A-Za-z'\-]{1,19})"
_SAME = (
    _re.compile(r"\b" + _W + r"\s+and\s+" + _W + r"\s+(?:are|is)\s+(?:the\s+)?same(?:\s+(?:person|guy|one|human))?\b", _re.I),
    _re.compile(r"\b" + _W + r"\s+(?:just\s+|also\s+)?(?:likes?|wants?|prefers?)\s+to\s+be\s+called\s+" + _W + r"\b", _re.I),
    _re.compile(r"\b" + _W + r"\s+is\s+(?:also\s+)?" + _W + r"(?:'s)?\s+nickname\b", _re.I),
)
_MY_NICK = _re.compile(r"\b(?:my|his)\s+nickname\s+is\s+" + _W + r"\b|\b" + _W + r"\s+is\s+(?:just\s+)?my\s+nickname\b", _re.I)


def same_person_in(text: str):
    """(a, b) when the line says two names are one person; None otherwise."""
    t = str(text or "")
    for rx in _SAME:
        m = rx.search(t)
        if m:
            a, b = m.group(1), m.group(2)
            if a.lower() not in _NOT_ALIAS and b.lower() not in _NOT_ALIAS and a.lower() != b.lower():
                return a.title(), b.title()
    return None


def sounds_like(a: str, b: str) -> bool:
    """Two spellings of one name: "Khaled" / "Khalid", "Mohamed" / "Muhammad" -- the same consonants in the same order, or one letter apart
    in a name of five letters or more. Whisper writes a name the way it sounds; "who is Khaled" went to Wikipedia."""
    a, b = str(a or "").lower().strip(), str(b or "").lower().strip()
    if not a or not b:
        return False
    if a == b:
        return True
    cons = lambda s: _re.sub(r"(.)\1+", r"\1", _re.sub(r"[aeiouyw'\- ]", "", s))   # noqa: E731
    if len(a) >= 4 and len(b) >= 4 and cons(a) == cons(b) and abs(len(a) - len(b)) <= 2:
        return True
    if min(len(a), len(b)) >= 5 and abs(len(a) - len(b)) <= 1:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1] <= 1
    return False


def alias_in(text: str):
    """(name, whole) -- a name the person says is theirs, and whether the line is only that statement. None if there is none."""
    t = str(text or "")
    m = _MY_NICK.search(t)
    if m:
        name = m.group(1) or m.group(2)
        if name and name.lower() not in _NOT_ALIAS:
            return name.title(), True
    for rx, whole in ((_ALIAS_SAY, True), (_ALIAS_IS_ME, True), (_ALIAS_PAREN, False)):
        m = rx.search(t)
        if m:
            name = " ".join(m.group(1).split())
            if name.lower() not in _NOT_ALIAS and not any(w.lower() in _NOT_ALIAS for w in name.split()):
                return name.title(), whole
    return None
