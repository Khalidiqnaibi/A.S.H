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


def alias_in(text: str):
    """(name, whole) -- a name the person says is theirs, and whether the line is only that statement. None if there is none."""
    t = str(text or "")
    for rx, whole in ((_ALIAS_SAY, True), (_ALIAS_IS_ME, True), (_ALIAS_PAREN, False)):
        m = rx.search(t)
        if m:
            name = " ".join(m.group(1).split())
            if name.lower() not in _NOT_ALIAS and not any(w.lower() in _NOT_ALIAS for w in name.split()):
                return name.title(), whole
    return None
