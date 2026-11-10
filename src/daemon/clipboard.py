"""
src/daemon/clipboard.py

The clipboard, only when you ask. ASH never watches it.

    what's on my clipboard / read my clipboard / what did I copy
                          reads the first part aloud
    summarize my clipboard / explain what I copied / translate my clipboard /
    proofread the clipboard
                          the text goes to the model with your question
                          (this one copy, then it is forgotten)
    copy that / copy your answer / put that on my clipboard
                          what ASH last said goes onto the clipboard
    clear my clipboard    empties it

Off unless switched on (`clipboard.enabled` in daemon.json): the clipboard
often holds passwords. Text that looks like a secret (a long unbroken token,
"password:" ...) is never read aloud or handed to the model; ASH says so.
Reading and writing are injectable, so tests never touch the real one.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Callable, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|can you\s+|could you\s+)?"
_Z = r"\s*[.!?]*\s*$"
_WHAT = r"(?:my\s+|the\s+)?clip\s?board(?:\s+(?:say|contain|have|hold))?"
_READ = re.compile(_A + r"(?:what(?:'s|\s+is)?\s+(?:on|in)\s+" + _WHAT + r"|read(?:\s+out)?\s+(?:me\s+)?" + _WHAT +
                   r"|what\s+(?:did|have)\s+i\s+(?:just\s+)?cop(?:y|ied)|what\s+does\s+" + _WHAT + r")" + _Z, re.I)
_ASK = re.compile(_A + r"(?P<verb>summari[sz]e|explain|translate|proofread|fix|shorten|rewrite|simplify)\s+(?:the\s+text\s+)?"
                  r"(?:" + _WHAT + r"|what\s+i\s+(?:just\s+)?copied|the\s+(?:text\s+)?i\s+(?:just\s+)?copied)(?P<rest>.*?)" + _Z, re.I)
_COPY = re.compile(_A + r"(?:copy\s+(?:that|this|your\s+(?:last\s+)?(?:answer|reply)|it)(?:\s+(?:to|onto|into)\s+(?:my\s+|the\s+)?clip\s?board)?"
                   r"|put\s+(?:that|your\s+(?:last\s+)?(?:answer|reply))\s+(?:on|onto|in|into)\s+(?:my\s+|the\s+)?clip\s?board)" + _Z, re.I)
_CLEAR = re.compile(_A + r"(?:clear|empty|wipe)\s+" + _WHAT + _Z, re.I)
_SECRET = re.compile(r"(?:pass(?:word|wd|phrase)|secret|api[_ -]?key|token|private key|bearer\s)\s*[:=]|"
                     r"^[A-Za-z0-9+/_\-=]{28,}$|-----BEGIN [A-Z ]*PRIVATE KEY|\b(?:sk|pk|ghp|xox[bp])[-_][A-Za-z0-9_\-]{16,}", re.I | re.M)

MAX_READ_CHARS = 220
MAX_MODEL_CHARS = 3000


def system_read() -> str:
    try:
        import pyperclip
        return pyperclip.paste() or ""
    except Exception:
        pass
    if os.name == "nt":
        try:
            return subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"], capture_output=True,
                                  text=True, timeout=5, creationflags=0x08000000).stdout.rstrip("\r\n")
        except Exception:
            return ""
    return ""


def system_write(text: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    if os.name == "nt":
        try:
            subprocess.run(["clip"], input=text.encode("utf-16-le"), timeout=5, creationflags=0x08000000)
            return True
        except Exception:
            return False
    return False


def looks_secret(text: str) -> bool:
    return bool(_SECRET.search(text or ""))


def parse(text: str) -> Optional[Tuple[str, str]]:
    """("read", "") | ("ask", verb-and-rest) | ("copy", "") | ("clear", "") | None."""
    t = (text or "").strip()
    if not t or len(t) > 120:
        return None
    if _READ.match(t):
        return "read", ""
    m = _ASK.match(t)
    if m:
        rest = (m.group("rest") or "").strip(" ,")
        return "ask", (m.group("verb").lower() + (" " + rest if rest else "")).strip()
    if _COPY.match(t):
        return "copy", ""
    if _CLEAR.match(t):
        return "clear", ""
    return None


class Clipboard:
    def __init__(self, read: Callable[[], str] = system_read, write: Callable[[str], bool] = system_write):
        self.read, self.write = read, write

    def say_contents(self) -> str:
        text = (self.read() or "").strip()
        if not text:
            return "Your clipboard is empty."
        if looks_secret(text):
            return "There's something on your clipboard that looks like a password or key, so I won't read it out."
        one = re.sub(r"\s+", " ", text)
        if len(one) <= MAX_READ_CHARS:
            return f"Your clipboard says: {one}"
        return f"Your clipboard starts: {one[:MAX_READ_CHARS].rsplit(' ', 1)[0]}... ({len(text)} characters in all.)"

    def for_model(self) -> Tuple[str, str]:
        """(text, problem). Problem is a sentence to say instead when the text can't be used."""
        text = (self.read() or "").strip()
        if not text:
            return "", "Your clipboard is empty."
        if looks_secret(text):
            return "", "That looks like a password or key, so I won't pass it to anything."
        return text[:MAX_MODEL_CHARS], ""

    def copy(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "I haven't said anything yet to copy."
        return "Copied." if self.write(text) else "I couldn't reach the clipboard."

    def clear(self) -> str:
        return "Cleared the clipboard." if self.write("") else "I couldn't reach the clipboard."
