"""
src/brain/phrasing.py

The words around a request that are not the request.

"what's 16 times 23 please" matched the calculator at 0.36 where "what's 15
times 23" matched at 0.85, and the tool itself choked on "please" -- the turn
went to the language model twice over. Names of address ("hey ash"), politeness
("please", "thank you", "could you") and filler are stripped from what the
classifier reads and from what a tool is handed. The sentence itself, as the
user said it, is still what memory and the model see.
"""

from __future__ import annotations

import re

# "hi ash", "hey ash, what time is it": the name is an address, not a topic.
ADDRESS = re.compile(r"^\s*(?:(?:hey|hi|hello|ok|okay|yo)[,\s]+)?ash\b[,.!\s]*|[,\s]+ash[.!?\s]*$", re.I)

_TAIL = re.compile(r"(?:[,\s]+(?:please|pls|thanks|thank\s+you|thx|for\s+me|real\s+quick|quickly|ash))+[.!?\s]*$", re.I)
_HEAD = re.compile(
    r"^\s*(?:(?:please|ok(?:ay)?|so|and|hey|hi|yo)[,\s]+)+|"
    r"^\s*(?:(?:can|could|would|will)\s+you(?:\s+please)?\s+)"
    r"(?=(?:please\s+)?(?:tell|show|give|read|list|open|close|convert|calculate|check|find|search|set|remind|get|look|"
    r"play|turn|what|how|work)\b)", re.I)


def clean(query: str, keep_original: bool = True) -> str:
    """The request without the name, the politeness or the filler. If nothing is
    left, the original (a tool should still get something), or "" for the
    classifier, which reads nothing as no request."""
    q = ADDRESS.sub("", query or "", count=1).strip()
    for _ in range(2):
        before = q
        q = _TAIL.sub("", q).strip()
        q = _HEAD.sub("", q, count=1).strip()
        if q == before:
            break
    return q or ((query or "").strip() if keep_original else "")
