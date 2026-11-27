"""
src/daemon/relations.py

The people graph ASH keeps (who is who to you, and who is who to each other), read back by rule. 27 Nov these went to the model (63-87 s each), which
made things up ("I don't know much about personal relationships"):

    "what relationships do you know?"                 the list: who each person is to you, and the ties between them
    "draw me a tree of relationships you know"        the same as an indented tree, you at the root
    "second degree relationships related to Khalid"   who is connected to the people connected to X
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please)[,\s]+)*(?:(?:can|could)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please))*\s*[?.!]*\s*$"
_LIST = re.compile(_A + r"(?:what|which)\s+(?:relationships?|relations|people|family|connections)\s+(?:do\s+you\s+(?:know|have|see)|are\s+(?:there|known))"
                   r"(?:\s+(?:about|of))?|(?:list|show|tell\s+me)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:relationships?|relations|people)\s+(?:you\s+know|you\s+have)?" + _Z, re.I)
_TREE = re.compile(_A + r"(?:draw|show|make|sketch|give)\s+(?:me\s+)?(?:a\s+|the\s+)?(?:tree|graph|diagram|map|web)\s+of\s+(?:the\s+)?(?:relationships?|relations|people|"
                   r"family|connections)(?:\s+you\s+(?:know|have|see))?" + _Z, re.I)
_SECOND = re.compile(_A + r"(?:what|which|who)\s+(?:are\s+)?(?:the\s+)?(?:second[\s-]degree|2nd[\s-]degree|indirect|friends?\s+of\s+friends?)\s*(?:relationships?|relations|"
                     r"connections|people)?\s*(?:you\s+can\s+see\s+)?(?:related\s+to|of|for|around|connected\s+to|to)\s+(?P<who>[\w.' -]{2,30}?)" + _Z, re.I)


def parse(text: str) -> Optional[Tuple[str, str]]:
    t = (text or "").strip()
    if not t or len(t) > 120:
        return None
    m = _SECOND.match(t)
    if m:
        return "second", m.group("who").strip()
    if _TREE.match(t):
        return "tree", ""
    if _LIST.match(t):
        return "list", ""
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _edges(kb: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]]]:
    """(people, ties) -- ties[a][b] = label, both ways."""
    people = dict(getattr(kb, "people", {}) or {})
    ties: Dict[str, Dict[str, str]] = {}
    for a, p in people.items():
        for b, label in ((p or {}).get("between") or {}).items():
            if label:
                ties.setdefault(a, {})[b] = str(label)
                ties.setdefault(b, {})[a] = str(label)
    return people, ties


def _name(people: Dict[str, Dict[str, Any]], key: str) -> str:
    return str((people.get(key) or {}).get("name") or key.title())


def answer(kb: Any, user: str, kind: str, who: str = "") -> str:
    people, ties = _edges(kb)
    me = user or "you"
    if not people:
        return "I don't know anyone yet -- tell me, like “Omar is my cousin”, and I'll keep it."
    direct = sorted(((k, p.get("relation") or p.get("desc") or "") for k, p in people.items()), key=lambda x: x[0])
    if kind == "list":
        parts = [f"{_name(people, k)} ({r})" if r else _name(people, k) for k, r in direct[:10]]
        s = "You: " + ", ".join(parts) + "."
        seen, tie_bits = set(), []
        for a, d in ties.items():
            for b, label in d.items():
                if (b, a) in seen:
                    continue
                seen.add((a, b))
                tie_bits.append(f"{_name(people, a)} and {_name(people, b)} are {label}")
        if tie_bits:
            s += " Between them: " + "; ".join(tie_bits[:5]) + "."
        more = len(direct) - 10
        return s + (f" ({more} more.)" if more > 0 else "")
    if kind == "tree":
        lines = [me]
        for k, r in direct[:10]:
            lines.append(f"  - {_name(people, k)}" + (f" ({r})" if r else ""))
            for b, label in sorted((ties.get(k) or {}).items())[:4]:
                lines.append(f"      - {_name(people, b)} ({label})")
        return "\n".join(lines)
    q = _norm(who)
    root = q if q in people else next((k for k in people if _norm(people[k].get("name")) == q or q in k.split()), "")
    if not root and q in (_norm(me), "me", "myself", "khalid", "immortal"):
        first = [k for k, _r in direct]
        second: List[str] = []
        for k in first:
            for b, label in sorted((ties.get(k) or {}).items()):
                if b not in first and b != k:
                    second.append(f"{_name(people, k)} → {_name(people, b)} ({label})")
        if not second:
            return "Nobody beyond the people you've told me about -- I don't have ties between them and anyone else yet."
        return f"Two steps from you: " + "; ".join(second[:6]) + "."
    if not root:
        return f"I don't have anyone called {who} -- say who they are and I'll keep it."
    first = sorted((ties.get(root) or {}).items())
    out = []
    for b, label in first:
        for c, label2 in sorted((ties.get(b) or {}).items()):
            if c != root and c not in (ties.get(root) or {}):
                out.append(f"{_name(people, root)} → {_name(people, b)} ({label}) → {_name(people, c)} ({label2})")
    if not out:
        return f"I only know {_name(people, root)}'s direct ties" + (": " + ", ".join(f"{_name(people, b)} ({l})" for b, l in first[:5]) if first else ", and there are none yet") + "."
    return "Two steps from " + _name(people, root) + ": " + "; ".join(out[:6]) + "."
