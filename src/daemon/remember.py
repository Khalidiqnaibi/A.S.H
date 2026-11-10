"""
src/daemon/remember.py

"Remember Lieth is my friend" and "who am I?" -- done by rule, from what ASH
really stores, never handed to the language model.

The model, asked to "remember lieth is khalid's best friend", replied "I'm aware
of this from other contexts" and stored nothing but the raw sentence: no person,
no relation, and a confirmation that was not true. Here the sentence is parsed,
written to the knowledge store (which mirrors it into entity memory), and the
answer says exactly what was written -- or that it could not read the sentence.

    remember Lieth is my friend            -> Lieth: friend of the user
    remember that lieth is khalid's best friend (khalid = the user) -> the same
    remember my brother is Omar            -> Omar: brother of the user
    remember Omar is Sara's cousin         -> a tie between Omar and Sara

"Best", "close", "good" friend is a friend: ASH keeps one kind of friend.

"Who am I?" / "who's the user?" / "what do you know about me?" is answered from
the user's name and the people on record, so it never turns into a line from
the activity log or an introduction of ASH itself.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .wondering import FAMILY, ROLE_CANON, norm, relations_conflict

_NAME = r"[a-z][a-z'\-]{1,19}(?:\s+[a-z][a-z'\-]{1,19})??"
_ADJ = r"(?:(?:best|close|closest|good|great|old|dear|really good|very good|real|true|lifelong|longtime)\s+)*"
_ROLES = "|".join(sorted(ROLE_CANON, key=len, reverse=True))
_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:remember|note|keep in mind|don'?t forget)\s+(?:that\s+)?"

# Lieth is my best friend / Lieth is Khalid's best friend
_IS_ROLE = re.compile(_LEAD + r"(?P<n>" + _NAME + r")\s+(?:is|was|'s)\s+(?:(?P<mine>my)|(?P<o>" + _NAME +
                      r")(?:'s|’s))\s+" + _ADJ + r"(?P<r>" + _ROLES + r")\b\s*[.!]?\s*$", re.I)
# my best friend is Lieth / Khalid's friend is Lieth
_ROLE_IS = re.compile(_LEAD + r"(?:(?P<mine>my)|(?P<o>" + _NAME + r")(?:'s|’s))\s+" + _ADJ + r"(?P<r>" + _ROLES +
                      r")\s+(?:is|was|'s)\s+(?P<n>" + _NAME + r")\s*[.!]?\s*$", re.I)
# Lieth is a best friend of mine / of Khalid's
_OF_MINE = re.compile(_LEAD + r"(?P<n>" + _NAME + r")\s+(?:is|was)\s+an?\s+" + _ADJ + r"(?P<r>" + _ROLES +
                      r")\s+of\s+(?:(?P<mine>mine)|(?P<o>" + _NAME + r")(?:'s)?)\s*[.!]?\s*$", re.I)

_NOT_NAMES = frozenset("""i me you he she it we they this that there who what which my your his her our their the a an
    ash user someone somebody everyone anyone nobody everything something nothing today tomorrow yesterday""".split())

_WHO_AM_I = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:but|so|and|ok(?:ay)?)\s+)?(?:please\s+)?"
    r"(?:who\s+am\s+i|who(?:\s+is|\s*['’]?s)\s+(?:the\s+)?user|who(?:\s+is|\s*['’]?s)\s+(?:this|me)|"
    r"do\s+you\s+(?:know|remember)\s+who\s+i\s+am|what(?:'s|\s+is)\s+my\s+name|what\s+do\s+you\s+know\s+about\s+me|"
    r"who\s+do\s+you\s+think\s+i\s+am|tell\s+me\s+(?:who\s+i\s+am|about\s+me))\s*\??\s*$", re.I)


def wants_identity(text: str) -> bool:
    return bool(_WHO_AM_I.match(text or ""))


def _title(name: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in name.strip().split())


def _ok_name(name: str) -> bool:
    words = [w for w in re.split(r"\s+", (name or "").strip().lower()) if w]
    return bool(words) and len(words) <= 2 and not any(w in _NOT_NAMES or w in ROLE_CANON for w in words)


def parse(text: str, user: str = "") -> Optional[Dict[str, str]]:
    """What a "remember ..." sentence says, or None if it isn't one ASH can read.

    {"name", "role", "of"}: `of` is "" when the tie is to the user."""
    t = (text or "").strip()
    for rx in (_IS_ROLE, _ROLE_IS, _OF_MINE):
        m = rx.match(t)
        if not m:
            continue
        name, role = m.group("n"), ROLE_CANON.get(m.group("r").lower(), "")
        owner = (m.group("o") or "").strip()
        if not role or not _ok_name(name):
            return None
        if owner and not _ok_name(owner):
            return None
        mine = bool(m.group("mine")) or (owner and user and norm(owner) == norm(user))
        if not mine and not owner:
            return None
        if mine and norm(name) == norm(user):
            return None                           # "remember Khalid is my friend" said by Khalid: a slip
        return {"name": _title(name), "role": role, "of": "" if mine else _title(owner)}
    return None


def remember(kb, text: str, user: str = "") -> Optional[Tuple[str, Any]]:
    """Store what `text` says. Returns (the sentence to say, an undo function),
    or None if the sentence is not one of the shapes above."""
    p = parse(text, user)
    if p is None or kb is None:
        return None
    from .forget import restorer, snapshot
    name, role, of = p["name"], p["role"], p["of"]
    snap = snapshot(kb)
    undo = restorer(kb, snap, getattr(kb, "entities", None))
    if of:
        label = f"{name} is {of}'s {role}"
        kb.store_between(name, of, label)
        return f"Got it -- {label}.", undo
    prev = kb.relation_of(name)
    if prev == role:
        return f"I already have {name} as your {role}.", None
    if prev and prev in FAMILY and relations_conflict(prev, role):
        return f"I have {name} as your {prev}, so I haven't changed that to {role}. Tell me if it was wrong.", None
    kb.store_relation(name, role=role)
    return f"Got it -- {name} is your {role}.", undo


def who_am_i(kb, user: str = "") -> str:
    """The user, as ASH knows them: the name, and the people on record."""
    user = (user or "").strip()
    people: List[str] = []
    try:
        with kb._lock:
            for p in kb.people.values():
                rel = p.get("relation") or ""
                if rel:
                    people.append(f"{p.get('name', '')} ({rel})")
            facts = dict(kb.facts)
    except Exception:
        people, facts = [], {}
    bits = []
    if user:
        bits.append(f"You're {user}.")
    else:
        bits.append("I don't have your name yet -- tell me and I'll keep it.")
    if people:
        bits.append("The people I know of in your life: " + ", ".join(people[:6]) + ".")
    for k in ("birthday", "anniversary"):
        if facts.get(k):
            bits.append(f"Your {k} is {facts[k]}.")
    liked = ""
    try:
        from .taste import Taste
        t = Taste(kb, user)
        liked = t.describe(t.user_key, 0, "", you=True) if t.user_key in kb.tastes else ""
    except Exception:
        liked = ""
    if liked:
        bits.append(liked)
    if not people and not liked and user:
        bits.append("I don't know much else about you yet.")
    return " ".join(bits)
