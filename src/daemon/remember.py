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
    r"who\s+do\s+you\s+think\s+i\s+am|tell\s+me\s+(?:who\s+i\s+am|about\s+me))(?:\s*,?\s*(?:ash|please|now|bro|man))*\s*[.!?]*\s*$", re.I)


def wants_identity(text: str) -> bool:
    return bool(_WHO_AM_I.match(text or ""))


# "What about my relationships?" / "who are the people in my life" / "what are my relationships that you remember" / "who is my family"
_RELATIONS = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:but|so|and|ok(?:ay)?|now)\s+)?(?:please\s+)?(?:(?:what|who)\s+(?:about|are|is)|tell\s+me\s+(?:about|who\s+are)|"
    r"do\s+you\s+(?:know|remember)|list)\s+(?:all\s+)?(?:of\s+)?(?:my|the)\s+(?:relationships?|relatives?|family(?:\s+members)?|"
    r"people(?:\s+in\s+my\s+life)?|friends(?:\s+and\s+family)?|loved\s+ones|close\s+ones)(?:\s+(?:that\s+)?you\s+(?:know|remember)(?:\s+of)?)?"
    r"(?:\s*,?\s*(?:ash|please|now))*\s*[.!?]*\s*$|"
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:what|who)\s+(?:are|is)\s+(?:my\s+)?(?:relationships?|relatives?|family)(?:\s+that\s+you\s+(?:know|remember))?\s*[.!?]*\s*$|"
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:who|what)\s+(?:are|is)\s+my\s+(?:relationships?|relatives?)\b.*$", re.I)


def wants_relationships(text: str) -> bool:
    return bool(_RELATIONS.match(text or ""))


def relationships(kb) -> str:
    """The people on record and how they are related -- from what ASH was told, never made up."""
    rows: List[str] = []
    try:
        with kb._lock:
            for p in kb.people.values():
                rel = (p.get("relation") or "").strip()
                name = (p.get("name") or "").strip()
                if name and rel:
                    rows.append(f"{name} is your {rel}")
                elif name:
                    rows.append(f"{name} ({(p.get('desc') or 'someone you mentioned').strip().rstrip('.')})")
    except Exception:
        rows = []
    if not rows:
        return "I don't have anyone written down yet. Tell me -- \"my sister is Linda\" -- and I'll keep it."
    return ". ".join(rows[:8]) + "."


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


# "My brother is not Omar, where did you get that" / "Omar isn't my brother" / "that's wrong, Linda is my cousin": a correction of what is on record.
# 25 Nov 14:xx it went to the model, which answered the question about where, and Omar stayed.
_WRONG = r"(?:(?:no|nope|wrong|incorrect|actually|that'?s\s+(?:wrong|not\s+(?:true|right)|incorrect)),?\s*)?"
_TAIL = r"(?:\s*[,.;]\s*.{0,60})?\s*[.!?]*\s*$"
_LEAD2 = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?" + _WRONG
_NOT_A = re.compile(_LEAD2 + r"(?P<n>" + _NAME + r")\s+(?:is\s+not|isn'?t|was\s+not|wasn'?t)\s+(?:my|your)\s+" + _ADJ + r"(?P<r>" + _ROLES + r")\b" + _TAIL, re.I)
_NOT_B = re.compile(_LEAD2 + r"my\s+" + _ADJ + r"(?P<r>" + _ROLES + r")\s+(?:is\s+not|isn'?t|was\s+not|wasn'?t)\s+(?P<n>" + _NAME + r")\b" + _TAIL, re.I)
_FIX = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:that'?s\s+(?:wrong|not\s+(?:true|right)|incorrect)|no|nope|wrong|incorrect|actually)\s*,?\s*(?P<n>" + _NAME +
                  r")\s+is\s+my\s+" + _ADJ + r"(?P<r>" + _ROLES + r")\b\s*[.!]*\s*$", re.I)


def parse_correction(text: str) -> Optional[Dict[str, str]]:
    """{"name", "role", "fix": bool}: the person says `name` is not their `role` (fix=False), or says no, `name` IS their `role` (fix=True)."""
    t = (text or "").strip()
    if not t or len(t) > 140:
        return None
    for rx, fix in ((_NOT_A, False), (_NOT_B, False), (_FIX, True)):
        m = rx.match(t)
        if m and _ok_name(m.group("n")) and ROLE_CANON.get(m.group("r").lower()):
            return {"name": _title(m.group("n")), "role": ROLE_CANON[m.group("r").lower()], "fix": fix}
    return None


def correct(kb, text: str) -> Optional[Tuple[str, Any]]:
    """Take a wrong relation out (or replace it) and say what changed: (the sentence, an undo), or None if this is not a correction."""
    p = parse_correction(text)
    if p is None or kb is None:
        return None
    from .forget import restorer, snapshot
    name, role = p["name"], p["role"]
    snap = snapshot(kb)
    undo = restorer(kb, snap, getattr(kb, "entities", None))
    if p["fix"]:
        old = kb.relation_of(name)
        if old == role:
            return f"I already have {name} as your {role}.", None
        if old:
            kb.unrelate(name)
        kb.store_relation(name, role=role)
        return (f"Fixed -- {name} is your {role}, not your {old}." if old else f"Got it -- {name} is your {role}."), undo
    had = kb.unrelate(name, role)
    if had:
        return f"Okay -- I've taken out {name} as your {role}.", undo
    return f"I don't have {name} as your {role}, so there was nothing to take out.", None


_WHO_IS = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:but|so|and|ok(?:ay)?)\s+)?(?:please\s+)?(?:do\s+you\s+(?:know|remember)\s+)?"
                     r"(?:who\s*(?:is|was|'s|’s)|what\s+do\s+you\s+(?:know|remember)\s+about|tell\s+me\s+(?:about|more\s+about)|"
                     r"(?:do\s+you\s+)?know\s+(?:about|who\s+is))\s+(?P<n>" + _NAME + r")(?:\s*,?\s*ash)?\s*[.!?]*\s*$", re.I)


def who_is(kb, text: str, user: str = "") -> Optional[str]:
    """"Who is Linda?" for someone ASH has on record: what it was told, in its own words. None for anyone else (a famous
    person goes to Wikipedia, as before) -- so a sister is never looked up, and never "I don't actually know Linda"."""
    said = role_is(kb, text)
    if said:
        return said
    m = _WHO_IS.match(text or "")
    if m is None or kb is None:
        return None
    name = m.group("n").strip()
    if not _ok_name(name):
        return None
    try:
        with kb._lock:
            p = dict(kb.people.get(norm(name)) or {})
            if not p and " " not in name.strip():                 # "Linda" for "Linda Iqnaibi", when only one person has it
                hits = [v for k, v in kb.people.items() if k.split()[:1] == [norm(name)]]
                if len(hits) == 1:
                    p = dict(hits[0])
    except Exception:
        return None
    if not p:
        return None
    shown = p.get("name") or _title(name)
    rel = (p.get("relation") or "").strip()
    desc = (p.get("desc") or "").strip().rstrip(".")
    between = p.get("between") or {}
    bits = []
    if rel:
        bits.append(f"{shown} is your {rel}")
    if desc:
        bits.append(desc if bits else f"{shown}: {desc}")
    for other, label in list(between.items())[:2]:
        bits.append(str(label))
    if not bits:
        return None
    return ". ".join(b[:1].upper() + b[1:] for b in bits) + "."


_ROLE_Q = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:but|so|and|ok(?:ay)?)\s+)?(?:please\s+)?(?:(?:what|who)(?:\s+is|\s*['\u2019]?s|\s+was)|"
    r"do\s+you\s+(?:know|remember)|you\s+(?:already\s+)?(?:know|remember)|tell\s+me)\s+my\s+(?P<r>" + _ROLES + r")(?:\s*['\u2019]?s\s+name|"
    r"\s+name|\s+called)?(?:\s+is)?(?:\s+(?:name|called))?\s*(?:\bbro\b|\bman\b|\bmate\b)?\s*[.!?]*\s*$", re.I)


def role_is(kb, text: str) -> Optional[str]:
    """"What's my dad's name?" / "do you know my father's name": from the role on record -- or an honest "I don't have it", never a name
    the language model makes up (it once answered with the user's own first name)."""
    m = _ROLE_Q.match(text or "")
    if m is None or kb is None:
        return None
    role = ROLE_CANON.get(m.group("r").lower(), m.group("r").lower())
    try:
        name = kb.role_name(role)
    except Exception:
        return None
    if name:
        return f"Your {role} is {name}."
    return f"I don't have your {role}'s name written down. Tell me once -- \"my {role} is ...\" -- and I'll keep it."


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
