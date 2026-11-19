"""
src/daemon/facets.py

An entity is more than a name. 311 of 316 entities held a name and the last sentence they appeared in -- "a list of words, not knowledge of
anyone". Compiled from what ASH already has (the episodes that mention it, what it was told about the person, what they like), each
entity gets facets: how often it came up, since when and last when, the relationship, what they like, and a few of the things said around
it. Like the concept gist, this is the slow store learning what the fast one keeps seeing; it runs in the nap and never asks anyone anything.
"""

import re
import time
from typing import Any, Dict, Iterable, List, Optional

FACET_KEYS = ("mentions", "first_seen", "last_seen", "relationship", "likes", "heard_in")


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _short(summary: str) -> str:
    s = str(summary or "")
    head, sep, rest = s.partition(":")
    s = rest if sep and len(head) <= 24 else s
    s = re.sub(r"\s*\[[^\]]*\]\s*$", "", s)                 # (not the "[mentions: ...]" the episode carries)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:100]


def compile_facets(store, episodes: Iterable[Any], kb=None, now: Optional[float] = None) -> int:
    """Fill the facets of every entity in `store` (an EntityStore). Returns how many entities changed; saves once if any."""
    ents = getattr(store, "entities", None) or {}
    if not ents:
        return 0
    seen: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        t = float(getattr(ep, "timestamp", 0.0) or 0.0)
        for eid in set(getattr(ep, "related_entities", None) or []):
            if eid not in ents:
                continue
            d = seen.setdefault(eid, {"n": 0, "first": t or 1e18, "last": t, "ctx": []})
            d["n"] += 1
            if t:
                d["first"] = min(d["first"], t)
                d["last"] = max(d["last"], t)
            c = _short(getattr(ep, "summary", ""))
            if c and c not in d["ctx"] and len(d["ctx"]) < 3:
                d["ctx"].append(c)
    people: Dict[str, Dict[str, Any]] = {}
    likes: Dict[str, List[str]] = {}
    if kb is not None:
        try:
            with kb._lock:
                for k, p in kb.people.items():
                    people[_canon(p.get("name") or k)] = dict(p)
                    first = _canon(str(p.get("name") or k).split()[:1][0]) if str(p.get("name") or k).split() else ""
                    if first and first not in people:
                        people[first] = dict(p)
                for k, tv in kb.tastes.items():
                    items = list((tv.get("items") or {}).keys())[:5]
                    if items:
                        likes[_canon(tv.get("name") or k)] = items
        except Exception:
            people, likes = {}, {}
    changed = 0
    for eid, ent in ents.items():
        at = ent.attributes
        name = at.get("canonical_name") or (ent.primary_identifiers or {}).get("name") or ""
        new: Dict[str, Any] = {}
        d = seen.get(eid)
        if d:
            new["mentions"] = d["n"]
            new["first_seen"] = round(d["first"], 0) if d["first"] < 1e17 else None
            new["last_seen"] = round(d["last"], 0)
            new["heard_in"] = list(d["ctx"])
        p = people.get(_canon(name))
        if p and (p.get("relation") or "").strip():
            new["relationship"] = str(p["relation"]).strip()
        lk = likes.get(_canon(name))
        if lk:
            new["likes"] = lk
        new = {k: v for k, v in new.items() if v not in (None, [], "")}
        if any(at.get(k) != v for k, v in new.items()):
            at.update(new)
            changed += 1
    if changed:
        try:
            store._save()
        except Exception:
            pass
    return changed
