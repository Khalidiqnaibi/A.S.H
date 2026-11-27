"""
src/daemon/selfcare.py

ASH looking after its own memory of you, and of your days -- all by rule, on this computer:

    merge_names     "merge Immortal into Khalid": the name ASH uses becomes the new one and the old stays as an alias, the entities named for the old one are
                    renamed, and a dated backup of the entity file is made first (so it can be put back)
    memory_search   "search your memory for Omar": the notes, terms, people, entities and what you said -- with where each came from
    plan            "plan my day" / "what should I do first?": the calendar, targets due, reminders, and where yesterday left off
    reflect         "let's reflect on my day" / "what went well today?": from what really happened (time, follow-ups that went well, streaks, slow answers)
    coach           "any tips?": the hour of your longest unbroken runs, the app you lose the most to, from the last week's window record
    last_mention    "when did I last mention Omar?": from the journal of what you said
"""

from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|for\s+me))*\s*[?.!]*\s*$"
DAY = 86400.0


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower()).strip()


# ----------------------------------------------------------------------------------------------------------------------------- names
_MERGE = re.compile(_A + r"(?:merge|combine|fold)\s+(?P<old>[A-Za-z][\w' -]{1,25}?)\s+(?:into|with|and)\s+(?P<new>[A-Za-z][\w' -]{1,25}?)(?:\s+(?:as\s+)?(?:my\s+name|me|one\s+person))?" + _Z, re.I)


def parse_merge(text: str) -> Optional[Tuple[str, str]]:
    m = _MERGE.match((text or "").strip())
    if not m:
        return None
    a, b = m.group("old").strip(), m.group("new").strip()
    return (a, b) if a.lower() != b.lower() else None


def merge_names(entity_manager: Any, old: str, new: str, backup_dir: str, now: Optional[float] = None) -> Tuple[int, str]:
    """(entities renamed, the backup file). The backup is made before anything changes; the rename is in memory and saved through the store."""
    now = time.time() if now is None else now
    store = getattr(entity_manager, "store", None)
    backup = ""
    if store is None:
        return 0, ""
    try:
        src = store.path
        if src and os.path.exists(src):
            os.makedirs(backup_dir, exist_ok=True)
            backup = os.path.join(backup_dir, time.strftime("entities-%Y%m%d-%H%M%S.json", time.localtime(now)))
            shutil.copy2(src, backup)
    except Exception:
        backup = ""
    n = 0
    for e in list(store.all()):
        attrs = getattr(e, "attributes", {}) or {}
        ids = getattr(e, "primary_identifiers", {}) or {}
        hit = False
        if str(attrs.get("canonical_name", "")).lower() == old.lower():
            attrs["canonical_name"] = new
            hit = True
        if str(ids.get("name", "")).lower() == old.lower():
            ids["name"] = new.lower()
            hit = True
        if hit:
            aliases = list(attrs.get("aliases") or [])
            if old not in aliases:
                aliases.append(old)
            attrs["aliases"] = aliases
            store.update(e)
            n += 1
    return n, backup


# ----------------------------------------------------------------------------------------------------------------------------- memory search
_SEARCH = re.compile(_A + r"(?:search|look\s+through|look\s+in|check)\s+(?:your\s+|my\s+)?memory\s+(?:for|about)\s+(?P<q>[\w .'&-]{2,40}?)" + _Z, re.I)


def parse_search(text: str) -> Optional[str]:
    m = _SEARCH.match((text or "").strip())
    return m.group("q").strip() if m else None


def memory_search(rt: Any, q: str) -> str:
    qn = _norm(q)
    if not qn:
        return "Search for what?"
    hits: List[str] = []
    kb = getattr(getattr(rt, "wonder", None), "kb", None)
    try:
        for k, p in (getattr(kb, "people", {}) or {}).items():
            if qn in _norm(p.get("name")) or qn in _norm(p.get("relation")):
                hits.append(f"a person: {p.get('name')}" + (f", your {p.get('relation')}" if p.get("relation") else ""))
        for k, t in (getattr(kb, "terms", {}) or {}).items():
            if qn in _norm(t.get("term")) or qn in _norm(t.get("definition")):
                hits.append(f"a term: {t.get('term')} is {str(t.get('definition'))[:60]}")
    except Exception:
        pass
    try:
        for n in list(getattr(getattr(rt, "notes", None), "items", None) or []):
            if qn in _norm(n.said):
                hits.append(f"a note: {n.back}")
    except Exception:
        pass
    try:
        store = getattr(getattr(getattr(rt, "ash", None), "entity_memory", None), "store", None)
        for e in (store.all() if store is not None else []):
            if qn in _norm((e.attributes or {}).get("canonical_name")):
                hits.append(f"{e.entity_type}: {(e.attributes or {}).get('canonical_name')}")
    except Exception:
        pass
    try:
        for t in list(getattr(rt, "_told", []) or [])[-60:]:
            if qn in _norm(t[1]):
                hits.append(f"you said: “{t[1][:70]}”")
    except Exception:
        pass
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    if not out:
        return f"Nothing about {q} in what I've kept."
    return f"About {q}: " + "; ".join(out[:6]) + (f" (and {len(out) - 6} more)" if len(out) > 6 else "") + "."


# ----------------------------------------------------------------------------------------------------------------------------- the plan
_PLAN = re.compile(_A + r"(?:plan\s+my\s+day|what(?:'s|\s+is)\s+(?:the\s+plan|my\s+plan(?:\s+for\s+today)?)|what\s+should\s+i\s+(?:do|work\s+on)\s+(?:first|today|now|next)|where\s+should\s+i\s+start)" + _Z, re.I)


def asks_plan(text: str) -> bool:
    return bool(_PLAN.match((text or "").strip()))


def plan(events: List[Tuple[float, str]], targets: List[str], reminders: List[str], yesterday: Optional[Tuple[str, float]], now: float) -> str:
    """First the fixed things (an event today, a target that is due), then where yesterday stopped."""
    bits = []
    soon = sorted(e for e in events if e[0] >= now - 600)
    if soon:
        t, title = soon[0]
        bits.append(f"first, {title} at {datetime.fromtimestamp(t).strftime('%H:%M')}" if len(soon) else "")
    if targets:
        bits.append("then your target: " + targets[0])
    if yesterday and yesterday[1] >= 1800:
        h = yesterday[1] / 3600.0
        bits.append(f"then pick up {yesterday[0]} -- you had {h:.1f} h on it yesterday")
    if reminders:
        bits.append("and remember: " + "; ".join(reminders[:2]))
    if not bits:
        return "Nothing fixed today -- no events, no targets. A good start is whatever you stopped on last."
    s = ", ".join(b for b in bits if b)
    return s[0].upper() + s[1:] + "."


# ----------------------------------------------------------------------------------------------------------------------------- reflection
_REFLECT = re.compile(_A + r"(?:let'?s\s+reflect(?:\s+on\s+(?:my|the)\s+(?:day|week))?|reflect\s+on\s+(?:my|the)\s+(?:day|week)|what\s+went\s+(?:well|right)(?:\s+(?:today|this\s+week))?|"
                      r"what\s+didn'?t\s+go\s+well(?:\s+(?:today|this\s+week))?)" + _Z, re.I)


def asks_reflect(text: str) -> bool:
    return bool(_REFLECT.match((text or "").strip()))


def reflect(worked: List[Tuple[str, float]], good: List[str], streak: Optional[Tuple[str, int]], slow: int, failures: int) -> str:
    well = []
    if worked:
        n, s = worked[0]
        well.append(f"{s / 3600:.1f} hours on {n}")
    if streak and streak[1] >= 2:
        well.append(f"{streak[1]} days in a row on {streak[0]}")
    well += good[:2]
    hard = []
    if slow >= 3:
        hard.append(f"{slow} of my answers were slow")
    if failures:
        hard.append(f"{failures} time{'s' if failures != 1 else ''} I got something wrong or you were frustrated")
    if not well and not hard:
        return "Not enough yet today to reflect on -- ask me again later."
    s = ("Went well: " + "; ".join(well) + ". ") if well else ""
    s += ("Harder: " + "; ".join(hard) + ". ") if hard else "Nothing went badly that I saw. "
    return s + "What would you change tomorrow?"


# ----------------------------------------------------------------------------------------------------------------------------- coaching
_COACH = re.compile(_A + r"(?:any\s+(?:tips|advice|suggestions)|coach\s+me|how\s+can\s+i\s+(?:be\s+more\s+productive|focus\s+better|use\s+my\s+time\s+better)|when\s+do\s+i\s+work\s+best)" + _Z, re.I)


def asks_coach(text: str) -> bool:
    return bool(_COACH.match((text or "").strip()))


def coach(day_rows: List[List[Dict[str, Any]]], friendly: Callable[[str], str] = lambda p: p) -> str:
    """From the window record of the last days: the hour with the most steady work, and the program that takes the most time that is not work."""
    hours: Dict[int, int] = {}
    apps: Dict[str, int] = {}
    for rows in day_rows:
        for r in rows:
            if r.get("source") != "active_window":
                continue
            m = r.get("meta") or {}
            if m.get("blocked") or m.get("private") or not m.get("process"):
                continue
            h = datetime.fromtimestamp(float(r.get("ts") or 0)).hour
            hours[h] = hours.get(h, 0) + 1
            apps[friendly(str(m["process"]))] = apps.get(friendly(str(m["process"])), 0) + 1
    if not hours:
        return "I don't have enough of your week yet to say."
    best = max(hours.items(), key=lambda kv: kv[1])[0]
    top = max(apps.items(), key=lambda kv: kv[1])[0]
    return (f"You're most active around {best:02d}:00 -- keep that hour for the hardest thing. Most of your window time goes to {top}; "
            "if that's not the work, a timer for it would show how much.")


# ----------------------------------------------------------------------------------------------------------------------------- last mention
_LAST = re.compile(_A + r"when\s+(?:did\s+i\s+(?:last\s+)?(?:mention|talk\s+about|say\s+anything\s+about)|was\s+the\s+last\s+time\s+i\s+(?:mentioned|talked\s+about))\s+(?P<q>[A-Za-z][\w .'-]{1,30}?)" + _Z, re.I)


def parse_last(text: str) -> Optional[str]:
    m = _LAST.match((text or "").strip())
    return m.group("q").strip() if m else None


def last_mention(day_rows: List[Tuple[str, List[Dict[str, Any]]]], name: str, user_sources=("microphone", "console", "mic")) -> str:
    """The newest line of what you said that holds `name`, across the days given (newest first)."""
    nn = _norm(name)
    for day, rows in day_rows:
        best = None
        for r in rows:
            if str(r.get("source", "")).lower().removeprefix("sim:") not in user_sources:
                continue
            if nn and nn in _norm(r.get("text")):
                if best is None or float(r.get("ts") or 0) >= float(best.get("ts") or 0):
                    best = r
        if best is not None:
            when = datetime.fromtimestamp(float(best["ts"]))
            return f"You last mentioned {name} on {when.strftime('%A at %H:%M')}: “{str(best['text'])[:90]}”"
    return f"I don't have you mentioning {name} in the days I still have."
