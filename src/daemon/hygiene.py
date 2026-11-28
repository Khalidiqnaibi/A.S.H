"""
src/daemon/hygiene.py

What ASH keeps, kept clean. Found by reading its own memory on 28 Nov: more than half the episodes linked to no entity, "[FREE MOMENT" text and "I'm looking at my own window" saved as
if they were events, "ApplicationFrameHost" as an activity, tools typed as people (PaddleOCR, C++, Whisper), folders and labels as entities (Users, Desktop, Fact, Reply), garbled
names, a speech-recogniser's mishearing of Ollama kept as a name.

Three parts, none of which needs a model or the network:

    keep_episode(summary) / keep_entity(name, label)   ingest filters: what is not worth keeping is never written
    audit(episodes, entities)                          a read-only report of what is wrong, with counts and examples
    clean(episodes_path, entities_path, apply)         a backup first, then junk episodes dropped, tool names retyped, junk entities removed; apply=False only reports

Run from "check my memory" / "clean up my memory" (the second says what it will do and waits for a yes).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------------------------------------------- what is not an event
_JUNK_EPISODE = [
    re.compile(r"\[FREE MOMENT", re.I),
    re.compile(r"\bI'?m looking at my own (?:window|screen)\b", re.I),
    re.compile(r"\bApplicationFrameHost\b", re.I),
    re.compile(r"^\s*(?:\[?(?:system|debug|internal)\]?:)", re.I),
    re.compile(r"\b(?:Traceback \(most recent call last\)|NoneType|KeyError|AttributeError)\b"),
]

TOOLS = {  # names of software, libraries and languages: never people
    "paddleocr", "whisper", "c++", "c#", "ollama", "mistral", "qwen", "llama", "pytorch", "tensorflow", "spacy", "numpy", "opencv", "pygame", "flask", "django", "react", "node.js",
    "python", "javascript", "typescript", "linux", "windows", "android", "docker", "git", "github", "sql", "json", "ocr", "tts", "stt", "langchain", "chromadb", "faiss", "piper",
    "kokoro", "vosk", "sapbert", "minilm", "bert", "gpt", "chatgpt", "gemini", "claude", "firebase", "huggingface", "ffmpeg", "bluetooth", "wifi",
}
# folders, labels and generic words that spaCy hands back as entities
NOT_THINGS = {"users", "user", "desktop", "documents", "downloads", "fact", "reply", "answer", "task", "given", "answering", "note", "notes", "file", "files", "folder", "none", "null",
              "unknown", "text", "image", "app", "window", "screen", "today", "tomorrow", "yesterday", "morning", "evening", "something", "anything", "everything"}
MISHEARD = {"allama": "ollama", "olama": "ollama", "alama": "ollama"}
_MOJIBAKE = re.compile(r"[ÃÂâ][\u0080-¿€‚-›]|�")


def keep_episode(summary: str) -> bool:
    s = (summary or "").strip()
    if len(s) < 3:
        return False
    return not any(rx.search(s) for rx in _JUNK_EPISODE)


def keep_entity(name: str, label: str = "") -> bool:
    n = (name or "").strip().lower().strip(".,;:!?'\"()[]")
    if not n or len(n) < 2 or n in NOT_THINGS or n.isdigit():
        return False
    if _MOJIBAKE.search(name or ""):
        return False
    if re.fullmatch(r"[a-z]:?[\\/].*|.*[\\/].*\.\w{1,5}", n):               # a path
        return False
    return True


def retype(name: str, label: str) -> str:
    """The label a name should carry: software is a 'tool', never a 'person'."""
    n = (name or "").strip().lower()
    if n in TOOLS and (label or "").lower() in ("person", "proper", "gpe", "fac", "norp", "org", "product", "work_of_art", ""):
        return "tool"
    return label


def fix_name(name: str) -> str:
    return MISHEARD.get((name or "").strip().lower(), name)


# ---------------------------------------------------------------------------------------------------------------------------- audit
def audit(episodes: List[Dict[str, Any]], entities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    junk_ep = [e for e in episodes if not keep_episode(e.get("summary", ""))]
    unlinked = [e for e in episodes if e.get("event_type") == "interaction" and not e.get("related_entities")]
    flat = sum(1 for e in episodes if float(e.get("importance", 0.5) or 0.5) in (0.5, 0.65))
    wrong_type, junk_ent, misheard, untyped = [], [], [], []
    for e in entities.values():
        name = (e.get("primary_identifiers") or {}).get("name", "")
        label = e.get("entity_type", "")
        if retype(name, label) != label:
            wrong_type.append(f"{name} ({label})")
        if not keep_entity(name, label):
            junk_ent.append(name)
        if name.lower() in MISHEARD:
            misheard.append(name)
        if label == "proper":
            untyped.append(name)
    return {
        "episodes": len(episodes), "entities": len(entities),
        "junk_episodes": len(junk_ep), "junk_episode_examples": [e.get("summary", "")[:70] for e in junk_ep[:3]],
        "episodes_without_entities": len(unlinked),
        "episodes_with_default_importance": flat,
        "tools_typed_as_things": wrong_type, "junk_entities": junk_ent, "misheard": misheard, "untyped": len(untyped),
    }


def describe(a: Dict[str, Any]) -> str:
    bits = []
    if a["junk_episodes"]:
        bits.append(f"{a['junk_episodes']} of {a['episodes']} episodes are not events (system text, my own window)")
    if a["tools_typed_as_things"]:
        bits.append(f"{len(a['tools_typed_as_things'])} tools filed under the wrong type ({', '.join(a['tools_typed_as_things'][:3])})")
    if a["junk_entities"]:
        bits.append(f"{len(a['junk_entities'])} entities that are only labels or folders ({', '.join(a['junk_entities'][:3])})")
    if a["misheard"]:
        bits.append(f"a misheard name ({', '.join(a['misheard'][:2])})")
    if a["episodes_without_entities"] >= 10 and a["episodes_without_entities"] > a["episodes"] * 0.4:
        bits.append(f"{a['episodes_without_entities']} episodes linked to nobody and nothing")
    if a["untyped"]:
        bits.append(f"{a['untyped']} names with no type")
    return ("Looking at my memory: " + "; ".join(bits) + ".") if bits else "My memory looks clean."


# ---------------------------------------------------------------------------------------------------------------------------- clean
def _load_eps(path: str) -> List[Dict[str, Any]]:
    out = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    try:
                        out.append(json.loads(ln))
                    except Exception:
                        pass
    return out


def backup(*paths: str, stamp: Optional[str] = None) -> List[str]:
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    made = []
    for p in paths:
        if os.path.exists(p):
            dst = f"{p}.bak-{stamp}"
            shutil.copy2(p, dst)
            made.append(dst)
    return made


def clean(episodes_path: str, entities_path: str, apply: bool = False) -> Dict[str, Any]:
    """Report (apply=False) or do (apply=True, after a backup): drop junk episodes and junk entities, retype tools, fix misheard names. Links to a removed entity are dropped from
    the episodes that held them -- the episodes stay."""
    eps = _load_eps(episodes_path)
    ents: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(entities_path):
        with open(entities_path, "r", encoding="utf-8") as fh:
            ents = json.load(fh)
    before = audit(eps, ents)
    plan = {"drop_episodes": before["junk_episodes"], "drop_entities": len(before["junk_entities"]), "retype": len(before["tools_typed_as_things"]), "rename": len(before["misheard"])}
    if not apply:
        return {"applied": False, "plan": plan, "audit": before}
    backups = backup(episodes_path, entities_path)
    gone = set()
    for eid, e in list(ents.items()):
        pid = e.setdefault("primary_identifiers", {})
        name = pid.get("name", "")
        if not keep_entity(name, e.get("entity_type", "")):
            gone.add(eid)
            del ents[eid]
            continue
        fixed = fix_name(name)
        if fixed != name:
            pid["name"] = fixed
            e.setdefault("attributes", {})["canonical_name"] = fixed.capitalize()
        t = retype(fixed, e.get("entity_type", ""))
        if t != e.get("entity_type"):
            e["entity_type"] = t
    kept = []
    for ep in eps:
        if not keep_episode(ep.get("summary", "")):
            continue
        if gone and ep.get("related_entities"):
            ep["related_entities"] = [r for r in ep["related_entities"] if r not in gone]
        kept.append(ep)
    tmp = episodes_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for ep in kept:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")
    os.replace(tmp, episodes_path)
    tmp = entities_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ents, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, entities_path)
    return {"applied": True, "plan": plan, "audit": before, "backups": backups, "after": audit(kept, ents)}


_ASK = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:can|could)\s+you\s+)?(?:please\s+)?(?P<v>check|audit|clean(?:\s*up)?|tidy(?:\s+up)?|fix)\s+(?:up\s+)?(?:your|my)?\s*(?:own\s+)?(?:memory|memories)(?:\s+for\s+(?:junk|mistakes|errors))?\W*$", re.I)


_IS_CLEAN = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:is\s+(?:your|my)\s+memory\s+(?:clean|ok(?:ay)?|tidy|messy|in\s+good\s+shape)|is\s+there\s+(?:any\s+)?(?:junk|rubbish|garbage|mess|mistakes?|errors?)\s+in\s+(?:your|my)\s+memory|how\s+(?:clean|tidy)\s+is\s+(?:your|my)\s+memory)\W*$", re.I)


def asks(text: str) -> Optional[str]:
    """'check' for the read-only report, 'clean' for the one that changes things (asked about first)."""
    if _IS_CLEAN.match((text or "").strip()):
        return "check"
    m = _ASK.match((text or "").strip())
    if not m:
        return None
    return "check" if m.group("v").lower() in ("check", "audit") else "clean"


# ---------------------------------------------------------------------------------------------------------------------------- on the running stores
def _ep_dict(ep) -> Dict[str, Any]:
    return {"summary": getattr(ep, "summary", ""), "event_type": getattr(ep, "event_type", ""), "related_entities": list(getattr(ep, "related_entities", None) or []),
            "importance": getattr(ep, "importance", 0.5)}


def _ent_dict(e) -> Dict[str, Any]:
    return {"entity_type": getattr(e, "entity_type", ""), "primary_identifiers": dict(getattr(e, "primary_identifiers", None) or {})}


def audit_live(episodic, entity) -> Dict[str, Any]:
    eps = [_ep_dict(e) for e in episodic.store.all()] if episodic is not None else []
    ents = {k: _ent_dict(v) for k, v in entity.store.entities.items()} if entity is not None else {}
    return audit(eps, ents)


def clean_live(episodic, entity) -> Dict[str, Any]:
    """The same clean, on the stores the daemon holds (so nothing it has in memory overwrites it): files backed up first, then the junk dropped and both files rewritten."""
    out = {"backups": [], "dropped_episodes": 0, "dropped_entities": 0, "retyped": 0, "renamed": 0}
    paths = [p for p in (getattr(getattr(episodic, "store", None), "path", None), getattr(getattr(entity, "store", None), "path", None)) if p]
    out["backups"] = backup(*paths)
    gone = set()
    if entity is not None:
        for eid, e in list(entity.store.entities.items()):
            pid = getattr(e, "primary_identifiers", None) or {}
            name = pid.get("name", "")
            if not keep_entity(name, getattr(e, "entity_type", "")):
                gone.add(eid)
                del entity.store.entities[eid]
                out["dropped_entities"] += 1
                continue
            fixed = fix_name(name)
            if fixed != name:
                pid["name"] = fixed
                (getattr(e, "attributes", None) or {})["canonical_name"] = fixed.capitalize()
                out["renamed"] += 1
            t = retype(fixed, getattr(e, "entity_type", ""))
            if t != e.entity_type:
                e.entity_type = t
                out["retyped"] += 1
        entity.store._save()
    if episodic is not None:
        with episodic.store.lock:
            for eid, ep in list(episodic.store.episodes.items()):
                if not keep_episode(getattr(ep, "summary", "")):
                    del episodic.store.episodes[eid]
                    out["dropped_episodes"] += 1
                elif gone and getattr(ep, "related_entities", None):
                    ep.related_entities = [r for r in ep.related_entities if r not in gone]
            episodic.store._save_all()
            try:
                episodic.index.rebuild(episodic.store.all())
            except Exception:
                pass
    return out


def say_cleaned(o: Dict[str, Any]) -> str:
    bits = [f"{o['dropped_episodes']} junk episodes dropped", f"{o['dropped_entities']} label-only entities removed", f"{o['retyped']} tools retyped", f"{o['renamed']} names fixed"]
    return "Done: " + ", ".join(bits) + ". A backup of both files is next to them."


# ---------------------------------------------------------------------------------------------------------------------------- the way back
_UNDO = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:can|could)\s+you\s+)?(?:please\s+)?(?:undo|revert|roll\s*back|reverse|restore)\s+(?:the\s+|that\s+|my\s+|your\s+)?(?:memory\b\s*)?(?:clean[\s-]*up|cleanup|clean(?:ing)?|tidy(?:ing)?)?(?:\s+of\s+(?:my|your)\s+memory)?(?:\s+(?:my|your)\s+memory)?\W*$", re.I)


def asks_undo(text: str) -> bool:
    t = (text or "").strip()
    return bool(_UNDO.match(t)) and bool(re.search(r"memory|clean|tidy", t, re.I))


def restore_live(episodic, entity, backups: List[str]) -> Dict[str, int]:
    """Put the backed-up files back where they were and load them again into the stores the daemon holds. Returns the counts it ends with."""
    for b in backups or []:
        orig = re.sub(r"\.bak-[\w-]+$", "", b)
        if os.path.exists(b) and orig != b:
            shutil.copy2(b, orig)
    n_eps = n_ents = 0
    if entity is not None:
        entity.store._load()
        n_ents = len(entity.store.entities)
    if episodic is not None:
        with episodic.store.lock:
            episodic.store.episodes = {}
            episodic.store._load()
            n_eps = len(episodic.store.episodes)
            try:
                episodic.index.rebuild(episodic.store.all())
            except Exception:
                pass
    return {"episodes": n_eps, "entities": n_ents}
