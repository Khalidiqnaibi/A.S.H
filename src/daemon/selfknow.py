"""
src/daemon/selfknow.py

ASH knows what it is running on, and does not claim what is not so.

    facts(rt)                      a dictionary read from the live runtime: the model, whether the face / camera / screen / microphone / voice are on, how much it remembers
    asks(text) -> topic            "what model are you running", "what can you see right now", "can you hear me", "what are you running on" -> answered from facts(), no model
    answer(topic, facts)           one honest sentence
    ungrounded(reply, facts)       a model answer that says ASH sees / hears / is looking at something it cannot right now -> what to say instead (None when the claim is fine)
    model_change(old, new)         "I'm running on X now; last time it was Y." -- said once, when the model ASH thinks with changed since it last ran (a flip was seen on 28 Nov)

Nothing here needs a model or the network.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def facts(rt) -> Dict[str, Any]:
    ash = getattr(rt, "ash", None)
    llm = getattr(ash, "llm", None)
    gate = getattr(rt, "gate", None)
    priv = getattr(rt, "privacy", None)
    looking = getattr(rt, "looking", None)
    cam = False
    try:
        cam = bool(looking is not None and getattr(looking, "camera_on", None) and looking.camera_on())
    except Exception:
        cam = False
    store = getattr(getattr(ash, "episodic_memory", None), "store", None)
    try:
        n_eps = len(store.all()) if store is not None else 0
    except Exception:
        n_eps = 0
    ents = getattr(getattr(getattr(ash, "entity_memory", None), "store", None), "entities", None) or {}
    return {
        "model": str(getattr(llm, "_ollama_model", "") or ""),
        "light_model": str(getattr(llm, "_light_model", "") or ""),
        "face": getattr(rt, "face", None) is not None,
        "camera": cam,
        "screen": bool(getattr(rt, "desktop", None) is not None),
        "muted": bool(getattr(gate, "muted", False)),
        "paused": bool(getattr(priv, "paused", False)),
        "episodes": n_eps, "entities": len(ents),
    }


_BASE = r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|ok(?:ay)?|and)[,\s]+)?"
def _t(alts: str):
    return re.compile(_BASE + r"(?:" + alts + r")\W*$", re.I)


_TOPICS = [
    ("model", _t(r"(?:what|which)\s+(?:model|llm|brain|ai)\s+(?:are\s+you\s+(?:running|using|on|thinking\s+with)|do\s+you\s+(?:use|run\s+on)|is\s+(?:this|that|answering(?:\s+me)?))|"
                r"(?:tell\s+me\s+)?(?:what|which)\s+(?:model|llm)\s+you\s+(?:run|use)(?:\s+on)?|"
                r"what\s+are\s+you\s+(?:running|thinking)\s+(?:on|with)")),
    ("see", _t(r"(?:what|who)\s+(?:can|are\s+you\s+able\s+to)\s+(?:you\s+)?(?:see|look\s+at)(?:\s+(?:right\s+)?now)?|can\s+you\s+see\s+(?:me|my\s+screen|anything|right\s+now)|are\s+you\s+(?:watching|looking\s+at)\s+(?:me|my\s+screen)")),
    ("hear", _t(r"can\s+you\s+(?:hear|listen\s+to)\s+(?:me|us|anything)(?:\s+(?:right\s+)?now)?|are\s+you\s+listening(?:\s+to\s+(?:me|us))?(?:\s+(?:right\s+)?now)?|is\s+(?:your|the)\s+(?:mic|microphone)\s+(?:on|working|live)")),
    ("state", _t(r"(?:what|how)\s+(?:are\s+you|is\s+your\s+setup)\s+(?:set\s+up|configured)|what(?:'?s|\s+is|\s+do\s+you\s+have)\s+(?:switched\s+on|turned\s+on|on)(?:\s+right\s+now)?")),
]


def asks(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or len(t) > 80:
        return None
    for name, rx in _TOPICS:
        if rx.match(t):
            return name
    return None


def answer(topic: str, f: Dict[str, Any]) -> str:
    if topic == "model":
        if not f["model"]:
            return "I can't tell which model is answering from here."
        light = f" (and {f['light_model']} for quick things)" if f["light_model"] and f["light_model"] != f["model"] else ""
        return f"I'm thinking with {f['model']}{light} -- but most of what I do is rules and memory, not the model."
    if topic == "see":
        if f["paused"]:
            return "No -- privacy is on, so I'm not looking at anything."
        parts = []
        parts.append("the camera is on" if f["camera"] else "the camera is off")
        parts.append("I can read your screen when you ask" if f["screen"] else "I have no screen access here")
        return "Right now " + " and ".join(parts) + "."
    if topic == "hear":
        if f["paused"] or f["muted"]:
            return "No -- the microphone is off." if f["muted"] else "No -- privacy is on."
        return "Yes, I'm listening."
    if topic == "state":
        bits = [f"model {f['model'] or 'unknown'}", "face on" if f["face"] else "no face", "camera on" if f["camera"] else "camera off",
                "microphone off" if f["muted"] else "microphone on", f"{f['episodes']} episodes and {f['entities']} people and things remembered"]
        return "Right now: " + ", ".join(bits) + "."
    return ""


# ---------------------------------------------------------------------------------------------------------------------------- claims
_SEE_CLAIM = re.compile(r"\bI\s+(?:can\s+see|see|am\s+seeing|'m\s+seeing|can\s+tell\s+from\s+(?:the|your)\s+(?:camera|screen))\s+(?:you|your\s+(?:face|screen|desk|room|hands)|that\s+you|the\s+(?:screen|room))\b|"
                        r"\bI'?m\s+(?:looking|watching)\s+(?:at\s+)?(?:you|your\s+(?:screen|face))\b|\bthrough\s+(?:the|your)\s+camera\b", re.I)
_HEAR_CLAIM = re.compile(r"\bI\s+(?:can\s+hear|heard|am\s+hearing|'m\s+hearing)\s+(?:you|your\s+voice|that|the\s+music|someone)\b", re.I)


def ungrounded(reply: str, f: Dict[str, Any]) -> Optional[str]:
    r = reply or ""
    if _SEE_CLAIM.search(r) and not f.get("camera"):
        return "I can't see you right now -- " + answer("see", f)[len("Right now "):] if not f.get("paused") else "I can't see anything -- privacy is on."
    if _HEAR_CLAIM.search(r) and (f.get("muted") or f.get("paused")):
        return "I can't hear anything right now -- the microphone is off."
    return None


def model_change(old: str, new: str) -> str:
    old, new = (old or "").strip(), (new or "").strip()
    if not old or not new or old == new:
        return ""
    return f"I'm thinking with {new} now; last time it was {old}."
