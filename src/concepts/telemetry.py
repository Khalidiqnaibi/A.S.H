"""
src/concepts/telemetry.py

What counts as a reading of the machine or the room: "memory at 93%", battery, "screen changed", a window switch, motion. The concept
graph perceives them (so they can be explained) but never grows a concept from one: they are the body's state, which homeostasis has,
and 23 of the graph's concepts had been made from them, one of them taking a quarter of all activation.
"""

import re

_TELEMETRY_TEXT = re.compile(
    r"\b(?:memory|cpu|ram|disk|gpu|battery|temperature)\b[^.]{0,24}?\d+(?:\.\d+)?\s*%|^\s*battery\b|\bscreen changed\b|"
    r"\bmotion in camera view\b|^\s*switched to\b.{0,120}?\(was on\b|^\s*(?:idle|away) for \d+|\bunplugged\b", re.I)
_TELEMETRY_SOURCES = frozenset({"system", "resources", "battery", "idle", "screen", "camera", "window"})
# ASH's own remarks come back through the loop as events: perceived (so they can be explained), never a thing to grow a concept
# from -- 40 of the graph's concepts were "You've been on <window> for 15 minutes", ASH's own sentence at different minutes.
_OWN_SOURCES = frozenset({"flow", "ash", "remark", "self"})


def is_telemetry(text: str, source: str = "") -> bool:
    """A reading of the machine or the room (memory at 93%, battery, a window switch, "screen changed", motion): not
    something to grow concepts from. A person's words never are, whatever source carries them."""
    # "Uh-oh.", "Back on some.", "I can sound.", "Mm.": a noise or a mishearing of one, from the room. Each grew a concept, and the
    # concepts became hubs 111 other concepts pointed at. A typed line is always a thing said on purpose.
    if str(source or "").lower() in ("microphone", "voice", "ambient") and len(str(text or "").split()) <= 3:
        return True
    if str(source or "").lower() in ("microphone", "console", "voice", "user", "chat", "wonder"):
        return False
    if str(source or "").lower() in _OWN_SOURCES:
        return True
    t = str(text or "")
    return bool(_TELEMETRY_TEXT.search(t)) or (str(source or "").lower() in _TELEMETRY_SOURCES and len(t.split()) <= 14)
