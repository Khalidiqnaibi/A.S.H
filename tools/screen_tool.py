"""
tools/screen_tool.py

"What do you see?"

ASH has a screenshot sensor and an OCR engine, and for a day, asked what it
could see, it answered "I can't capture your screen because I don't have a
graphical user interface." Nothing in the tool catalog could look, so the
language model produced a reason why it couldn't -- a confident, specific,
false one. This is the tool that can.

The tool itself holds no sensor. The daemon hands it a callable at startup
(`set_looker`), the same way actuators are bound to the registry, so this
module stays importable -- and answers honestly, "I'm not running with a
screen sensor" -- anywhere the daemon is not.

Everything the sensor refuses it reports as a refusal with its reason, in
words, rather than returning an empty look: a blocked window, a chat or a
mailbox, a paused privacy policy, ASH's own window. "I'm not looking at that
one, it's a chat" is an answer. A shrug is not.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("ash.tools.screen")

_LOOKER: Optional[Callable[[], Dict[str, Any]]] = None

# How much of what was read is worth saying out loud. The full reading stays
# in the result for the console; a spoken answer that recites a screenful of
# OCR is worse than one that gives the gist.
MAX_SPOKEN_TEXT = 170


def set_looker(fn: Optional[Callable[[], Dict[str, Any]]]) -> None:
    """Called by the daemon with the live sensor's look function, or None."""
    global _LOOKER
    _LOOKER = fn


def available() -> bool:
    return _LOOKER is not None


_WITHHELD = {
    "blocked": "That window is on my blocked list, so I don't look at it.",
    "private": "That's a chat or a mailbox, so I leave it alone.",
    "paused": "My privacy pause is on, so I'm not looking at anything right now.",
    "off": "Screen capture is switched off.",
    "self": ("I'm looking at my own window, so there's nothing to see. "
             "Switch to what you want me to look at and ask again."),
}

_SCENE = {
    "text": "It's mostly text.",
    "media": "It looks like video or images.",
    "blank": "The screen looks idle.",
    "mixed": "",
}


def _spoken_text(text: str) -> str:
    """The gist as a spoken clause: the line breaks the OCR used become full
    stops, and it is cut at a word, not in the middle of one."""
    # "def f(x): / return y" must not become "def f(x):. return y".
    pieces = [p.strip().rstrip(".:;,") for p in re.split(r"\s*/\s*", text or "") if p.strip()]
    flat = ". ".join(pieces)
    if len(flat) <= MAX_SPOKEN_TEXT:
        return flat
    cut = flat[:MAX_SPOKEN_TEXT].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "..."


def describe_view(view: Dict[str, Any]) -> str:
    """One short spoken paragraph from what the sensor found."""
    from src.daemon.social import friendly_app

    why = view.get("withheld")
    if why:
        return _WITHHELD.get(why, "I can't look at that right now.")

    app = friendly_app(view.get("process", "")) or (view.get("title") or "").strip()
    subject, project = view.get("subject", ""), view.get("project", "")
    if subject and project:
        where = f"{app}, on {subject} in {project}" if app else f"{subject} in {project}"
    elif subject:
        where = f"{app}, on {subject}" if app else subject
    else:
        where = app

    parts = []
    if not view.get("fresh", True):
        age = int(view.get("age_s", 0))
        ago = f"{age} seconds" if age < 90 else f"{age // 60} minutes"
        parts.append(f"The last thing I looked at, about {ago} ago, was"
                     + (f" {where}." if where else " another window."))
    else:
        parts.append(f"You're in {where}." if where else "I'm looking at your screen.")

    scene = _SCENE.get(view.get("scene", ""), "")
    if scene:
        parts.append(scene)

    text = _spoken_text(view.get("text", ""))
    if text:
        parts.append(f"The part I can read says: {text}")
    elif view.get("reader") is False:
        parts.append("I can tell what kind of screen it is, but I don't have anything "
                     "installed that reads the words on it.")
    else:
        parts.append("I couldn't make out any text on it.")
    return " ".join(parts)


def screen_view_tool(query: str) -> Dict[str, Any]:
    if _LOOKER is None:
        return {"ok": False,
                "error": "I'm not running with a screen sensor, so I can't look. "
                         "It needs the daemon with sensors.screenshot enabled."}
    try:
        view = _LOOKER()
    except Exception as e:                      # never take the brain down for a look
        logger.exception("Screen look failed")
        return {"ok": False, "error": f"looking failed: {type(e).__name__}"}
    if not view.get("ok", True):
        return {"ok": False, "error": view.get("error", "I couldn't look.")}
    return {"ok": True, "verbatim": True, "result": describe_view(view),
            "window": view.get("title", ""), "scene": view.get("scene", ""),
            "withheld": view.get("withheld"), "read": bool(view.get("text"))}
