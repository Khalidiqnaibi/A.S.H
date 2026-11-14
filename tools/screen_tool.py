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

# Past this, a reading is history and is described as such, not recited.
STALE_TEXT_S = 600


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


# What is not worth reading out: log lines ("2026-11-09 14:43:43,078 INFO Intent candidate..."), tracebacks, and lines that
# are mostly symbols. Read back verbatim they made ASH sound like a terminal.
_LOGLINE = re.compile(r"^\s*(?:\d{4}-\d\d-\d\d[ T]\d\d:\d\d|\[\d\d:\d\d|\d\d:\d\d:\d\d)|"
                      r"\b(?:INFO|DEBUG|WARNING|ERROR|CRITICAL|Traceback|Exception)\b|^\s*File \"", re.I)


def _readable(piece: str) -> bool:
    if _LOGLINE.search(piece):
        return False
    letters = sum(c.isalpha() for c in piece)
    return len(piece) > 0 and letters / len(piece) >= 0.6


def _spoken_text(text: str) -> str:
    """The gist as a spoken clause: the line breaks the OCR used become full
    stops, and it is cut at a word, not in the middle of one."""
    # "def f(x): / return y" must not become "def f(x):. return y".
    pieces = [p.strip().rstrip(".:;,") for p in re.split(r"\s*/\s*", text or "") if p.strip()]
    pieces = [p for p in pieces if _readable(p)]
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
    age = int(view.get("age_s", 0))
    stale = not view.get("fresh", True)
    if stale:
        ago = f"{age} seconds" if age < 90 else f"{age // 60} minutes"
        # Asked from my own window, which I do not look at: say so, rather
        # than present an old look at something else as the answer.
        lead = ("You're talking to me in my own window, so I can't look at what you're on. "
                if view.get("self_focus") else "")
        parts.append(lead + f"The last thing I looked at, about {ago} ago, was"
                     + (f" {where}." if where else " another window."))
    else:
        parts.append(f"You're in {where}." if where else "I'm looking at your screen.")

    scene = _SCENE.get(view.get("scene", ""), "")
    if scene and not (stale and age > STALE_TEXT_S):
        parts.append(scene)

    if stale and age > STALE_TEXT_S:
        # A reading from a long time ago describes a screen that has almost
        # certainly changed; the words from it would be passed off as now.
        parts.append("That was a while back, so I won't guess what's on it now.")
        return " ".join(parts)

    text = _spoken_text(view.get("text", ""))
    if text:
        parts.append(f"I can make out: {text}")
    elif (view.get("text") or "").strip():
        parts.append("It's mostly code and logs -- nothing I'd read out.")
    elif view.get("reader") is False:
        parts.append("I can tell what kind of screen it is, but I don't have anything "
                     "installed that reads the words on it.")
    else:
        parts.append("I couldn't make out any text on it.")
    return " ".join(parts)


def screen_view_tool(query: str) -> Dict[str, Any]:
    if _LOOKER is None:
        return {"ok": False, "declined": True,
                "error": "I'm not running with a screen sensor, so I can't look. "
                         "It needs the daemon with sensors.screenshot enabled."}
    try:
        view = _LOOKER()
    except Exception as e:                      # never take the brain down for a look
        logger.exception("Screen look failed")
        return {"ok": False, "error": f"looking failed: {type(e).__name__}"}
    if not view.get("ok", True):
        return {"ok": False, "declined": bool(view.get("declined")), "error": view.get("error", "I couldn't look.")}
    return {"ok": True, "verbatim": True, "result": describe_view(view),
            "window": view.get("title", ""), "scene": view.get("scene", ""),
            "withheld": view.get("withheld"), "read": bool(view.get("text"))}
