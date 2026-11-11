"""
src/daemon/grammar.py

"Is this a command ASH has a handler for?" -- as evidence that it was said to ASH.

The address scorer (senses/address.py) decides whether a sentence was meant for
ASH from how it is shaped: a name, "you", a question, an opening verb from a short
list. The handlers that act on the world each have their own, much richer idea of
what a command is -- and they will only act on a sentence the scorer rated at
least "probably for ASH". So the two disagreed: "switch to spotify", "work mode",
"dim the lights", "launch notepad", "wind down", "quiet for an hour", "focus for
45 minutes" are exactly what the handlers parse, and the scorer gave them 0, so
the handler never ran and ASH sat silent. 82 of 98 such sentences did nothing.

This asks the handlers. `strength(text, runtime)` is 0.5 if any of their parsers
recognises the sentence as one of theirs (more than "probably", less than
"certainly": a recognised command still might be said to someone else, so it is
never promoted to certain), else 0. Nothing here acts on anything, and every
parser used is a pure function of the text.

Left out on purpose: a lone "stop", "play", "next", "pause" or "yes" -- they are
as likely to be said to a person or a child, and mean something only in a
context the handlers already read (ASH speaking, a game running, a question
just asked).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional

logger = logging.getLogger("ash.daemon.grammar")

GRAMMAR_CONF = 0.5                 # "probably for ASH": above LIKELY_AT, below ADDRESSED_AT

_REPLY_ONLY = frozenset("stop play pause next resume continue skip yes no ok okay louder quieter softer".split())


def _words(t: str) -> int:
    return len((t or "").split())


def _rx_any(rxs, text: str) -> bool:
    return any(rx.match(text) for rx in rxs)


def _checks(runtime: Any = None) -> List[Callable[[str], bool]]:
    """The parsers, built once per call site. Imports are local: a missing module must not stop the scorer."""
    out: List[Callable[[str], bool]] = []

    def add(fn):
        out.append(fn)

    try:
        from . import media
        music = [rx for rx, _k, _s in media._MUSIC]
        add(lambda t, music=music: _words(t) >= 2 and _rx_any(music, t))
        add(lambda t: bool(media._PHOTO.match(t)))
    except Exception:
        logger.debug("media grammar unavailable", exc_info=True)
    try:
        from . import desktop
        rxs = [desktop._OPEN, desktop._SWITCH, desktop._CLOSE, desktop._ROUTINE]
        add(lambda t, rxs=rxs: _rx_any(rxs, t))
    except Exception:
        logger.debug("desktop grammar unavailable", exc_info=True)
    try:
        from . import reminders
        rxs = [reminders._REMIND, reminders._TIMER, reminders._TIMER_N, reminders._ALARM, reminders._SNOOZE,
               reminders._CANCEL, reminders._LIST]
        add(lambda t, rxs=rxs: _rx_any(rxs, t))
    except Exception:
        logger.debug("reminders grammar unavailable", exc_info=True)
    try:
        from . import focus
        add(lambda t: bool((focus._ON.match(t) and (_words(t) >= 2)) or focus._OFF.match(t)))
    except Exception:
        logger.debug("focus grammar unavailable", exc_info=True)
    try:
        from . import smarthome
        rxs = [smarthome._ON_OFF, smarthome._DIM]
        add(lambda t, rxs=rxs: _rx_any(rxs, t))
    except Exception:
        logger.debug("smarthome grammar unavailable", exc_info=True)
    try:
        from . import play
        add(lambda t: bool(play._START.match(t)))
    except Exception:
        logger.debug("play grammar unavailable", exc_info=True)
    try:
        from . import tasks
        rxs = [tasks._TEACH, tasks._LIST, tasks._FORGET, tasks._SAVE]
        add(lambda t, rxs=rxs: _rx_any(rxs, t))
    except Exception:
        logger.debug("tasks grammar unavailable", exc_info=True)
    try:
        from . import voicectl
        add(lambda t: voicectl.parse(t) is not None)
    except Exception:
        logger.debug("voicectl grammar unavailable", exc_info=True)
    try:
        from . import quiet
        add(lambda t: quiet.parse(t, time.time()) is not None)
    except Exception:
        logger.debug("quiet grammar unavailable", exc_info=True)
    try:
        from . import personas
        add(lambda t: personas.parse(t, time.time()) is not None)
    except Exception:
        logger.debug("personas grammar unavailable", exc_info=True)
    try:
        from . import watch, research
        add(lambda t: watch.parse_request(t) is not None or research.parse_request(t) is not None)
    except Exception:
        logger.debug("watch/research grammar unavailable", exc_info=True)
    try:
        from . import clipboard, arrival, browsing, uxreport
        add(lambda t: clipboard.parse(t) is not None or browsing.parse(t) is not None or uxreport.parse(t) is not None)
        rxs = [arrival._WHEN, arrival._WHEN_END, arrival._SET, arrival._FORGET, arrival._LIST, arrival._CANCEL]
        add(lambda t, rxs=rxs: _rx_any(rxs, t))
    except Exception:
        logger.debug("clipboard/arrival grammar unavailable", exc_info=True)
    try:
        from . import voiceid
        add(lambda t: voiceid.is_request(t))
    except Exception:
        logger.debug("voice-id grammar unavailable", exc_info=True)
    try:
        from . import forget, remember
        add(lambda t: forget.wants_undo(t) or forget.wants_forget(t) is not None)
        add(lambda t: remember.parse(t, "") is not None)
    except Exception:
        logger.debug("forget/remember grammar unavailable", exc_info=True)
    sc = getattr(runtime, "scenes", None)
    if sc is not None:
        add(lambda t, sc=sc: (sc.parse(t) or (None,))[0] in ("start", "end"))
    return out


class Grammar:
    """Built once for a runtime; callable as the address scorer's grammar hook."""

    def __init__(self, runtime: Any = None):
        self.checks = _checks(runtime)

    def __call__(self, text: str) -> float:
        t = (text or "").strip()
        if not t or len(t) > 160 or t.lower().strip(" .!?") in _REPLY_ONLY:
            return 0.0
        for fn in self.checks:
            try:
                if fn(t):
                    return GRAMMAR_CONF
            except Exception:
                continue
        return 0.0
