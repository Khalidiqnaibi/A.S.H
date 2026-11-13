"""
src/daemon/availability.py

Can the user talk right now?

A free moment used to mean "a pause after two minutes of steady typing". That is rare, and it is also the wrong thing to
look for. The point of a free moment is the opposite question: when CAN'T the user be spoken to -- so ASH keeps out of
those and is there in every other moment. A person who is alive in a room does not wait for a bell; they notice you are
in the middle of something, and when you are not, they say a word.

CAN'T (never spoken to, whatever the budget)
  away          no input for ten minutes
  in a call     a calling program in front (Zoom, Teams, Webex, Skype...), or a window called a meeting or a call
  gaming        a game or a game launcher in front
  presenting    a slide show or a presentation window
  focusing      "I'm focusing for an hour" is on
  talking       the user is in the middle of a conversation with ASH, or ASH is speaking
  typing        input in the last few seconds

BUSY (only something worth interrupting for)
  working       an editor, a document, a design tool in front, with input in the last minute
  watching      a video in front

FREE
  present, not in any of the above: browsing, reading, between things, hands off the keyboard in anything else. The longer
  they have been free (up to a few minutes), the better the moment.

By rule from the window and the clock; nothing here calls a model, and no title is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .occasion import ACTIVITY_OF, BROWSERS

_CALL_APPS = frozenset("zoom teams ms-teams msteams webex skype lync whereby bluejeans slack-huddle facetime".split())
_CALL_TITLE = re.compile(r"\b(?:meeting|on a call|video call|huddle|zoom meeting|google meet|meet -|teams call|webinar|interview)\b", re.I)
_PRESENT = frozenset("powerpnt keynote slideshow impress".split())
_PRESENT_TITLE = re.compile(r"\b(?:slide show|slideshow|presenting|presenter view)\b", re.I)
_VIDEO_APPS = frozenset("vlc mpc-hc mpc-hc64 mpv potplayer potplayermini64 wmplayer netflix primevideo disneyplus kodi plex".split())
_VIDEO_TITLE = re.compile(r"\b(?:youtube|netflix|prime video|disney\+|twitch|hulu|crunchyroll|vimeo|watch)\b", re.I)
_WORKING = frozenset("coding writing designing working".split())


def _stem(process: str) -> str:
    return re.sub(r"\.(?:exe|app|bin)$", "", (process or "").strip().lower())


@dataclass
class Avail:
    state: str                 # free | working | watching | away | call | gaming | presenting | focusing | talking | typing | unknown
    can_talk: bool             # False: never spoken to now
    score: float               # 0..1: how good a moment this is for something that is not urgent
    why: str = ""

    @property
    def free(self) -> bool:
        return self.state == "free"


def judge(process: str = "", subject: str = "", idle_s: Optional[float] = None, since_free_s: float = 0.0,
          focus: bool = False, conversation: bool = False, speaking: bool = False, away_s: float = 600.0,
          private: bool = False, video_page: bool = False) -> Avail:
    """The state from what is in front and how long the hands have been off."""
    if idle_s is None:
        return Avail("unknown", False, 0.0, "no way to tell when they are at the machine")
    if idle_s >= away_s:
        return Avail("away", False, 0.0, "away")
    if focus:
        return Avail("focusing", False, 0.0, "focus mode is on")
    if speaking or conversation:
        return Avail("talking", False, 0.0, "in a conversation")
    stem = _stem(process)
    title = "" if private else (subject or "")
    if stem in _CALL_APPS or (_CALL_TITLE.search(title) and stem not in BROWSERS):
        return Avail("call", False, 0.0, "in a call")
    if stem in BROWSERS and _CALL_TITLE.search(title) and re.search(r"\b(?:meet|zoom|teams|webex)\b", title, re.I):
        return Avail("call", False, 0.0, "in a call")
    if ACTIVITY_OF.get(stem) == "gaming":
        return Avail("gaming", False, 0.0, "playing a game")
    if _PRESENT_TITLE.search(title):
        return Avail("presenting", False, 0.0, "presenting")
    if idle_s < 5.0:
        return Avail("typing", False, 0.0, "typing right now")
    act = ACTIVITY_OF.get(stem, "")
    if stem in _VIDEO_APPS or video_page or (stem in BROWSERS and _VIDEO_TITLE.search(title)):
        return Avail("watching", True, 0.25, "watching something")
    if act in _WORKING and idle_s < 60.0:
        return Avail("working", True, 0.3, f"{act}")
    # present and not in the middle of anything: the longer the better, up to a few minutes
    warm = min(1.0, max(0.0, since_free_s) / 180.0)
    return Avail("free", True, 0.55 + 0.45 * warm, "free")
