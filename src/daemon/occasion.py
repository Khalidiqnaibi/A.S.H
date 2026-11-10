"""
src/daemon/occasion.py

What the user was doing while something else went on.

"Khalid listens to this while he's coding" is two facts, a song and a stretch of
work, and the link between them is only worth keeping once it has happened
several times. The song comes from senses/nowplaying.py. This is the other half:
a short memory of which kind of work was in front of the user, so a play can be
asked "what were you doing during this?".

It is deliberately small and deliberately honest:

  * Only a short list of programs say what the user is doing -- editors and
    IDEs are "coding", Word and Obsidian "writing", Office and chat tools
    "working", a PDF reader "reading", Figma and Photoshop "designing", a game
    launcher or a game "gaming". A browser says nothing (it could be anything),
    and neither does a terminal (ASH's own is one), so those leave the
    occasion empty rather than guessed.
  * The words are the ones taste.py already allows for an occasion, so what is
    observed and what is said ("I listen to lo-fi while I'm coding") land in the
    same place.
  * It keeps two hours of segments, no titles and no text: a program's name
    maps to a word, and the program's name is not kept.
  * An occasion is claimed only if one activity covered at least half of the
    play. Idle time is not an activity.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

_GROUPS = {
    "coding": """code cursor windsurf zed pycharm pycharm64 idea idea64 webstorm webstorm64 clion clion64 rider rider64
        goland goland64 rustrover rustrover64 datagrip datagrip64 phpstorm phpstorm64 studio64 devenv eclipse
        sublime_text notepad++ atom vim gvim nvim nvim-qt emacs""",
    "writing": "winword obsidian typora notion scrivener wordpad",
    "working": "excel powerpnt outlook onenote teams ms-teams slack zoom msteams",
    "reading": "acrobat acrord32 sumatrapdf foxitreader kindle calibre ebook-viewer",
    "designing": "figma photoshop illustrator blender inkscape gimp xd indesign afterfx premiere",
    "gaming": """steam epicgameslauncher valorant-win64-shipping leagueclient fortniteclient-win64-shipping cs2 csgo dota2
        robloxplayerbeta eldenring gta5 minecraft javaw riotclientservices battle.net""",
}
ACTIVITY_OF = {p: act for act, procs in _GROUPS.items() for p in procs.split()}


def activity_of(process: str) -> str:
    """The word for what a program says the user is doing, or ""."""
    p = re.sub(r"\.(?:exe|app|bin)$", "", (process or "").strip().lower())
    return ACTIVITY_OF.get(p, "")


class Occasion:
    """Which activity was in front of the user, over the last couple of hours."""

    def __init__(self, clock: Callable[[], float] = time.time, keep_s: float = 7200.0, idle_cut_s: float = 120.0):
        self.clock = clock
        self.keep_s = float(keep_s)
        self.idle_cut_s = float(idle_cut_s)
        self._segs: Deque[Tuple[float, float, str]] = deque()      # (start, end, activity), closed
        self._cur: Optional[Tuple[float, str]] = None               # (since, activity) still going
        self._lock = threading.Lock()

    # -- feeding it ------------------------------------------------------------
    def window(self, process: str, now: Optional[float] = None, blocked: bool = False):
        """The focused window changed."""
        now = self.clock() if now is None else now
        act = "" if blocked else activity_of(process)
        with self._lock:
            if self._cur is not None and self._cur[1] == act:
                return
            self._close(now)
            self._cur = (now, act) if act else None

    def tick(self, now: Optional[float] = None, idle_s: Optional[float] = None):
        """Away from the keyboard is not an activity."""
        now = self.clock() if now is None else now
        with self._lock:
            if self._cur is not None and idle_s is not None and idle_s >= self.idle_cut_s:
                self._close(max(self._cur[0], now - idle_s))        # it ended when they went quiet
            while self._segs and now - self._segs[0][1] > self.keep_s:
                self._segs.popleft()

    def _close(self, end: float):
        if self._cur is not None and end > self._cur[0]:
            self._segs.append((self._cur[0], end, self._cur[1]))
        self._cur = None

    # -- asking it -------------------------------------------------------------
    def during(self, start: float, end: float, now: Optional[float] = None) -> str:
        """The activity that covered at least half of [start, end], or ""."""
        now = self.clock() if now is None else now
        span = end - start
        if span <= 0:
            return ""
        with self._lock:
            segs: List[Tuple[float, float, str]] = list(self._segs)
            if self._cur is not None:
                segs.append((self._cur[0], max(now, self._cur[0]), self._cur[1]))
        got = {}
        for s, e, act in segs:
            overlap = min(e, end) - max(s, start)
            if overlap > 0:
                got[act] = got.get(act, 0.0) + overlap
        if not got:
            return ""
        top = max(got, key=lambda a: got[a])
        return top if got[top] >= span * 0.5 else ""

    def now(self) -> str:
        with self._lock:
            return self._cur[1] if self._cur is not None else ""

    def status(self) -> dict:
        return {"now": self.now(), "segments": len(self._segs)}
