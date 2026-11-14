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
}
# Games and their launchers, by program name. A name with spaces is how the program itself is called
# ("League of Legends.exe"), so this is a list, not a split string.
GAME_PROCS = frozenset("""steam steamwebhelper epicgameslauncher battle.net riotclientservices riotclientux leagueclient
    leagueclientux leagueclientuxrender league of legends leagueoflegends valorant valorant-win64-shipping
    fortniteclient-win64-shipping fortniteclient-win64-shipping_be cs2 csgo dota2 robloxplayerbeta robloxstudiobeta
    eldenring gta5 gtav minecraft minecraftlauncher overwatch overwatch2 r5apex apex rocketleague destiny2 cod
    modernwarfare warzone helldivers2 palworld-win64-shipping marvelrivals-win64-shipping bg3 bg3_dx11 baldursgate3
    witcher3 cyberpunk2077 hogwartslegacy starfield genshinimpact yuanshen starrail zenlesszonezero pubg tslgame
    hearthstone wow worldofwarcraft diablo iv diablo4 d4 fallout4 skyrimse eldenring nightreign sekiro darksouls3
    fifa23 eafc24 eafc25 fc25 nba2k24 nba2k25 rdr2 godofwar terraria stardewvalley amongus fallguys rustclient rust
    escapefromtarkov tarkov pathofexile pathofexile_x64 lostark kovaaks aimlab valheim satisfactory
    cities2 eu4 ck3 hoi4 stellaris""".split() + ["league of legends", "league of legends (tm) client"]) - {"of", "iv", "league", "legends", "diablo"}
ACTIVITY_OF = {p: act for act, procs in _GROUPS.items() for p in procs.split()}
ACTIVITY_OF.update({p: "gaming" for p in GAME_PROCS})
# Launchers and stores: "gaming" for who is free to talk, but not a game running behind everything.
# (The League client is open for hours before and after a match, and so is Steam: neither is "a game is running".)
LAUNCHERS = frozenset("""steam steamwebhelper epicgameslauncher battle.net riotclientservices riotclientux minecraftlauncher
    leagueclient leagueclientux leagueclientuxrender""".split())


def is_known_game(stem: str, launchers: bool = True) -> bool:
    s = (stem or "").strip().lower()
    return s in GAME_PROCS and (launchers or s not in LAUNCHERS)


BROWSERS = frozenset({"chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "chromium", "arc"})
# What a page category says about the user's activity, once the tab sensor reports it. Video, social, news and
# shopping say nothing (that is leisure or errands, not an activity the profile has a word for).
LEISURE_PAGES = frozenset({"video", "music", "social", "news", "shopping"})
PAGE_ACTIVITY = {"code": "coding", "learning": "studying", "work": "working", "docs": "working",
                 "reading": "reading", "games": "gaming"}


def page_activity(category: str) -> str:
    return PAGE_ACTIVITY.get((category or "").strip().lower(), "")


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
        self._page = ""                                             # the activity of the browser tab in front, if any
        self._browser_front = False
        self._leisure = False                                       # the tab in front is video, social, news...
        self._lock = threading.Lock()

    # -- feeding it ------------------------------------------------------------
    def window(self, process: str, now: Optional[float] = None, blocked: bool = False):
        """The focused window changed."""
        now = self.clock() if now is None else now
        stem = re.sub(r"\.(?:exe|app|bin)$", "", (process or "").strip().lower())
        with self._lock:
            self._browser_front = stem in BROWSERS
            if not self._browser_front:
                self._leisure = False
            act = "" if blocked else (activity_of(process) or (self._page if self._browser_front else ""))
            if self._cur is not None and self._cur[1] == act:
                return
            self._close(now)
            self._cur = (now, act) if act else None

    def page(self, category: str, now: Optional[float] = None, stopped: bool = False):
        """The tab sensor says which kind of page is in front of a focused browser (or that it stopped).
        A browser's window title says nothing; this does, so "while coding" works for a day of Stack Overflow."""
        now = self.clock() if now is None else now
        with self._lock:
            if stopped:
                self._page, self._leisure = "", False
                if self._browser_front:                 # only if a browser is still what is in front: a late
                    self._close(now)                    # "stopped" must not end what an editor started since
                self._browser_front = False
                return
            act = page_activity(category)
            self._page, self._browser_front = act, True
            self._leisure = (category or "").strip().lower() in LEISURE_PAGES
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

    def context(self) -> str:
        """The activity for the social learner: the one in front, else "leisure" if a leisure page is, else "".
        A short fixed list, so what it learns is not spread thin over names."""
        with self._lock:
            if self._cur is not None and self._cur[1]:
                return self._cur[1]
            return "leisure" if (self._browser_front and self._leisure) else ""

    def now(self) -> str:
        with self._lock:
            return self._cur[1] if self._cur is not None else ""

    def status(self) -> dict:
        return {"now": self.now(), "segments": len(self._segs)}
