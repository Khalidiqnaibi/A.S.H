"""
src/senses/nowplaying.py

What is playing right now: a song in Spotify, a video in a browser tab, a film
in VLC -- read from the titles of the windows that are open, not from the
screen and not from the audio.

Why titles: a player puts what it is playing in its window title ("Artist -
Song" in Spotify, "Title - YouTube - Google Chrome" in a tab), and the active
window sensor already trusts titles. Nothing is captured, nothing is read off
the pixels, and only windows of *known players* are looked at -- every other
title is dropped on the spot, never kept.

The point is not "now playing" for its own sake. A track that played to the end
is evidence the person likes it; one that was replaced after ten seconds is
evidence they don't; the same one three times in a week is a favourite. So the
sensor reports, with each change, what has just *finished* and for how long:

    event: "started" / "changed" / "stopped"
    meta:  player, kind, title, artist, item            (what is playing now)
           finished, finished_s, finished_how, ...      (what just ended)

taste.Taste.watch() turns those into a profile, with weights.

Honest limits, which the manual repeats:
  * A title says what is *open*, not that it is making sound. A paused browser
    tab looks the same as a playing one; Spotify is the exception (its title
    goes back to "Spotify" when paused). Pass `audible=` (pids that are making
    sound) to be sure, where a platform offers it.
  * A browser window shows only its front tab.
  * Stremio and similar players keep a blank title, so they are not seen.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .base import Capability, Modality, Sensor, SensorEvent

logger = logging.getLogger("ash.senses.nowplaying")

_SEP = r"\s+[-–—|]\s+"
_NOISE = re.compile(r"^\s*(?:[●•*▶⏵]\s*|\(\d+\)\s*)+")

_AUDIO_EXT = ("mp3", "flac", "m4a", "ogg", "opus", "wav", "aac", "wma")
_VIDEO_EXT = ("mkv", "mp4", "avi", "mov", "webm", "m4v", "wmv", "ts")
_EXT = re.compile(r"\.(" + "|".join(_AUDIO_EXT + _VIDEO_EXT) + r")\s*$", re.I)

# What a browser adds after the page title: its name, Edge's profile before
# that, and "and 3 more pages".
_BROWSER_TAIL = re.compile(
    _SEP + r"(?:Google Chrome|Microsoft Edge|Mozilla Firefox|Brave|Opera|Vivaldi|Chromium|Arc)\s*$", re.I)
_PROFILE_TAIL = re.compile(_SEP + r"(?:Personal|Work|School|Guest|Profile\s*\d+)\s*$", re.I)
_MORE_PAGES = re.compile(r"\s+and\s+\d+\s+more\s+pages?\s*$", re.I)

BROWSERS = frozenset({"chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "chromium", "arc"})
_SPOTIFY_IDLE = frozenset({"spotify", "spotify free", "spotify premium", "advertisement", "spotify - web player",
                           "spotify – web player", "web player"})


@dataclass(frozen=True)
class Playing:
    player: str             # "Spotify", "Chrome", "VLC"
    kind: str               # music | video | show | stream | media
    title: str
    artist: str = ""

    @property
    def item(self) -> str:
        return f"{self.title} by {self.artist}" if self.artist else self.title

    @property
    def ident(self) -> Tuple[str, str]:
        return (self.kind, self.item.lower())


def _proc(process: str) -> str:
    return re.sub(r"\.(?:exe|app|bin)$", "", (process or "").strip().lower())


def _tidy(s: str) -> str:
    return re.sub(r"\s+", " ", _NOISE.sub("", s or "")).strip(" -–—|")


def _by_extension(title: str, default: str) -> Tuple[str, str]:
    m = _EXT.search(title)
    if not m:
        return title, default
    ext = m.group(1).lower()
    return _EXT.sub("", title).strip(), ("music" if ext in _AUDIO_EXT else "video")


def _split_artist(s: str) -> Tuple[str, str]:
    """"Artist - Song" -> (song, artist); no separator -> (s, "")."""
    parts = re.split(_SEP, s, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[1].strip(), parts[0].strip()
    return s.strip(), ""


# Sites a browser tab can be playing, as (pattern on the page title, kind).
# The first group is the thing; a second, if any, is the artist/channel.
_SITES: List[Tuple["re.Pattern", str, str]] = [
    (re.compile(r"^(?P<t>.+?)" + _SEP + r"YouTube Music$", re.I), "music", "YouTube Music"),
    (re.compile(r"^(?P<t>.+?)" + _SEP + r"YouTube$", re.I), "video", "YouTube"),
    (re.compile(r"^(?:Watch\s+)?(?P<t>.+?)(?:\s*\|\s*|" + _SEP + r")Netflix$", re.I), "show", "Netflix"),
    (re.compile(r"^Netflix\s*[-–—|:]\s*(?:Watch\s+)?(?P<t>.+)$", re.I), "show", "Netflix"),
    (re.compile(r"^Prime Video:\s*(?P<t>.+)$", re.I), "show", "Prime Video"),
    (re.compile(r"^(?:Watch\s+)?(?P<t>.+?)\s*\|\s*Disney\+$", re.I), "show", "Disney+"),
    (re.compile(r"^(?:Watch\s+)?(?P<t>.+?)" + _SEP + r"Crunchyroll$", re.I), "show", "Crunchyroll"),
    (re.compile(r"^(?P<t>.+?)" + _SEP + r"Twitch$", re.I), "stream", "Twitch"),
    (re.compile(r"^Stream\s+(?P<t>.+?)\s+by\s+(?P<a>.+?)\s*\|\s*Listen online.*SoundCloud$", re.I),
     "music", "SoundCloud"),
]
_SITE_IDLE = frozenset({"youtube", "youtube music", "netflix", "twitch", "soundcloud", "prime video", "disney+",
                        "crunchyroll", "spotify", "home", "new tab"})


def parse(title: str, process: str = "") -> Optional[Playing]:
    """The song or video a window says is playing, or None.

    Pure: a title and the name of the program that owns it. Anything that is not
    a recognised player, or is a player showing nothing, is None.
    """
    from .screen import clean_title
    proc = _proc(process)
    t = _tidy(clean_title(title))
    if not t or not proc:
        return None

    if proc == "spotify":
        if t.lower() in _SPOTIFY_IDLE or not re.search(_SEP, t):
            return None
        song, artist = _split_artist(t)
        return Playing("Spotify", "music", song, artist) if artist else None

    if proc in BROWSERS:
        page = _BROWSER_TAIL.sub("", t).strip()
        if page != t:
            page = _MORE_PAGES.sub("", _PROFILE_TAIL.sub("", page)).strip()
        if not page or page.lower() in _SITE_IDLE:
            return None
        for rx, kind, site in _SITES:
            m = rx.match(page)
            if not m:
                continue
            thing = _tidy(m.group("t"))
            artist = _tidy(m.groupdict().get("a") or "")
            if not thing or thing.lower() in _SITE_IDLE:
                return None
            return Playing(site, kind, thing, artist)
        return None

    if proc in ("vlc", "mpv", "mpc-hc", "mpc-hc64", "potplayermini64", "potplayermini"):
        name = {"vlc": "VLC", "mpv": "mpv"}.get(proc, "Media player")
        tail = {"vlc": r"VLC media player", "mpv": r"mpv"}.get(proc, r"(?:MPC-HC|PotPlayer)")
        m = re.match(r"^(?P<t>.+?)" + _SEP + tail + r"$", t, re.I)
        if not m:
            return None
        thing, kind = _by_extension(m.group("t").strip(), "video")
        return Playing(name, kind, thing) if thing else None

    if proc in ("foobar2000", "musicbee", "aimp"):
        m = re.match(r"^(?P<t>.+?)\s*(?:\[foobar2000\]|" + _SEP + r"MusicBee|" + _SEP + r"AIMP)$", t, re.I)
        if not m:
            return None
        song, artist = _split_artist(m.group("t"))
        return Playing({"foobar2000": "foobar2000", "musicbee": "MusicBee"}.get(proc, "AIMP"),
                       "music", song, artist)
    return None


# ----------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------
Window = Tuple[str, str, bool, int]            # (title, process, is_foreground, pid)


def enum_windows() -> List[Window]:
    """Visible top-level windows with a title, on Windows; [] elsewhere."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    u32 = ctypes.windll.user32
    u32.GetForegroundWindow.restype = wintypes.HWND
    fg = u32.GetForegroundWindow()
    found: List[Tuple[str, int, bool]] = []
    proto = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def each(hwnd, _lp):
        try:
            if not u32.IsWindowVisible(hwnd):
                return True
            n = u32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            found.append((buf.value, int(pid.value), bool(fg) and int(hwnd) == int(fg)))
        except Exception:
            pass
        return True

    u32.EnumWindows(proto(each), 0)
    names: Dict[int, str] = {}
    out: List[Window] = []
    try:
        import psutil
    except Exception:
        psutil = None
    for title, pid, foreground in found:
        if pid not in names:
            try:
                names[pid] = psutil.Process(pid).name() if psutil else ""
            except Exception:
                names[pid] = ""
        out.append((title, names[pid], foreground, pid))
    return out


# ----------------------------------------------------------------------
# The sensor
# ----------------------------------------------------------------------
@dataclass
class _Slot:
    cur: Optional[Playing] = None
    since: float = 0.0
    last_seen: float = 0.0
    carried: float = 0.0
    cand: Optional[Playing] = None
    cand_n: int = 0
    cand_t: float = 0.0


class NowPlayingSensor(Sensor):
    """One slot per player, so music in Spotify and a video in a tab are two
    separate plays, each with its own beginning and end."""

    capability = Capability(
        name="now_playing",
        description="What song or video is playing, from the titles of player windows.",
        privacy_sensitive=True,
    )
    default_interval = 5.0
    modality = Modality.TEXT

    def __init__(self, interval: Optional[float] = None, enabled: bool = True,
                 window_sensor=None, windows: Optional[Callable[[], List[Window]]] = None,
                 audible: Optional[Callable[[], Iterable[int]]] = None,
                 clock: Callable[[], float] = time.time,
                 settle: int = 2, grace_s: float = 12.0, resume_s: float = 600.0,
                 skip_below_s: float = 30.0):
        self.window_sensor = window_sensor
        self._windows = windows or enum_windows
        self._audible = audible
        self.clock = clock
        self.settle = max(1, int(settle))
        self.grace_s = float(grace_s)
        self.resume_s = float(resume_s)
        self.skip_below_s = float(skip_below_s)
        self.slots: Dict[str, _Slot] = {}
        self._carry: Dict[Tuple[str, str], Tuple[float, float]] = {}     # play ident -> (seconds, when it stopped)
        self.last_note = ""
        super().__init__(interval=interval, enabled=enabled)

    def probe(self) -> Tuple[bool, str]:
        if self._windows is not enum_windows or sys.platform == "win32":
            return True, ""
        return False, "no window list on this platform"

    # -- what is playing now ----------------------------------------------------
    def _blocked(self, title: str, process: str) -> bool:
        ws = self.window_sensor
        try:
            if ws is not None and ws.is_blocked(title):
                return True
            from .screen import is_private_content, looks_private
            return bool(looks_private(title, process) or is_private_content(title, process))
        except Exception:
            return True                  # can't tell: leave it out

    def playing_now(self) -> List[Playing]:
        return [s.cur for s in self.slots.values() if s.cur is not None]

    def _candidates(self) -> Dict[str, Playing]:
        out: Dict[str, Playing] = {}
        fg_seen: Dict[str, bool] = {}
        loud = None
        if self._audible is not None:
            try:
                loud = set(self._audible())
            except Exception:
                loud = None
        for title, process, foreground, pid in self._windows():
            p = parse(title, process)
            if p is None or self._blocked(title, process):
                continue
            if loud is not None and pid not in loud:
                continue                 # open, but making no sound
            key = _proc(process)
            if key in out and not (foreground and not fg_seen.get(key)):
                continue                 # one per program: the focused window's, else the first
            out[key] = p
            fg_seen[key] = foreground
        return out

    # -- the loop -------------------------------------------------------------------
    def read(self) -> List[SensorEvent]:
        now = self.clock()
        cands = self._candidates()
        events: List[SensorEvent] = []
        for key in set(cands) | set(self.slots):
            ev = self._step(key, cands.get(key), now)
            if ev is not None:
                events.append(ev)
        for key in [k for k, s in self.slots.items() if s.cur is None and s.cand is None]:
            del self.slots[key]
        for k in [k for k, (_s, t) in self._carry.items() if now - t > self.resume_s]:
            del self._carry[k]
        return events

    def _step(self, key: str, c: Optional[Playing], now: float) -> Optional[SensorEvent]:
        s = self.slots.setdefault(key, _Slot())
        if c is not None and c == s.cur:
            s.last_seen, s.cand, s.cand_n = now, None, 0
            return None
        if c is not None:
            if s.cand == c:
                s.cand_n += 1
            else:
                s.cand, s.cand_n, s.cand_t = c, 1, now
            if s.cand_n < self.settle:
                return None
            done = self._finish(s, "changed")
            began = s.cand_t
            s.cur, s.cand, s.cand_n = c, None, 0
            s.since = s.last_seen = began
            s.carried = 0.0
            seen = self._carry.pop(c.ident, None)
            if seen is not None and now - seen[1] <= self.resume_s:
                s.carried = seen[0]                  # a pause and a resume is one listen, not two
            return self._event("changed" if done else "started", c, done, now)
        s.cand, s.cand_n = None, 0
        if s.cur is not None and now - s.last_seen >= self.grace_s:
            done = self._finish(s, "stopped")
            s.cur = None
            return self._event("stopped", None, done, now)
        return None

    def _finish(self, s: _Slot, how: str) -> Optional[Dict[str, Any]]:
        if s.cur is None:
            return None
        secs = max(0.0, s.last_seen - s.since) + s.carried
        if how == "stopped":
            self._carry[s.cur.ident] = (secs, s.last_seen)
        return {"p": s.cur, "secs": secs, "how": how}

    def _event(self, what: str, c: Optional[Playing], done: Optional[Dict[str, Any]], now: float) -> SensorEvent:
        meta: Dict[str, Any] = {"event": what}
        if c is not None:
            meta.update(player=c.player, kind=c.kind, title=c.title, artist=c.artist, item=c.item)
        if done is not None:
            p = done["p"]
            meta.update(finished=p.item, finished_title=p.title, finished_artist=p.artist, finished_kind=p.kind,
                        finished_player=p.player, finished_s=int(done["secs"]), finished_how=done["how"],
                        finished_skip=bool(done["how"] == "changed" and done["secs"] < self.skip_below_s))
        if c is not None:
            text = f"now playing {c.item} on {c.player}"
        elif done is not None:
            text = f"stopped playing {done['p'].item}"
        else:
            text = "nothing playing"
        return SensorEvent(source=self.name, modality=Modality.TEXT, text=text, salience=0.1, ts=now,
                           persistable=False, meta=meta)

    def status(self) -> Dict[str, Any]:
        d = super().status()
        d["playing"] = [p.item for p in self.playing_now()]
        return d
