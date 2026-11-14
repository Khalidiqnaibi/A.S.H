"""
src/daemon/gamemode.py

Stay out of the way of a game.

ASH is always on, and on a gaming PC that is the problem: with a game running it was reading the screen, listening to the
game's sound and transcribing it, running a vision model on every captured frame, and holding two gigabytes -- all on the
machine the game needs. Task Manager showed 100% CPU and RAM.

When a game is in front -- a known game, or an unknown program that fills the whole screen -- or a known game is running
at all, ASH goes quiet in the ways that cost something:

  * its own process drops to idle priority (and Ollama's, if it is running), so the game always wins a core
  * the screen sensor (capture, OCR, vision) and the microphone (speech recognition on game audio) are paused -- only the
    ones this switched off, and only until the game is gone
  * the vision model is released from memory, and the language model is told to unload (it loads again, in a few
    seconds, the next time you ask something)
  * nothing is said unprompted (availability.py already treats a game as "can't talk")

A short delay going in (a game's launcher is not the game) and a longer one coming out (alt-tabbing for a minute is not
leaving), so it does not flap. Everything comes back by itself. `ashctl game` shows the state; `game_mode` in daemon.json
turns any part of it off. Nothing here calls a model.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .occasion import BROWSERS, GAME_PROCS, is_known_game

logger = logging.getLogger("ash.daemon.gamemode")

# A program that fills the screen is a game unless it is one of these.
_NOT_GAMES = frozenset("""explorer applicationframehost searchhost shellexperiencehost startmenuexperiencehost
    lockapp logonui dwm vlc mpv mpc-hc mpc-hc64 mpc-be mpc-be64 potplayer potplayermini64 wmplayer kodi plex netflix primevideo
    soffice soffice.bin acrord32 acrobat sumatrapdf screenclippinghost parsecd video.ui msteams slack discord lightshot
    gimp-2.10 krita
    disneyplus stremio stremio-shell-ng spotify powerpnt keynote excel winword outlook teams zoom code cursor
    windowsterminal wt cmd powershell pwsh conhost mintty devenv pycharm64 idea64 studio64 figma photoshop
    illustrator blender obs64 obs streamlabs obs-studio mstsc vmconnect virtualboxvm vmware teamviewer anydesk
    chrome msedge firefox brave opera vivaldi arc chromium""".split())


@dataclass
class GameModeConfig:
    enabled: bool = True
    enter_after_s: float = 8.0          # a game in front this long
    leave_after_s: float = 45.0         # and gone this long, before everything comes back
    scan_s: float = 15.0                # how often to look for a known game running behind
    lower_priority: bool = True
    pause_screen: bool = True           # capture, OCR and the vision model
    pause_mic: bool = True              # speech recognition on the game's audio
    free_model: bool = True             # unload the language model; it reloads on the next question
    fullscreen_counts: bool = True      # an unknown program that fills the screen is taken for a game

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GameModeConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                try:
                    setattr(c, k, type(getattr(c, k))(v))
                except (TypeError, ValueError):
                    pass
        return c


def _stem(process: str) -> str:
    import re
    return re.sub(r"\.(?:exe|app|bin)$", "", (process or "").strip().lower())


def looks_like_game(process: str, fullscreen: bool = False, fullscreen_counts: bool = True) -> str:
    """Why this foreground program is a game: "game", "fullscreen", or ""."""
    s = _stem(process)
    if not s:
        return ""
    if is_known_game(s, launchers=False):
        return "game"
    if fullscreen and fullscreen_counts and s not in _NOT_GAMES and s not in BROWSERS:
        return "fullscreen"
    return ""


def running_games() -> List[str]:
    """Known games among the running programs, by name only (a few milliseconds)."""
    try:
        import psutil
        out = []
        for p in psutil.process_iter(["name"]):
            n = _stem(p.info.get("name") or "")
            if n and is_known_game(n, launchers=False) and n not in out:
                out.append(n)
        return out
    except Exception:
        return []


class GameMode:
    def __init__(self, cfg: Optional[GameModeConfig] = None, clock: Callable[[], float] = time.time,
                 periphery=None, release: Optional[Callable[[], None]] = None,
                 restore: Optional[Callable[[], None]] = None,
                 scan: Callable[[], List[str]] = running_games,
                 own_priority: Optional[Callable[[bool], None]] = None,
                 unload_model: Optional[Callable[[], None]] = None,
                 held: Optional[Callable[[], bool]] = None,
                 on_held: Optional[Callable[[str], None]] = None):
        self.cfg = cfg or GameModeConfig()
        self.clock = clock
        self.periphery = periphery
        self._release = release            # free what is in memory (the vision model)
        self._restore = restore
        self._scan = scan
        self._priority = own_priority or _set_priority
        self._unload = unload_model or unload_ollama
        self._held = held                  # privacy mode is on: nothing sensitive may be switched on, by anyone
        self._on_held = on_held            # ...so a sensor that would have been resumed is handed to privacy mode instead
        self.active = False
        self.why = ""
        self.since = 0.0
        self.paused: List[str] = []         # sensors this switched off, and so is the one to switch back on
        self._seen_since = 0.0              # a game first seen at
        self._gone_since = 0.0
        self._last_scan = -1e9
        self._running: List[str] = []
        self.stats = {"entered": 0, "left": 0}
        self._lock = threading.Lock()

    # -- looking ------------------------------------------------------------------------------------
    def look(self, process: str, fullscreen: bool, now: Optional[float] = None) -> str:
        """Feed it what is in front. Returns why a game is up ("" if none)."""
        if not self.cfg.enabled:
            return ""
        now = self.clock() if now is None else now
        why = looks_like_game(process, fullscreen, self.cfg.fullscreen_counts)
        if not why and now - self._last_scan >= self.cfg.scan_s:
            self._last_scan = now
            self._running = self._scan() or []
        if not why and self._running:
            why = "running"
        with self._lock:
            if why:
                self._gone_since = 0.0
                self._seen_since = self._seen_since or now
                if not self.active and now - self._seen_since >= self.cfg.enter_after_s:
                    self._enter(why, now)
            else:
                self._seen_since = 0.0
                if self.active:
                    self._gone_since = self._gone_since or now
                    if now - self._gone_since >= self.cfg.leave_after_s:
                        self._leave(now)
        return why

    # -- going in and out ----------------------------------------------------------------------------
    def _enter(self, why: str, now: float):
        self.active, self.why, self.since = True, why, now
        self.stats["entered"] += 1
        logger.warning("Game mode on (%s): keeping out of the way", why)
        c = self.cfg
        if c.lower_priority:
            self._safely(self._priority, True)
        if c.pause_screen or c.pause_mic:
            self._pause_sensors()
        if c.pause_screen and self._release is not None:
            self._safely(self._release)
        if c.free_model:
            threading.Thread(target=self._safely, args=(self._unload,), daemon=True, name="ash-unload").start()

    def _leave(self, now: float):
        self.active = False
        self.stats["left"] += 1
        logger.warning("Game mode off after %.0f minutes", (now - self.since) / 60.0)
        if self.cfg.lower_priority:
            self._safely(self._priority, False)
        self._resume_sensors()
        if self._restore is not None:
            threading.Thread(target=self._safely, args=(self._restore,), daemon=True, name="ash-wake").start()
        self.why, self._seen_since = "", 0.0

    def _pause_sensors(self):
        sensors = getattr(self.periphery, "sensors", None) or {}
        names = []
        if self.cfg.pause_screen:
            names.append("screenshot")
        if self.cfg.pause_mic:
            names.append("microphone")
        for n in names:
            s = sensors.get(n)
            try:
                if s is not None and s.state.value == "active":      # never touch what the user already turned off
                    s.disable()
                    self.paused.append(n)
            except Exception:
                logger.exception("Could not pause %s for the game", n)

    def forget(self, name: str):
        """The user (or privacy mode) switched this sensor themselves while a game was up: it is theirs now, and game mode
        must not switch it back on when the game ends."""
        with self._lock:
            if name in self.paused:
                self.paused.remove(name)

    def _resume_sensors(self):
        sensors = getattr(self.periphery, "sensors", None) or {}
        held = False
        try:
            held = bool(self._held()) if self._held is not None else False
        except Exception:
            held = False
        for n in self.paused:
            s = sensors.get(n)
            try:
                if s is None or s.state.value != "disabled":
                    continue
                if held:
                    if self._on_held is not None:
                        self._on_held(n)          # privacy mode brings it back when privacy mode ends, not before
                    continue
                s.enable()
            except Exception:
                logger.exception("Could not resume %s after the game", n)
        self.paused = []

    @staticmethod
    def _safely(fn, *a):
        try:
            fn(*a)
        except Exception:
            logger.exception("Game mode step failed")

    # -- saying ---------------------------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "active": self.active, "why": self.why,
                "since": round(self.since, 1) if self.active else None, "paused": list(self.paused),
                "running": list(self._running), **self.stats}

    def report(self) -> str:
        if not self.cfg.enabled:
            return "Game mode is off (game_mode.enabled in daemon.json)."
        if not self.active:
            return "No game up, so ASH is running normally."
        mins = int((self.clock() - self.since) / 60)
        paused = " and ".join(self.paused) if self.paused else "nothing"
        return (f"A game is up ({self.why}) for {mins} minute{'s' if mins != 1 else ''}: ASH is at idle priority, "
                f"with {paused} paused, and says nothing unprompted.")


# ----------------------------------------------------------------------------
# What it does to the machine
# ----------------------------------------------------------------------------
_BASELINE: Dict[int, Tuple[float, int]] = {}


def _set_priority(low: bool):
    """Idle priority for ASH (and Ollama's processes) while a game runs; normal after."""
    import psutil
    if os.name == "nt":
        lowest = psutil.IDLE_PRIORITY_CLASS
    else:
        lowest = 19
    procs = [psutil.Process()]
    if low:
        try:
            for p in psutil.process_iter(["name"]):
                if (p.info.get("name") or "").lower().startswith("ollama"):
                    procs.append(p)
        except Exception:
            pass
        for p in procs:
            try:
                _BASELINE[p.pid] = (p.create_time(), p.nice())      # what it was (ASH itself starts below normal), and which process
                p.nice(lowest)
            except Exception:
                logger.debug("Could not lower the priority of %s", getattr(p, "pid", "?"), exc_info=True)
        return
    # Back: only the processes that were lowered, only if they are still the same process (a pid can be reused), and what
    # started during the game -- a new Ollama runner -- is left as it is.
    for pid, (created, was) in list(_BASELINE.items()):
        try:
            p = psutil.Process(pid)
            if abs(p.create_time() - created) < 1.0:
                p.nice(was)
        except Exception:
            pass
    _BASELINE.clear()


def unload_ollama(model: str = "", url: str = ""):
    """Ask Ollama to drop the model from memory now. It reloads on the next request."""
    base = url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    gen = base.replace("/api/chat", "/api/generate")
    model = model or os.environ.get("OLLAMA_MODEL", "").strip('"') or "mistral:latest"
    if not gen.startswith(("http://127.0.0.1", "http://localhost")):
        return                                           # only ever a model on this machine
    req = urllib.request.Request(gen, data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=4).read()
