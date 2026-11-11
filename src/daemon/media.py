"""
src/daemon/media.py

Music, sounds and photos.

    music    "pause the music", "play", "next song", "previous track",
             "turn it up / down", "mute the sound" -- sent as the keyboard's
             media keys, so they work with whatever is playing (Spotify, a
             video in the browser) without ASH knowing which player it is
    sounds   a short system chime when a reminder or a watch goes off, so a
             spoken line is not the only sign (sounds.enabled)
    photo    "take a photo" / "take a picture": one frame from the webcam,
             saved to Pictures/ASH with the time in its name, and remembered
             as an episode so "when did I take that photo?" has an answer.
             Only on request -- never on its own, never while privacy is
             paused, never for a guest's voice.

The machine is reached through a `MediaExecutor`, so tests hand in their own.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("ash.daemon.media")

KEYS = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "volume_up": 0xAF, "volume_down": 0xAE,
        "mute": 0xAD, "stop": 0xB2}

_MUSIC = [
    (re.compile(r"^\s*(?:please\s+)?(?:pause|stop)\s+(?:the\s+)?(?:music|song|video|playback|it)\s*[.!]?\s*$"
                r"|^\s*(?:pause|stop)\s*[.!]?\s*$", re.I), "play_pause", "Paused."),
    (re.compile(r"^\s*(?:please\s+)?(?:resume|play|unpause|continue)(?:\s+(?:the\s+|some\s+|my\s+)?(?:music|songs?|playback|it|tunes))?"
                r"\s*[.!]?\s*$", re.I), "play_pause", "Playing."),
    (re.compile(r"^\s*(?:please\s+)?(?:next|skip)(?:\s+(?:this\s+|the\s+)?(?:song|track|one))?\s*[.!]?\s*$"
                r"|^\s*(?:play\s+)?(?:the\s+)?next\s+(?:song|track)\s*[.!]?\s*$", re.I), "next", "Next one."),
    (re.compile(r"^\s*(?:please\s+)?(?:previous|go back|last)(?:\s+(?:song|track|one))?\s*[.!]?\s*$"
                r"|^\s*(?:play\s+)?(?:the\s+)?previous\s+(?:song|track)\s*[.!]?\s*$", re.I), "previous", "Going back."),
    (re.compile(r"^\s*(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:make\s+it\s+(?:a\s+(?:bit|little)\s+)?louder"
                r"|(?:turn|bring|pump|crank)\s+(?:it|the\s+(?:volume|music|sound))\s+up(?:\s+(?:a\s+)?(?:bit|little|notch))?"
                r"|(?:raise|increase|boost)\s+the\s+volume(?:\s+(?:a\s+)?(?:bit|little|notch))?"
                r"|volume\s+up|(?:a\s+(?:bit|little)\s+)?louder)(?:\s+please)?\s*[.!]?\s*$",
                re.I), "volume_up", "Louder."),
    (re.compile(r"^\s*(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:make\s+it\s+(?:a\s+(?:bit|little)\s+)?(?:quieter|softer|lower)"
                r"|(?:turn|bring)\s+(?:it|the\s+(?:volume|music|sound))\s+down(?:\s+(?:a\s+)?(?:bit|little|notch))?"
                r"|(?:lower|reduce|decrease)\s+the\s+volume(?:\s+(?:a\s+)?(?:bit|little|notch))?"
                r"|volume\s+down|(?:a\s+(?:bit|little)\s+)?(?:quieter|softer))(?:\s+please)?\s*[.!]?\s*$",
                re.I), "volume_down", "Quieter."),
    (re.compile(r"^\s*(?:please\s+)?(?:mute|unmute)\s+(?:the\s+)?(?:sound|volume|audio|music)\s*[.!]?\s*$", re.I),
     "mute", "Done."),
]
_PHOTO = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|can you\s+|could you\s+)?(?:take|snap|grab)\s+"
                    r"(?:a|my|another|one)\s+(?:photo|picture|pic|selfie|snapshot)(?:\s+(?:of me|please|now))*"
                    r"\s*[.!?]?\s*$", re.I)


class MediaExecutor:
    """Does it on Windows. Every method returns True if it worked."""

    def key(self, name: str) -> bool:
        if not sys.platform.startswith("win"):
            return False
        try:
            import ctypes
            vk = KEYS[name]
            user32 = ctypes.windll.user32
            user32.keybd_event(vk, 0, 0, 0)
            user32.keybd_event(vk, 0, 2, 0)            # KEYEVENTF_KEYUP
            return True
        except Exception:
            logger.exception("Media key failed")
            return False

    def chime(self, kind: str) -> bool:
        try:
            import winsound
            alias = {"reminder": "SystemNotification", "alert": "SystemExclamation"}.get(kind, "SystemDefault")
            winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
            return True
        except Exception:
            return False

    def photo(self, path: str, index: int = 0) -> bool:
        try:
            import cv2
            cap = cv2.VideoCapture(index)
            try:
                if not cap.isOpened():
                    return False
                for _ in range(5):                    # let the exposure settle
                    cap.read()
                ok, frame = cap.read()
                if not ok or frame is None:
                    return False
                os.makedirs(os.path.dirname(path), exist_ok=True)
                return bool(cv2.imwrite(path, frame))
            finally:
                cap.release()
        except Exception:
            logger.exception("Taking a photo failed")
            return False


@dataclass
class MediaConfig:
    enabled: bool = True
    music: bool = True
    sounds: bool = True
    photos: bool = True
    photo_dir: str = "~/Pictures/ASH"
    camera_index: int = 0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MediaConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                cur = getattr(c, k)
                try:
                    setattr(c, k, bool(v) if isinstance(cur, bool) else type(cur)(v))
                except (TypeError, ValueError):
                    pass
        return c


class Media:
    def __init__(self, cfg: Optional[MediaConfig] = None, executor: Optional[MediaExecutor] = None,
                 episodic=None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or MediaConfig()
        self.ex = executor or MediaExecutor()
        self.episodic = episodic
        self.clock = clock
        self.stats = {"keys": 0, "chimes": 0, "photos": 0, "failed": 0}

    def hear(self, text: str, privacy_paused: bool = False) -> Optional[str]:
        """A music or photo request: what to say back, or None."""
        if not self.cfg.enabled:
            return None
        t = (text or "").strip()
        if not t or len(t) > 80:
            return None
        if self.cfg.music:
            for rx, key, say in _MUSIC:
                if rx.match(t):
                    ok = self.ex.key(key)
                    self.stats["keys" if ok else "failed"] += 1
                    return say if ok else "I couldn't reach the media controls."
        if self.cfg.photos and _PHOTO.match(t):
            if privacy_paused:
                return "I'm paused for privacy, so no photos right now."
            now = self.clock()
            name = datetime.fromtimestamp(now).strftime("%Y-%m-%d_%H%M%S") + ".jpg"
            path = os.path.join(os.path.expanduser(self.cfg.photo_dir), name)
            if not self.ex.photo(path, self.cfg.camera_index):
                self.stats["failed"] += 1
                return "I couldn't get a picture from the camera."
            self.stats["photos"] += 1
            self._remember(path, now)
            return f"Got it -- saved as {name} in {os.path.basename(os.path.expanduser(self.cfg.photo_dir))}."
        return None

    def _remember(self, path: str, now: float):
        if self.episodic is None:
            return
        try:
            when = datetime.fromtimestamp(now).strftime("%A %d %B at %H:%M")
            self.episodic.add_episode(f"The user asked ASH to take a photo on {when}; it was saved to {path}.",
                                      "observation", importance=0.6)
        except Exception:
            logger.exception("Could not remember the photo")

    def cue(self, kind: str) -> bool:
        """A short sound with something ASH is about to say (reminder, alert)."""
        if not (self.cfg.enabled and self.cfg.sounds):
            return False
        ok = self.ex.chime(kind)
        if ok:
            self.stats["chimes"] += 1
        return ok

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, **self.stats}
