"""
src/daemon/feelings.py

BMO's moods: feelings that last -- across the afternoon, and across a restart -- instead of a face that only reacts
for half a minute. By rule, from what actually happened, never made up by a model.

Two numbers, kept in state/feelings.json:
    cheer    -1 .. 1   thanks, laughing, "good job", a celebration lift it; "you're useless", "shut up" lower it
    energy    0 .. 1   high in the day, low late at night; being left alone for hours wears it down into boredom
Both drift back to the middle over a few hours, so yesterday still shows a little in the morning, and is gone by noon.

    how are you feeling? / how are you? / are you ok?     answered from them, with the reason:
        "Pretty happy -- you've thanked me three times today."
        "A bit bored, honestly. It's been quiet since lunch."
        "Sleepy. It's late -- you should be too."
    and the face wears the feeling while it lasts (happy, love, sad, annoyed, sleepy, a bored look around).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

_KIND = [
    ("thanks", +0.12, re.compile(r"\b(?:thank\s*you|thanks|thx|ty|cheers|appreciate\s+it)\b", re.I)),
    ("praise", +0.18, re.compile(r"\b(?:good\s+(?:job|bot|boy|girl|work)|well\s+done|nice\s+(?:one|work)|you(?:'re|\s+are)\s+(?:the\s+best|awesome|amazing|great|wonderful|smart|clever|funny)|i\s+love\s+you|love\s+you)\b", re.I)),
    ("laugh", +0.10, re.compile(r"\b(?:ha(?:ha)+|lol|lmao|hehe+|that'?s\s+(?:funny|hilarious))\b", re.I)),
    ("hurt", -0.25, re.compile(r"\b(?:you(?:'re|\s+are)\s+(?:useless|stupid|dumb|annoying|wrong|bad|terrible)|i\s+hate\s+you|you\s+suck|bad\s+(?:bot|job))\b", re.I)),
    ("hushed", -0.12, re.compile(r"\b(?:shut\s+up|go\s+away|be\s+quiet|stop\s+talking)\b", re.I)),
]
_ASK = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|hey|hi|hello)[,\s]+)?"
                  r"(?:how\s+(?:are\s+you|r\s+u|are\s+you\s+(?:feeling|doing)|do\s+you\s+feel|'?s\s+it\s+going\s+with\s+you)(?:\s+today|\s+now)?|"
                  r"are\s+you\s+(?:ok(?:ay)?|alright|happy|sad|bored|tired)|what'?s\s+your\s+mood|how'?s\s+your\s+mood|"
                  r"you\s+(?:ok(?:ay)?|good|alright))(?:\s*,?\s*ash)?\s*\??\s*$", re.I)

# What the face wears for each feeling (animations/builtin); "" = nothing special.
FACE = {"delighted": "very_happy", "loved": "love", "happy": "happy", "fine": "", "bored": "look_around",
        "sleepy": "sleepy", "low": "sad", "grumpy": "annoyed"}


class Feelings:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, half_life_s: float = 4 * 3600.0,
                 bored_after_s: float = 3 * 3600.0):
        self.path = path
        self.clock = clock
        self.half_life_s = half_life_s
        self.bored_after_s = bored_after_s
        self.cheer = 0.1
        self.energy = 0.7
        self.at = clock()
        self.last_talk = self.at
        self.today: Dict[str, int] = {}
        self.day = datetime.fromtimestamp(self.at).strftime("%Y-%m-%d")
        self._load()

    # -- keeping ------------------------------------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.cheer, self.energy = float(d.get("cheer", self.cheer)), float(d.get("energy", self.energy))
            self.at, self.last_talk = float(d.get("at", self.at)), float(d.get("last_talk", self.last_talk))
            self.today, self.day = dict(d.get("today", {})), str(d.get("day", self.day))
        except Exception:
            pass

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"cheer": round(self.cheer, 4), "energy": round(self.energy, 4), "at": self.at,
                       "last_talk": self.last_talk, "today": self.today, "day": self.day}, fh)
        os.replace(tmp, self.path)

    # -- changing -----------------------------------------------------------------------------------------
    @staticmethod
    def _target_energy(hour: int) -> float:
        if 0 <= hour < 6:
            return 0.15
        if hour >= 23:
            return 0.3
        if 6 <= hour < 9:
            return 0.55
        return 0.8

    def _drift(self, now: float):
        dt = max(0.0, now - self.at)
        k = 0.5 ** (dt / self.half_life_s)
        self.cheer = 0.1 + (self.cheer - 0.1) * k              # back toward slightly cheerful
        target = self._target_energy(datetime.fromtimestamp(now).hour)
        if now - self.last_talk > self.bored_after_s and 9 <= datetime.fromtimestamp(now).hour < 23:
            target = min(target, 0.35)                          # left alone for hours: running down
        ke = 0.5 ** (dt / 1800.0)
        self.energy = target + (self.energy - target) * ke
        self.at = now
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if day != self.day:
            self.day, self.today = day, {}

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """Something the user said. Returns the kind of thing it was for feelings ("thanks", "hurt"...), or None."""
        now = self.clock() if now is None else now
        self._drift(now)
        self.last_talk = now
        self.energy = min(1.0, self.energy + 0.05)
        for kind, delta, rx in _KIND:
            if rx.search(text or ""):
                self.cheer = max(-1.0, min(1.0, self.cheer + delta))
                self.today[kind] = self.today.get(kind, 0) + 1
                return kind
        return None

    def celebrate(self, now: Optional[float] = None, amount: float = 0.2):
        now = self.clock() if now is None else now
        self._drift(now)
        self.cheer = min(1.0, self.cheer + amount)
        self.today["celebrated"] = self.today.get("celebrated", 0) + 1

    def tick(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        self._drift(now)
        return self.label(now)

    # -- saying it ---------------------------------------------------------------------------------------
    def label(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        hour = datetime.fromtimestamp(now).hour
        if self.energy < 0.35 and (hour >= 23 or hour < 6):
            return "sleepy"
        if self.cheer <= -0.15:
            hushed, hurt = self.today.get("hushed", 0), self.today.get("hurt", 0)
            return "grumpy" if hushed >= 3 and hushed > hurt else "low"
        if self.energy < 0.42 and now - self.last_talk > self.bored_after_s:
            return "bored"
        if self.cheer >= 0.6:
            return "loved" if self.today.get("praise", 0) >= 2 else "delighted"
        if self.cheer >= 0.2:
            return "happy"
        return "fine"

    @staticmethod
    def wants(text: str) -> bool:
        return bool(_ASK.match(text or ""))

    def say(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        self._drift(now)
        lab = self.label(now)
        t = self.today
        quiet_h = (now - self.last_talk) / 3600.0
        if lab == "sleepy":
            return "Sleepy. It's late -- you should be too."
        if lab == "grumpy":
            return "A bit grumpy, if I'm honest. I've been told to be quiet a lot today."
        if lab == "low":
            return ("A little down, honestly -- today's been rough on me." if t.get("hurt") else
                    "A little flat. It'll pass.")
        if lab == "bored":
            return f"A bit bored, honestly. It's been quiet for {int(quiet_h)} hours."
        if lab in ("loved", "delighted"):
            why = (f"you've said nice things to me {t.get('praise')} times today" if t.get("praise") else
                   f"you've thanked me {t.get('thanks')} times today" if t.get("thanks") else
                   "we've had a good day")
            return f"Really good -- {why}."
        if lab == "happy":
            why = (f"you've thanked me {_times(t.get('thanks', 0))} today" if t.get("thanks") else
                   "you made me laugh" if t.get("laugh") else "things are going well")
            return f"Pretty happy -- {why}."
        return "I'm good, thanks. Steady." + (" How about you?" if 9 <= datetime.fromtimestamp(now).hour < 22 else "")

    def face(self, now: Optional[float] = None) -> str:
        return FACE.get(self.label(now), "")

    def status(self) -> Dict[str, Any]:
        return {"feeling": self.label(), "cheer": round(self.cheer, 2), "energy": round(self.energy, 2), "today": dict(self.today)}


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")
