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
# "What's my emotional status?" / "how am I feeling" / "what do you think I'm feeling" / "what is the user feeling right now": about the
# PERSON, not ASH. These were answered with ASH's own mood ("Pretty happy -- you've thanked me 3 times") or looked up on Wikipedia.
_USER_MOOD = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|but|ok(?:ay)?|now)[,\s]+)?(?:"
    r"(?:what'?s|whats|what\s+is|how'?s|hows|how\s+is|tell\s+me|what\s+about)\s+(?:about\s+)?(?:my|the\s+user'?s?)\s+(?:emotional\s+|mental\s+|current\s+)?"
    r"(?:status|state|mood|condition|feelings?|vibe|energy|state\s+of\s+mind)(?:\s+(?:right\s+)?now|\s+today)?|"
    r"(?:how|what)\s+(?:am\s+i|do\s+i)\s+(?:feeling|feel)(?:\s+(?:today|now|right\s+now))?|how\s+am\s+i(?:\s+doing)?(?:\s+(?:today|now))?|"
    r"what\s+do\s+you\s+think\s+(?:i'?m|i\s+am)\s+feeling(?:\s+(?:right\s+)?now)?|"
    r"(?:what|how)\s+(?:is|does)\s+the\s+user\s+(?:feeling|feel)(?:\s+(?:right\s+)?now|\s+today)?)"
    r"(?:\s*,?\s*(?:ash|please))*\s*[?.!]*\s*$", re.I)


def wants_user_mood(text: str) -> bool:
    return bool(_USER_MOOD.match(text or ""))


_DETAIL = re.compile(r"\b(?:telemetry|numbers?|vector|victor|vitals?|stats|readings?|breakdown|details?|exactly|data|values?|metrics?|drives?)\b", re.I)


def wants_detail(text: str) -> bool:
    return bool(_DETAIL.search(text or ""))


_POS = frozenset("happy love trust excitement relief pride optimism joy calm curiosity confidence anticipation surprise".split())
_NEG = frozenset("frustration angry sad fear anxiety boredom shame guilt disgust envy jealousy pessimism".split())
_NEG_LABEL = re.compile(r"short[-\s]?tempered|irritab|annoy|frustrat|tense|upset|down|gloomy|anxious|sad|angry|grumpy|bored", re.I)


def _scale(emos) -> float:
    """The emotion engine keeps 0..10; other callers hand over 0..1. Numbers are printed against the scale they are on."""
    return 10.0 if any(v > 1.0 for _, v in emos) else 1.0


def from_engine(res, drives=None, detail: bool = False, extra: str = "") -> str:
    """How ASH feels, from the SAME state the prompt, the face and the thinking budget use (the brain's emotion engine and drives) -- not from a
    second, separate set of canned moods. Short: the feeling and what is strongest. `detail`: the numbers, and why.
    25 Nov: the numbers were printed as 10.00 / 9.72 as if 0..1, and "short-tempered" sat next to "happy, trust" -- the label is dropped when the
    strongest emotions say the opposite, and the numbers carry their scale."""
    mod = getattr(res, "modulation", None)
    label = str(getattr(mod, "tone_label", "") or "").strip().rstrip(".")
    emos = sorted(((k, float(v)) for k, v in (getattr(res, "emotions", None) or {}).items()), key=lambda t: -t[1])
    sc = _scale(emos)
    top = [(k, v) for k, v in emos if v >= 0.25 * sc][:3]
    if label and top and _NEG_LABEL.search(label) and top[0][0] in _POS:
        neg = max((v for k, v in emos if k in _NEG), default=0.0)
        if top[0][1] > neg:
            label = "Mostly " + " and ".join(k for k, _ in top[:2] if k in _POS)
    s = (label[:1].upper() + label[1:] + ".") if label else "I don't have a feeling to report yet."
    if not detail:
        if top:
            s += " Strongest: " + ", ".join(k for k, _ in top[:2]) + "."
        return (s + (" " + extra if extra else "")).strip()
    bits = [s]
    unit = " (out of 10)" if sc == 10.0 else ""
    if emos:
        bits.append("Emotions" + unit + ": " + ", ".join(f"{k} {v:.1f}" if sc == 10.0 else f"{k} {v:.2f}" for k, v in emos[:5]) + ".")
    mood = getattr(res, "mood", None) or {}
    if mood:
        bits.append("Mood baseline: " + ", ".join(f"{k} {float(v):.1f}" if sc == 10.0 else f"{k} {float(v):.2f}" for k, v in list(mood.items())[:4]) + ".")
    if drives is not None:
        bits.append("Drives (0 to 1): " + ", ".join(f"{k} {float(getattr(drives, k, 0.0)):.2f}" for k in
                                           ("urgency", "caution", "curiosity", "effort_budget", "social", "fatigue")).replace("effort_budget", "effort") + ".")
    why = [str(r) for r in (getattr(res, "reasons", None) or [])][:2]
    if why:
        bits.append("Because: " + "; ".join(why) + ".")
    return " ".join(bits)


def mood_trend(episodes, now: float, hours: float = 24.0):
    """(how many lines, average mood, share that sounded low) over the episodes in the last `hours` that carry a reading; None if too few."""
    vals = [float(e.mood) for e in episodes if getattr(e, "mood", None) is not None and now - float(getattr(e, "timestamp", 0.0) or 0.0) <= hours * 3600.0]
    if len(vals) < 5:
        return None
    return len(vals), sum(vals) / len(vals), sum(1 for v in vals if v <= -0.3) / len(vals)


def user_mood_say(mood, trend=None) -> str:
    """What ASH can honestly say about how the person feels: a reading of their WORDS (the sentiment model), said as that. With `trend`
    (mood_trend over the stored episodes) it says how the last day has been, from what is actually remembered."""
    if trend is not None:
        n, avg, low = trend
        word = "upbeat" if avg >= 0.3 else "low" if avg <= -0.3 else "mixed" if low >= 0.3 else "fairly level"
        now_part = "" if mood is None else (" Right now you sound " + ("upbeat" if mood >= 0.3 else "low" if mood <= -0.3 else "level") + ".")
        return (f"Going by the {n} things you said in the last day, you've sounded {word} -- {int(round(low * 100))}% of them read as low."
                f"{now_part} That's from your words only, so tell me if I've got it wrong.")
    if mood is None:
        return "I can't tell from here -- I only read your words, and you haven't said how you feel. How are you?"
    if mood >= 0.3:
        return "You've sounded upbeat lately -- going only by your words, so tell me if I've got it wrong."
    if mood <= -0.3:
        return "You've sounded a bit low lately -- going only by your words, so tell me if I've got it wrong."
    return "Fairly level, going by your words. I can't see more than that -- tell me if I'm off."


_ASK = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|hey|hi|hello)[,\s]+)?"
                  r"(?:how\s+(?:are\s+you|r\s+u|are\s+you\s+(?:feeling|doing)|do\s+you\s+feel|'?s\s+it\s+going\s+with\s+you)(?:\s+today|\s+now)?|"
                  r"are\s+you\s+(?:ok(?:ay)?|alright|happy|sad|bored|tired)|what'?s\s+your\s+mood|how'?s\s+your\s+mood|"
                  r"(?:what'?s|whats|what\s+is|how'?s|hows|how\s+is|tell\s+me|give\s+me|show\s+me|read\s+me|let'?s\s+(?:see|have))\s+"
                  r"(?:about\s+)?your\s+(?:emotional|mental|inner|current)\s+"
                  r"(?:status|state|mood|condition|health|telemetry|readout|report|reading|levels?|check)(?:\s+(?:right\s+)?now|\s+today)?|"
                  r"(?:what'?s|whats|what\s+is)\s+your\s+(?:state\s+of\s+mind|vibe|energy)(?:\s+(?:right\s+)?now|\s+today)?|"
                  r"how\s+(?:is|are)\s+you\s+(?:feeling|doing)\s+(?:today|now|right\s+now)|"
                  r"you\s+(?:ok(?:ay)?|good|alright))(?:\s*,?\s*ash)?\s*[?.!]*\s*$", re.I)

# What a Chooser (chooser.py) learns "is this asking how ASH feels, or for the machine's status?" from. The rules above catch the phrasings
# somebody listed (and teach these as they do); "emotional status" was read as "status" and answered with the CPU, because one regex
# had the word "status" in it and the other did not know "emotional".
SEEDS = {
    "feelings": ["how are you", "how are you feeling", "how do you feel", "are you ok", "what's your mood", "how's your mood",
                 "what's your emotional state", "are you happy", "are you sad today", "how's it going with you", "you doing alright",
                 "how's your day been", "what mood are you in", "are you in a good mood", "how are you doing today",
                 "what is your emotional status", "do you have feelings today", "are you feeling okay", "are you tired",
                 "are you bored", "what's your vibe today", "how's your state of mind", "give me your emotional telemetry",
                 "show me your emotional readout", "read me your mood report", "give me an emotional status report", "what are your emotional levels"],
    "status": ["what's your status", "system status", "how's the cpu", "what's your cpu usage", "memory usage", "how much ram is free",
               "battery level", "give me a status report", "what's using my memory", "how much disk space is left", "what's your load",
               "show system stats", "how is the system doing", "check system resources", "how's the computer doing"],
    "none": ["what time is it", "play some music", "who is ada lovelace", "tell me a joke", "thanks", "what's the weather", "yes",
             "open spotify", "what do you remember", "what's my name", "set a timer", "turn the volume up", "how do i fix this error",
             "what's the plan for today", "how long will it take", "how many files are in this folder", "what's new", "how do you spell necessary",
             "how are you going to do that", "how's my schedule today", "how am i feeling", "do you think i'm stressed", "how do i feel about this",
             "what's my emotional state", "how is the weather", "how's the traffic", "what are you doing", "who are you", "what can you do",
             "remember that i parked on level three", "remind me to call mom", "i'm tired", "i'm bored", "tell me about yourself"],
}

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
            why = (f"you've said nice things to me {_times(t.get('praise'))} today" if t.get("praise") else
                   f"you've thanked me {_times(t.get('thanks'))} today" if t.get("thanks") else
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
