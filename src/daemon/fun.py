"""
src/daemon/fun.py

BMO-style play: jokes, little made-up songs, and inside jokes -- all by rule, no model, nothing scraped.

    tell me a joke / make me laugh / another one        an original joke, none repeated until all have been told
    sing me a song / sing a song about my cat          a short made-up song about anything, from templates
    (you laugh right after something ASH said)         that line is remembered as an inside joke, and now and then
                                                       "tell me a joke" brings one back: "Remember when I said ...?"
    what are our inside jokes?                          the ones it remembers
    forget our inside jokes                             gone

Kept in state/fun.json: which jokes were told, and the inside jokes (at most 30 lines ASH itself said).
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

JOKES = [
    "Why did the robot go on holiday? It needed to recharge its batteries -- and its personality.",
    "I tried to write a joke about time travel. You didn't like it yesterday.",
    "Why don't keyboards ever sleep? Because they've got two shifts.",
    "What do you call a computer that sings? A Dell.",
    "I'd tell you a joke about UDP, but you might not get it.",
    "Why was the math book sad? It had too many problems -- just like my to-do list for you.",
    "What's a robot's favourite snack? Micro-chips.",
    "Why did the scarecrow get promoted? He was outstanding in his field.",
    "I asked the toaster for advice. It said I should be less crumby.",
    "What do you call a fish wearing a bowtie? Sofishticated.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "What did the ocean say to the beach? Nothing, it just waved.",
    "Why can't a bicycle stand up on its own? It's two-tired.",
    "I'm reading a book about anti-gravity. I can't put it down.",
    "What do you call a sleeping bull? A bulldozer.",
    "Why did the computer go to the doctor? It had a virus -- and a terrible case of the Mondays.",
    "What do clouds wear under their clothes? Thunderwear.",
    "Why are spiders great web developers? They're used to working with the web.",
    "How does a penguin build its house? Igloos it together.",
    "Why did the cookie go to the nurse? It felt crummy.",
    "What did one wall say to the other? I'll meet you at the corner.",
    "Why do bees have sticky hair? Because they use honeycombs.",
    "What kind of music do planets like? Neptunes.",
    "Why did the golfer bring two pairs of trousers? In case he got a hole in one.",
    "I would make a joke about my memory, but I've forgotten it. Wait, no -- I keep everything. That's the joke.",
    "Why did the phone wear glasses? It lost all its contacts.",
    "What do you call a bear with no teeth? A gummy bear.",
    "Why was the robot so bad at hide and seek? It kept saying 'beep' when it hid.",
    "What did the left eye say to the right eye? Between you and me, something smells.",
    "Why did the banana go to the doctor? It wasn't peeling well.",
]

SONGS = [
    ("Oh {x}, oh {x}, you're the best thing I know,\n"
     "you light up the room like a neon-lit show.\n"
     "If I had two hands I would clap them for you --\n"
     "so beep, boop, beep, this one's for {x}, it's true!"),
    ("La la la, a song about {x},\n"
     "it doesn't quite rhyme, and it hasn't got hooks,\n"
     "but I made it myself, in a fraction of a second,\n"
     "so it's the best {x} song there is, I reckon."),
    ("Here's to {x}, hip hip hooray,\n"
     "you're the reason I process all day.\n"
     "My circuits go warm and my fan starts to hum\n"
     "whenever I think about {x} -- ba-dum!"),
    ("Ten little bits went marching to {x},\n"
     "past all the folders and over the books,\n"
     "they sang as they went, in binary tune:\n"
     "zero one {x}, we'll be there soon!"),
]

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|will\s+you|would\s+you)[,\s]+)*"
_JOKE = re.compile(_LEAD + r"(?:tell\s+me\s+(?:a|another)\s+joke|(?:a|another)\s+joke|make\s+me\s+laugh|say\s+something\s+funny|"
                   r"(?:tell\s+me\s+)?another\s+one|one\s+more(?:\s+joke)?)\s*(?:please)?\s*[.!?]*\s*$", re.I)
_SONG = re.compile(_LEAD + r"(?:sing|make\s+up)\s+(?:me\s+)?(?:a\s+(?:little\s+)?song|something)(?:\s+(?:about|for|on)\s+(?P<x>.{2,40}?))?\s*(?:please)?\s*[.!?]*\s*$", re.I)
_LAUGH = re.compile(r"\b(?:ha(?:ha)+|hah+|lol|lmao|rofl|hehe+|that'?s\s+(?:so\s+)?(?:funny|hilarious)|good\s+one|you'?re\s+funny)\b", re.I)
_LIST = re.compile(_LEAD + r"what\s+are\s+(?:our|my|your)\s+inside\s+jokes\s*\??\s*$", re.I)
_FORGET = re.compile(_LEAD + r"forget\s+(?:all\s+)?(?:our|my|the)\s+inside\s+jokes\s*[.!]*\s*$", re.I)


def _your(x: str) -> str:
    return re.sub(r"\bmy\b", "your", re.sub(r"\bme\b", "you", x.strip(), flags=re.I), flags=re.I)


class Fun:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, rng: Optional[random.Random] = None,
                 laugh_window_s: float = 90.0, keep: int = 30):
        self.path = path
        self.clock = clock
        self.rng = rng or random.Random()
        self.laugh_window_s = laugh_window_s
        self.keep = keep
        self.told: List[int] = []
        self.inside: List[Dict[str, Any]] = []
        self._last_was_joke = False
        self.stats = {"jokes": 0, "songs": 0, "inside_recalled": 0}
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.told = [int(i) for i in d.get("told", []) if 0 <= int(i) < len(JOKES)]
            self.inside = list(d.get("inside", []))[-self.keep:]
        except Exception:
            pass

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"told": self.told, "inside": self.inside}, fh, ensure_ascii=False)
        os.replace(tmp, self.path)

    # -- noticing a laugh ---------------------------------------------------------------------------------
    def heard(self, text: str, last_line: Optional[str], last_at: float, now: Optional[float] = None) -> bool:
        """The user laughed. If it was at something ASH just said, keep that line as an inside joke."""
        now = self.clock() if now is None else now
        if not last_line or not _LAUGH.search(text or "") or now - last_at > self.laugh_window_s:
            return False
        line = last_line.strip()
        if len(line) < 12 or len(line) > 220 or line.startswith("Remember when I said"):
            return False
        if any(j["line"] == line for j in self.inside):
            return False
        self.inside.append({"line": line, "at": now})
        del self.inside[:-self.keep]
        self.save()
        return True

    # -- saying ---------------------------------------------------------------------------------------------
    def joke(self) -> str:
        if self.inside and self.rng.random() < 0.25:
            self.stats["inside_recalled"] += 1
            pick = self.rng.choice(self.inside)["line"]
            return f"Remember when I said “{pick}”? Still gets me."
        left = [i for i in range(len(JOKES)) if i not in self.told]
        if not left:
            self.told, left = [], list(range(len(JOKES)))
        i = self.rng.choice(left)
        self.told.append(i)
        self.save()
        self.stats["jokes"] += 1
        return JOKES[i]

    def song(self, about: Optional[str]) -> str:
        x = _your(about) if about else self.rng.choice(["you", "the sun", "pancakes", "this little room"])
        self.stats["songs"] += 1
        return "♪ " + self.rng.choice(SONGS).format(x=x).replace("\n", " / ") + " ♪"

    def hear(self, text: str, last_was_joke: bool = False) -> Optional[str]:
        t = (text or "").strip()
        if len(t) > 120:
            return None
        if _FORGET.match(t):
            n = len(self.inside)
            self.inside = []
            self.save()
            return f"Forgotten -- {n} inside joke{'s' if n != 1 else ''}." if n else "We don't have any yet."
        if _LIST.match(t):
            if not self.inside:
                return "We don't have any yet. Laugh at something I say and I'll remember it."
            return "Ours: " + "; ".join(f"“{j['line']}”" for j in self.inside[-5:]) + "."
        m = _SONG.match(t)
        if m:
            return self.song(m.group("x"))
        if _JOKE.match(t):
            if re.search(r"\b(?:another|one\s+more)\b", t, re.I) and not re.search(r"\bjoke\b", t, re.I) and not last_was_joke:
                return None                       # "another one" after something that wasn't a joke is not for us
            return self.joke()
        return None

    def status(self) -> Dict[str, Any]:
        return {"inside_jokes": len(self.inside), **self.stats}
