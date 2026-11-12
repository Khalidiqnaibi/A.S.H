"""
src/daemon/commands.py

Saying a command any reasonable way -- and getting it done fast.

Every rule in the daemon (media keys, apps, the camera, reminders, privacy...) has its own exact grammar, and a line
that missed it by a word went to the language model: 20-70 seconds for "Ash, pause the music", "can you pause the
music for me", or "pasue the musci". This module sits in front of all of those grammars, uses no model and no
network, and takes well under a millisecond:

    clean(text)          the wake word ("Ash,", ", Ash", "hey Ash"), politeness and filler ("can you", "please",
                         "for me", "real quick", "um"), stray punctuation, "whats" -> "what's"
    Speller.correct()    a word that is not a word at all ("pasue", "musci", "wether", "calender") becomes the
                         command word it is one or two letters from. A real English word is never changed (the
                         encoder's own word list says what is one), so "what should I cook" stays "cook".
    Catalog.nearest()    the known command a line is closest to, by its words: "pause the music that's playing"
                         -> "pause the music". Sure enough: it is run. Close but not sure: "Did you mean ...?"

How the daemon uses it (AmbientRuntime._loose_command): only when no rule took the line as it was said, and only for
a line meant for ASH -- overheard speech is never stretched into a command. A line with a negation ("don't pause the
music") or one asking about a command ("why did you pause the music?") is never run as one.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------------------------------- cleaning
_WAKE = r"(?:(?:hey|hi|ok(?:ay)?|yo)\s+)?(?:ash|a\.s\.h\.?|ass)"
_LEAD_WAKE = re.compile(rf"^\s*{_WAKE}\b[\s,.:;!-]*", re.I)
_TRAIL_WAKE = re.compile(rf"[\s,.:;!-]+{_WAKE}\s*[.!?]*\s*$", re.I)
_LEAD = re.compile(
    r"^\s*(?:please|pls|plz|kindly|just|so|okay\s+so|ok\s+so|alright|all\s+right|um+|uh+|er+|hmm+|oh|hey|yo|and|also|now|then|"
    r"(?:can|could|would|will|wo?n'?t)\s+(?:you|u)(?:\s+(?:please|pls|just|quickly))?|"
    r"i\s+(?:want|need|would\s+like|'d\s+like)\s+(?:you\s+)?to|i'd\s+like\s+(?:you\s+)?to|"
    r"go\s+ahead\s+and|do\s+me\s+a\s+favou?r\s+and|quickly|quick)\b[\s,]*", re.I)
_TRAIL = re.compile(
    r"[\s,]+(?:please|pls|plz|for\s+me|thanks|thank\s+you|thx|ty|now|right\s+now|real\s+quick|quickly|asap|"
    r"if\s+you\s+(?:can|could|would|don'?t\s+mind)|will\s+you|would\s+you|could\s+you|ok(?:ay)?|yeah|alright|"
    r"for\s+a\s+(?:sec|second|bit|moment))\s*[.!?]*\s*$", re.I)
_SPELLED = {"whats": "what's", "wheres": "where's", "whos": "who's", "hows": "how's", "dont": "don't",
            "cant": "can't", "wont": "won't", "im": "i'm", "u": "you", "ur": "your", "pls": "please",
            "plz": "please", "vol": "volume", "tmrw": "tomorrow", "tmr": "tomorrow", "2moro": "tomorrow",
            "pic": "picture", "msg": "message", "calc": "calculator", "rn": "right now", "teh": "the", "hte": "the",
            "wat": "what", "wht": "what", "ths": "this", "tis": "this", "adn": "and", "yuo": "you", "thats": "that's",
            "whta": "what", "tiem": "time"}
_WORD = re.compile(r"[A-Za-z][A-Za-z']*|\d+(?:[.:]\d+)?|\S")


def _fix_spelled(text: str) -> str:
    return re.sub(r"\b[A-Za-z0-9]+\b", lambda m: _SPELLED.get(m.group(0).lower(), m.group(0)), text)


def clean(text: str) -> str:
    """The command inside a line: no wake word, no politeness or filler, no trailing punctuation."""
    t = _fix_spelled((text or "").strip())
    for _ in range(6):                                   # "ok so can you please ..." peels in a few passes
        before = t
        t = _LEAD_WAKE.sub("", t)
        t = _TRAIL_WAKE.sub("", t)
        t = _LEAD.sub("", t)
        t = _TRAIL.sub("", t)
        t = t.strip(" ,;:-")
        if t == before:
            break
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*[.!?]+$", "", t)
    return t


# ---------------------------------------------------------------------------------------------------- spelling
def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Damerau-Levenshtein (a swap of two letters counts once), stopping early past `cap`."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2: List[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


def english_words() -> FrozenSet[str]:
    """Real English words, from the local encoder's own word list (no download, no network). Empty if it is not
    there -- then nothing is ever 'corrected' except what the command words themselves are close to."""
    roots = [os.environ.get("HF_HUB_CACHE", ""), os.path.join(os.environ.get("HF_HOME", ""), "hub"),
             os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
    for root in [r for r in roots if r and os.path.isdir(r)]:
        for path in glob.glob(os.path.join(root, "models--sentence-transformers--*", "snapshots", "*", "vocab.txt")):
            try:
                with open(path, encoding="utf-8") as fh:
                    words = {w.strip() for w in fh if w.strip().isalpha() and not w.startswith("##")}
                if len(words) > 5000:
                    return frozenset(words)
            except OSError:
                continue
    return frozenset()


class Speller:
    """Corrects only words that are not words: never a real English word, never a short one, never a number."""

    def __init__(self, vocab: Iterable[str], known: Iterable[str] = ()):
        self.vocab: Set[str] = {w.lower() for w in vocab if w and w.isalpha()}
        self.known: Set[str] = {w.lower() for w in known} | self.vocab
        self._by_len: Dict[int, List[str]] = {}
        for w in self.vocab:
            self._by_len.setdefault(len(w), []).append(w)

    def word(self, w: str) -> str:
        low = w.lower()
        if len(low) < 4 or not low.isalpha() or low in self.known:
            return w
        cap = 1 if len(low) <= 7 else 2         # "readme" is not two letters from "resume" for a reason
        best: List[Tuple[int, str]] = []
        for n in range(len(low) - cap, len(low) + cap + 1):
            for v in self._by_len.get(n, ()):
                d = edit_distance(low, v, cap)
                if d <= cap:
                    best.append((d, v))
        if not best:
            return w
        best.sort()
        if len(best) > 1 and best[0][0] == best[1][0]:
            return w                                     # two equally close command words: do not guess
        return best[0][1]

    def correct(self, text: str) -> str:
        return re.sub(r"[A-Za-z]+", lambda m: self.word(m.group(0)), text or "")


# ---------------------------------------------------------------------------------------------------- the catalog
# The commands a line can be matched to, by the rule that runs them ("rule"), or by the brain's fast tools ("brain").
# Each "rule" phrase is checked by tests/tst_commands.py to be taken by the daemon's rules exactly as written.
CATALOG: Sequence[Tuple[str, str]] = (
    # media
    ("pause the music", "rule"), ("play the music", "rule"), ("resume the music", "rule"), ("next song", "rule"),
    ("previous song", "rule"), ("skip this song", "rule"), ("turn the volume up", "rule"),
    ("turn the volume down", "rule"), ("mute the sound", "rule"), ("take a photo", "rule"),
    # ASH itself
    ("turn off the mic", "rule"), ("turn on the mic", "rule"), ("turn on the camera", "rule"),
    ("turn off the camera", "rule"), ("turn on screen capture", "rule"), ("turn off screen capture", "rule"),
    ("go to privacy mode", "rule"), ("privacy off", "rule"), ("go to sleep", "rule"), ("wake up", "rule"),
    ("what's your status", "rule"), ("what faces can you make", "rule"),
    # time-keeping
    ("what are my reminders", "rule"), ("focus for 25 minutes", "rule"), ("start focus mode", "rule"),
    ("stop focus mode", "rule"),
    # the camera
    # (a bare "what do you see?" is the SCREEN's question, not the camera's -- deliberately not here)
    ("who is this", "rule"), ("look around", "rule"), ("what am I holding", "rule"),
    # about what it said
    ("why did you say that", "rule"),
    # the brain's own tools
    ("what time is it", "brain"), ("what's the date today", "brain"), ("how much memory am I using", "brain"),
)

# Small words that do not change which command it is. Not "this" or "it": "who is this" is not "who is Messi".
_STOP = frozenset("the a an my your some that these those me to of for on in at please".split())
_NEGATION = re.compile(r"\b(?:don'?t|do\s+not|doesn'?t|never|not|no\s+need|stop\s+asking|without)\b", re.I)
_ASKS_ABOUT = re.compile(r"^\s*(?:why|how\s+come|did\s+you|have\s+you|when\s+did|who\s+(?:told|said)|what\s+did|"
                         r"what\s+happens|what\s+does|can\s+you\s+explain|tell\s+me\s+(?:about|why))\b", re.I)
_QUESTION_PHRASE = re.compile(r"^(?:what|who|why|how|when|where|is|are|do|does|can)\b", re.I)


_EXPAND = {"what's": "what is", "who's": "who is", "where's": "where is", "how's": "how is", "it's": "it is",
           "that's": "that is", "i'm": "i am", "what're": "what are"}


def content(text: str) -> List[str]:
    low = " ".join(_EXPAND.get(w, w) for w in (text or "").lower().split())
    return [w for w in re.findall(r"[a-z0-9']+", low) if w not in _STOP]


def _has(word: str, words: List[str]) -> bool:
    """A number in a command stands for any number: "focus for 25 minutes" is also "focus for 45 minutes"."""
    return any(w.isdigit() for w in words) if word.isdigit() else word in words


def _with_numbers(phrase: str, text: str) -> str:
    said = re.findall(r"\d+", text or "")
    if not said:
        return phrase
    it = iter(said)
    return re.sub(r"\d+", lambda m: next(it, m.group(0)), phrase)


@dataclass
class Near:
    phrase: str
    kind: str          # "rule" | "brain"
    sure: bool         # True: run it; False: ask "Did you mean ...?"
    score: float


class Catalog:
    def __init__(self, entries: Sequence[Tuple[str, str]] = CATALOG):
        self.entries = [(p, k, content(p)) for p, k in entries]

    def words(self) -> Set[str]:
        out: Set[str] = set()
        for p, _k, _c in self.entries:
            out.update(re.findall(r"[a-z]+", p.lower()))
        return out

    def nearest(self, text: str) -> Optional[Near]:
        """The command `text` is closest to, or None. Never for a negation, nor for a question about a command."""
        if not text or _NEGATION.search(text):
            return None
        words = content(text)
        if not words or len(words) > 10:
            return None
        asks_about = bool(_ASKS_ABOUT.search(text))
        scored: List[Tuple[float, int, str, str]] = []
        for phrase, kind, want in self.entries:
            if not want:
                continue
            if asks_about and not _QUESTION_PHRASE.search(phrase):
                continue                                 # "why did you pause the music" is not "pause the music"
            have = [w for w in want if _has(w, words)]
            cover = len(have) / len(want)
            if cover < 0.6 or (cover < 1.0 and _QUESTION_PHRASE.search(phrase)):
                continue                                 # a question is matched whole, or not at all
            if not _has(want[0], words):
                continue                                 # the command's own word must be there: "I love this song" is not "skip this song"
            # in the same order, too: "the volume turn up" is still it, but "up the volume turn" scores lower
            pos = [next(i for i, x in enumerate(words) if (x.isdigit() if w.isdigit() else x == w)) for w in have]
            ordered = 1.0 if pos == sorted(pos) else 0.85
            extra = len([w for w in words if not (w in want or (w.isdigit() and any(x.isdigit() for x in want)))])
            score = cover * ordered - 0.06 * extra
            scored.append((score, -extra, phrase, kind))
        if not scored:
            return None
        scored.sort(reverse=True)
        best = scored[0]
        tie = len(scored) > 1 and abs(scored[1][0] - best[0]) < 0.05 and scored[1][3] == best[3]
        full = all(_has(w, words) for w in content(best[2]))
        extra = -best[1]
        sure = full and extra <= 2 and not tie and best[0] >= 0.85
        if not sure and (tie or best[0] < 0.5):
            return None                                  # too vague to even ask about
        return Near(_with_numbers(best[2], text), best[3], sure, round(best[0], 3))


# ---------------------------------------------------------------------------------------------------- together
class Commands:
    """clean -> correct -> nearest, for one line. The daemon tries each in turn, only while nothing has taken it."""

    YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|do\s+it|go\s+ahead|please(?:\s+do)?|correct|right|"
                     r"that'?s\s+(?:it|right|what\s+i\s+(?:meant|said)))\b[\s.!]*$", re.I)
    NO = re.compile(r"^\s*(?:no|nope|nah|never\s*mind|cancel|forget\s+it|not\s+that|wrong)\b[\s.!]*$", re.I)

    def __init__(self, catalog: Optional[Catalog] = None, known: Optional[Iterable[str]] = None,
                 extra_vocab: Iterable[str] = ()):
        self.catalog = catalog or Catalog()
        vocab = self.catalog.words() | set(EXTRA_VOCAB) | {w.lower() for w in extra_vocab}
        self.speller = Speller(vocab, english_words() if known is None else known)

    def variants(self, text: str) -> List[Tuple[str, str]]:
        """[(text, how)] to try after the line as said: 'cleaned', then 'corrected' -- each only if it differs."""
        out: List[Tuple[str, str]] = []
        c = clean(text)
        if c and c.lower() != (text or "").strip().lower():
            out.append((c, "cleaned"))
        f = self.speller.correct(c or text)
        if f and f.lower() not in {(text or "").strip().lower(), (c or "").lower()}:
            out.append((f, "corrected"))
        return out

    def nearest(self, text: str) -> Optional[Near]:
        return self.catalog.nearest(self.speller.correct(clean(text)))

    def spelled(self, text: str) -> str:
        """The line as said, only with its misspellings put right ("whats the tiem" -> "what's the time") -- no
        filler taken off, nothing reworded. What the brain is given when nothing else took the line."""
        return self.speller.correct(_fix_spelled(text or ""))


# Words the rules know beyond the catalog's phrases (so "notpad", "calender", "wether", "remnd" are fixed too).
EXTRA_VOCAB = (
    "open launch start close quit exit stop pause resume play skip next previous louder quieter softer volume mute unmute "
    "music song track video playback microphone camera screen capture privacy private sleep wake status memory cpu "
    "remind reminder reminders timer alarm focus minutes hours tomorrow tonight morning evening weather forecast rain "
    "calendar meetings schedule email emails inbox briefing brief notepad calculator chrome firefox edge word excel "
    "outlook spotify discord teams explorer settings photo picture selfie lights light lamp brightness forget remember "
    "holding look around who what time date today convert miles kilometers").split()
