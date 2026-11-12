"""
src/daemon/notes.py

"Remember that I parked on level 3" ... "where did I park?" -- kept and found by rule, no language model.

People are already handled by remember.py ("remember Lieth is my friend"). This is everything else a person asks to be
kept: where the car is, a birthday, the wifi password, which drawer the passports are in. Each note is the sentence as
said (turned around to be said back: "I parked" -> "you parked"), with the words that find it again.

    remember (that) I parked on level 3             kept: "you parked on level 3"
    note that the wifi password is hunter2
    keep in mind my sister's birthday is May 3rd
    don't forget the passports are in the top drawer

    where did I park?                               "You told me you parked on level 3 (this morning)."
    when is my sister's birthday?                   "You told me your sister's birthday is May 3rd."
    do you remember where I put the passports?
    what did I tell you about the wifi?             all of what was said about it, newest first
    what do you remember?                           the last few notes
    forget where I parked / forget the wifi password  (through the daemon's "forget ...")

How a question finds a note: every word that matters in the question must be in the note (parked = park = parking),
and the note must not be about somebody the question does not name -- "when is my birthday?" is not answered with
your sister's. A question that finds nothing is left alone (the brain answers it), unless it was plainly asking what
ASH was told ("what did I tell you about X?"): then it says it was told nothing about X.

Not for guests, never while a stranger is in view (the daemon holds the answer), and nothing leaves this computer.
"Remember to call mom" is a reminder, not a note: left to the reminders.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from .wondering import ROLE_CANON

_LEAD = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|would\s+you)[,\s]+)*"
                   r"(?:remember|note|make\s+a\s+note|take\s+a\s+note|keep\s+in\s+mind|don'?t\s+forget)"
                   r"(?:\s+(?:that|this))?[,:\s]+", re.I)
_NOT_A_NOTE = re.compile(r"^(?:to\s|me\s|in\s+\d|at\s+\d|tomorrow\b|tonight\b|it\b\s*[.!]?$|this\s*[.!]?$|that\s*[.!]?$|"
                         r"(?:when|where|what|who|how|why)\b)", re.I)

_ASK_LEAD = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|ok(?:ay)?|um|uh)[,\s]+)?(?:(?:do|can)\s+you\s+remember\s+|"
                       r"(?:can|could)\s+you\s+(?:tell|remind)\s+me\s+|remind\s+me\s+|tell\s+me\s+)?", re.I)
_RECALL = re.compile(r"^(?:where|when|what|which|who|how|is|are|am|was|were|did|do|does|have|has|can|could|should|will)\b",
                     re.I)
_TOLD = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:what\s+(?:did|have)\s+i\s+(?:tell|told)\s+you|"
                   r"what\s+do\s+you\s+(?:remember|know)|what\s+have\s+you\s+(?:got|noted))\s+(?:about\s+|on\s+)(?P<s>.+?)\s*\??\s*$",
                   re.I)
_ALL = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:what\s+(?:do\s+you\s+remember|have\s+i\s+(?:asked\s+you\s+to\s+remember|told\s+you\s+to\s+remember))|"
                  r"what\s+(?:notes|have\s+you\s+noted)|(?:list|read)\s+(?:my|your|the)\s+notes)\s*\??\s*$", re.I)

# Words that say nothing about which note is meant.
_FILLER = frozenset("""a an the my your our their his her its this that these those i me you we he she it they is are was were
    be been am do does did done have has had will would can could should shall may might must to of in on at by for from
    with about as and or but so if then than there here where when what which who whom whose how why please again
    remember tell told say said know put left kept keep ash also just still now ever already yet any some anything
    something everything nothing thing things one i'm im i've i'll i'd you're you've""".split())
_TURN = {"i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself", "i'm": "you're", "im": "you're",
         "i've": "you've", "i'll": "you'll", "i'd": "you'd", "am": "are", "we": "you", "us": "you", "our": "your",
         "ours": "yours", "we're": "you're", "we've": "you've"}
_PEOPLE = frozenset(ROLE_CANON) | frozenset("mom mum mother dad father wife husband son daughter kid kids boss".split())


def stem(w: str) -> str:
    w = w.lower().strip("'’.,!?;:\"()")
    if w.endswith("'s") or w.endswith("’s"):
        w = w[:-2]
    if len(w) >= 5 and re.search(r"(?:ss|sh|ch|x|z)es$", w):
        w = w[:-2]                                          # "passes" -> "pass", "boxes" -> "box"
    else:
        for suf in ("ing", "ed", "s"):                      # not a bare "es": "comes" -> "come", not "com"
            if len(w) - len(suf) >= 3 and w.endswith(suf) and not w.endswith("ss"):
                w = w[: -len(suf)]
                break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]                                          # "stopped" -> "stopp" -> "stop"
    return w


def keys(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[A-Za-z0-9'’]+", text or ""):
        s = stem(w)
        if s and w.lower() not in _FILLER and s not in _FILLER:      # "what's" -> "what": still filler
            out.append(s)
    return out


def turn_round(text: str) -> str:
    """'I parked on level 3' -> 'you parked on level 3'; 'my sister's' -> 'your sister's'."""
    def one(m: "re.Match[str]") -> str:
        w = m.group(0)
        t = _TURN.get(w.lower())
        if t is None:
            return w
        return t
    out = re.sub(r"[A-Za-z']+", one, text)
    return re.sub(r"\byou are\b(?=\s)", "you are", out)


@dataclass
class Note:
    id: int
    said: str                   # as the user said it ("I parked on level 3")
    back: str                   # as it is said back ("you parked on level 3")
    keys: List[str] = field(default_factory=list)
    at: float = 0.0


class Notes:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, keep: int = 500):
        self.path = path
        self.clock = clock
        self.keep = keep
        self._lock = threading.RLock()
        self.items: List[Note] = []
        self._next = 1
        self.stats = {"kept": 0, "found": 0, "not_found": 0, "forgotten": 0}
        self._load()

    # -- storage ------------------------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self.items = [Note(**n) for n in raw.get("notes", [])]
            self._next = max([n.id for n in self.items] + [0]) + 1
        except Exception:
            self.items = []

    def save(self):
        if not self.path:
            return
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"notes": [asdict(n) for n in self.items]}, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)

    # -- keeping --------------------------------------------------------------------------------
    @staticmethod
    def wants_note(text: str) -> Optional[str]:
        """What to keep from "remember that ...", or None (not a note: a reminder, a question, too vague)."""
        t = (text or "").strip()
        m = _LEAD.match(t)
        if not m:
            return None
        body = t[m.end():].strip(" ,.!")
        if not body or _NOT_A_NOTE.match(body) or len(body) > 200 or len(keys(body)) < 1 or len(body.split()) < 3:
            return None
        return body

    def keep_note(self, body: str, now: Optional[float] = None) -> Tuple[str, Note]:
        now = self.clock() if now is None else now
        with self._lock:
            n = Note(id=self._next, said=body, back=turn_round(body), keys=sorted(set(keys(body))), at=now)
            self._next += 1
            self.items.append(n)
            del self.items[:-self.keep]
            self.save()
        self.stats["kept"] += 1
        return f"Noted: {n.back}.", n

    def drop(self, nid: int) -> bool:
        with self._lock:
            before = len(self.items)
            self.items = [n for n in self.items if n.id != nid]
            if len(self.items) != before:
                self.save()
                return True
        return False

    # -- finding --------------------------------------------------------------------------------
    def find(self, question: str) -> List[Note]:
        """Notes that answer `question`, newest first: every word that matters in it is in the note, and the note is
        not about a person the question does not name."""
        want = set(keys(question))
        if not want:
            return []
        # One word more than the note has is allowed only for a question about something the user did ("where did
        # I park the car?"), not for any question that happens to share a word ("what's the weather password?").
        lenient = bool(re.search(r"\b(?:did|have|had)\s+i\b|\bi\s+(?:put|left|kept|parked|told)\b", question, re.I))
        people = {stem(p) for p in _PEOPLE}
        q_people = want & people
        full, near = [], []
        for n in self.items:
            have = set(n.keys)
            if (have & people) - q_people:
                continue                                    # "when is my birthday?" is not your sister's birthday
            missing = want - have
            if not missing:
                full.append(n)
            elif lenient and len(want) >= 2 and len(missing) == 1 and not (missing & people):
                near.append(n)                              # "where did I park the car?" -- "you parked on level 3"
        return sorted(full or near, key=lambda n: -n.at)

    def _when(self, at: float, now: float) -> str:
        if at <= 86400:
            return ""
        d, today = datetime.fromtimestamp(at), datetime.fromtimestamp(now)
        days = (today.date() - d.date()).days
        if days == 0:
            part = "this morning" if d.hour < 12 else ("this afternoon" if d.hour < 18 else "this evening")
            return part
        if days == 1:
            return "yesterday"
        if days < 7:
            return "on " + d.strftime("%A")
        return "on " + d.strftime("%d %B").lstrip("0")

    def recall(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """The answer to a question about what was noted, or None to leave the question to the brain."""
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if _ALL.match(t):
            if not self.items:
                return "You haven't asked me to remember anything yet."
            last = sorted(self.items, key=lambda n: -n.at)[:5]
            return "You told me: " + "; ".join(n.back for n in last) + "."
        told = _TOLD.match(t)
        if told:
            subject = told.group("s")
            found = self.find(subject)
            if not found:
                self.stats["not_found"] += 1
                return f"You haven't told me anything about {turn_round(subject.strip(' ?.'))}."
            self.stats["found"] += 1
            return "You told me: " + "; ".join(n.back for n in found[:4]) + "."
        q = _ASK_LEAD.sub("", t, count=1).strip()
        if not _RECALL.match(q) or len(q) > 120 or re.match(r"how\s+(?:do|does|can|should|to|would)\b", q, re.I):
            return None                                      # "how do I park a car?" asks how, not what was noted
        found = self.find(q)
        if not found:
            return None
        self.stats["found"] += 1
        n = found[0]
        when = self._when(n.at, now)
        return f"You told me {n.back}" + (f" ({when})" if when else "") + "."

    def forget(self, subject: str) -> List[str]:
        """Forget the notes about `subject` (all of its words in the note). Returns what was dropped, as said back."""
        want = set(keys(subject))
        if not want:
            return []
        with self._lock:
            gone = [n for n in self.items if want <= set(n.keys)]
            if gone:
                self.items = [n for n in self.items if n not in gone]
                self.save()
        self.stats["forgotten"] += len(gone)
        return [n.back for n in gone]

    def status(self) -> Dict[str, int]:
        return {"notes": len(self.items), **self.stats}
