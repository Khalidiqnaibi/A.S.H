"""
src/daemon/arbiter.py

A question that can be answered from different places is decided, not hard-wired to one.

"Who is Khalid?" can be the person talking (identity), somebody they told ASH about (people), something in the notes, something said in
earlier conversations (memory), an idea the concept graph has formed, or a public figure (Wikipedia). One fixed order is wrong half the
time: today "who is khalid" went to Wikipedia (a name-origin blurb) because Wikipedia was ahead of everything, and "who is immortal" got
"which one do you mean?" -- both were about the person asking. Then "i mean from your memory" went to the model, which had no idea what
"memory" referred to.

So each source says how well it fits THIS question (cheaply, with no network call), explicit words in the question can pull a source to
the top ("from your memory", "on wikipedia", "in my notes"), what was asked a moment ago counts, and what the person has corrected
before nudges the standing preference. Sources are then asked best-first, and the first that has an answer gives it. A correction
("i mean from your memory") re-asks the last question from that source. The ranking and why is kept, so ASH can say how it chose.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Words that name a source outright.
CUES: Dict[str, "re.Pattern[str]"] = {
    "memory": re.compile(r"\b(?:from|in|out\s+of)\s+(?:your|ur)\s+memor(?:y|ies)\b|\bdo\s+you\s+(?:remember|recall)\b|\bwhat\s+you\s+(?:remember|recall)\b|"
                         r"\bfrom\s+(?:what\s+)?(?:we|i)\s+(?:talked|said|spoke)\b|\bfrom\s+our\s+(?:chats?|conversations?|talks?)\b", re.I),
    "notes": re.compile(r"\b(?:in|from)\s+(?:my|the|your)\s+notes?\b|\bwhat\s+i\s+(?:told|asked)\s+you\s+to\s+remember\b|\bnotes?\s+i\s+(?:gave|made)\b", re.I),
    "people": re.compile(r"\b(?:in|from)\s+(?:my|the)\s+(?:people|contacts?|family|friends)\b|\bpeople\s+i\s+(?:told|know)\b", re.I),
    "wikipedia": re.compile(r"\bwikipedia\b|\bon\s+the\s+(?:web|internet)\b|\bgoogle\b|\blook\s+(?:it|that|him|her)\s+up\b|\bsearch\s+(?:for\s+)?(?:it|him|her|that)\b|"
                            r"\b(?:the\s+)?(?:famous|real|actual)\b|\bin\s+general\b|\bpublic(?:ly)?\b", re.I),
    "concepts": re.compile(r"\b(?:in|from)\s+(?:your|the)\s+(?:concepts?|mind|head|graph)\b|\bwhat\s+(?:do\s+)?you\s+(?:associate|link|connect)\b", re.I),
}
# A correction: "no, I mean from your memory" / "I meant wikipedia" / "from my notes".
_CORRECTION = re.compile(r"^\s*(?:(?:no|nope|nah|not\s+that|wrong|hm+|well|ok(?:ay)?|ash)[,.\s]+)*(?:i\s+(?:mean|meant)|i\s+was\s+asking|ask|try|use|check|look|"
                         r"what\s+about|how\s+about|and|or)?\s*(?:it\s+|that\s+|this\s+)?(?:from|in|on|at|with|using)\s+", re.I)

_CORRECTION_BARE = re.compile(r"^\s*(?:(?:no|nope|nah|not\s+that|wrong|hm+|well|ok(?:ay)?|ash)[,.\s]+)*(?:i\s+(?:mean|meant)|try|use|check|ask)\s+"
                              r"(?:it\s+|that\s+)?(?:the\s+|your\s+|my\s+)?(?:wikipedia|memory|notes?|people|concepts?|web)\s*[.!]*\s*$", re.I)

PERSONAL = ("me", "people", "notes", "memory")


@dataclass
class Option:
    """One place the answer could come from, and how well it fits this question."""
    source: str
    fit: float                                   # 0..1 evidence that THIS source is the right place, from cheap checks
    why: str                                     # a few words, for "why did you answer that"
    ask: Callable[[], Optional[str]]


@dataclass
class Decision:
    source: Optional[str]
    answer: Optional[str]
    ranking: List[Tuple[str, float, str]] = field(default_factory=list)
    forced: bool = False


class Arbiter:
    FLOOR = 0.15            # below this a source is not asked at all unless it was named

    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path = path
        self.clock = clock
        self.prefs: Dict[str, float] = {}               # standing preference per source, from corrections (-0.3 .. 0.3)
        self.last: Optional[Dict[str, Any]] = None      # {"subject", "text", "source", "at", "ranking"}
        self.stats = {"decided": 0, "redirected": 0}
        self._load()

    # -- storage ------------------------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.prefs = {str(k): max(-0.3, min(0.3, float(v))) for k, v in (d.get("prefs") or {}).items()}
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"prefs": self.prefs}, fh)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # -- deciding -----------------------------------------------------------------------------------
    @staticmethod
    def named(text: str) -> Optional[str]:
        """The source the words name, if any ("from your memory" -> memory)."""
        for src, rx in CUES.items():
            if rx.search(text or ""):
                return src
        return None

    def score(self, opt: Option, text: str, subject: str, named: Optional[str], now: float) -> float:
        s = opt.fit + self.prefs.get(opt.source, 0.0)
        if named == opt.source:
            s += 1.0                                               # said outright
        last = self.last
        if last and last.get("subject") == subject and now - last.get("at", 0.0) < 120.0 and last.get("source") == opt.source:
            s += 0.1                                               # the same thing, a moment later: the same place
        return s

    def decide(self, text: str, subject: str, options: List[Option], force: Optional[str] = None,
               now: Optional[float] = None) -> Decision:
        now = self.clock() if now is None else now
        named = force or self.named(text)
        if force:
            options = [o for o in options if o.source == force]          # "from my notes" means the notes, and nothing else
        personal = max([o.fit for o in options if o.source in PERSONAL] or [0.0])
        scored: List[Tuple[float, Option]] = []
        for o in options:
            sc = self.score(o, text, subject, named, now)
            if o.source == "wikipedia" and named != "wikipedia" and personal >= 0.6:
                sc *= 0.4                                          # somebody close to home: not a public figure to look up
            scored.append((sc, o))
        scored.sort(key=lambda t: -t[0])
        ranking = [(o.source, round(sc, 2), o.why) for sc, o in scored]
        dec = Decision(None, None, ranking, forced=bool(named and force))
        for sc, o in scored:
            if sc < self.FLOOR and o.source != named:
                continue
            try:
                ans = o.ask()
            except Exception:
                ans = None
            if ans:
                dec.source, dec.answer = o.source, ans
                break
        self.stats["decided"] += 1
        self.last = {"subject": subject, "text": text, "source": dec.source, "at": now, "ranking": ranking}
        return dec

    # -- being corrected ----------------------------------------------------------------------------
    def correction(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """The source a correction names ("i mean from your memory"), when it follows a decision a moment ago; else None."""
        now = self.clock() if now is None else now
        if not self.last or now - self.last.get("at", 0.0) > 180.0 or not (_CORRECTION.match(text or "") or _CORRECTION_BARE.match(text or "")):
            return None
        src = self.named(text)
        if src is None:
            m = re.search(r"\b(?:from|in|on|with|using|mean|meant|try|use|check|ask)\s+(?:my|your|the)?\s*(memory|notes?|people|concepts?|wikipedia|web)\b", text or "", re.I)
            if m:
                src = {"memory": "memory", "note": "notes", "notes": "notes", "people": "people", "concept": "concepts",
                       "concepts": "concepts", "wikipedia": "wikipedia", "web": "wikipedia"}.get(m.group(1).lower())
        return src

    def learn(self, wanted: str, was: Optional[str]) -> None:
        """The person asked for `wanted` after being given `was`: lean toward it a little, and away from it a little."""
        self.prefs[wanted] = min(0.3, self.prefs.get(wanted, 0.0) + 0.05)
        if was and was != wanted:
            self.prefs[was] = max(-0.3, self.prefs.get(was, 0.0) - 0.05)
        self.stats["redirected"] += 1
        self.save()

    def why(self) -> Optional[str]:
        """How the last answer was chosen, in words."""
        l = self.last
        if not l or not l.get("source"):
            return None
        rest = [f"{s} ({why})" for s, sc, why in l["ranking"] if s != l["source"]][:2]
        top = next((why for s, sc, why in l["ranking"] if s == l["source"]), "")
        return f"I answered “{l['subject']}” from {l['source']} ({top})" + (f"; the other places I weighed were {', '.join(rest)}." if rest else ".")

    def status(self) -> Dict[str, Any]:
        return {**self.stats, "prefs": {k: round(v, 2) for k, v in self.prefs.items() if abs(v) > 1e-6}}


# -- which questions are "who is X" -------------------------------------------------------------------------------------------
_SUBJECT = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|but|ok(?:ay)?|well)[,\s]+)*(?:please\s+)?"
                      r"(?:who(?:\s+is|\s+was|\s*['’]?s)|tell\s+me\s+(?:more\s+)?about|what\s+do\s+you\s+(?:know|remember)\s+about|"
                      r"do\s+you\s+(?:know|remember)(?:\s+who\s+is|\s+about)?|what\s+(?:can\s+you\s+tell\s+me|have\s+you\s+got)\s+(?:about|on))\s+"
                      r"(?P<s>[^?.!,;:]{1,40}?)(?:\s+in\s+(?:your|my)\s+(?:memory|notes)|\s+from\s+(?:your|my)\s+(?:memory|notes)|"
                      r"\s+on\s+wikipedia)?\s*[?.!]*\s*$", re.I)
_NOT_SUBJECT = frozenset(("this that it he she they there you your yours my mine our we us i here these those them him her who what which "
                          "someone somebody anyone anybody everyone nobody something anything").split())


def subject_of(text: str) -> Optional[str]:
    """The thing a "who is X / tell me about X / what do you know about X / do you know X" question is about, or None for a question
    that is about something else ("who is this", "who is my friend" are other rules')."""
    m = _SUBJECT.match(text or "")
    if m is None:
        return None
    s = " ".join(m.group("s").split())
    words = s.lower().split()
    if not words or len(words) > 4 or any(w in _NOT_SUBJECT for w in words) or re.search(r"\d", s):
        return None
    if len(words) == 1 and words[0] in ("the", "a", "an"):
        return None
    return s
