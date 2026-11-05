"""
src/brain/corrections.py

Learning from being corrected.

When ASH reads "what's the weather like in my code" as a weather request and
the user says "no, I meant search my files", the router made a mistake it
will make again tomorrow: nothing it learns from is told. This keeps what the
user corrected, as plain records:

    (what was said, the reading that was wrong, the reading that was meant)

and turns them into a nudge on the classifier's ranking for anything said
the same way again: the wrong reading down, the meant one up. It is a prior
like the structural ones in tools/classification.py -- bounded, inspectable,
and forgettable -- not a retrained model.

Also here: reading whether an utterance IS a correction ("no, I meant ...",
"that's not what I asked", "wrong one"), and what it says was meant.

The store is empty and in memory until `configure(path)` is called (the
daemon does), so tests and one-off runs never read the user's corrections.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("ash.brain.corrections")

_STOP = frozenset("a an the to of for in on at my me i you your it is are be please can could would "
                  "will do does just what whats what's how and or with ash hey".split())

# Said right after a reply, to say the reply was for the wrong thing.
_CORRECTION = re.compile(
    r"^\s*(?:wrong one|wrong thing|wrong|no+|nope|nah|not that|that'?s wrong|that'?s not it|"
    r"you misunderstood|you got it wrong)\b[\s,.!:;-]*(?:i\s+(?:meant|mean|wanted|asked(?:\s+for)?|said)\b"
    r"[\s,:]*)?(?P<rest>.*)$"
    r"|^\s*(?:i\s+(?:meant|mean|wanted)|what i meant (?:was|is))\b[\s,:]*(?P<rest2>.*)$"
    r"|^\s*(?:that'?s|this is|that was|it'?s)\s+not\s+what\s+i\s+(?:asked(?: for)?|meant|wanted|said)\b"
    r"[\s,.!:;-]*(?:i\s+(?:meant|wanted|asked(?:\s+for)?)\s*)?(?P<rest3>.*)$"
    r"|^\s*i\s+didn'?t\s+(?:ask|want|mean)\b(?:\s+(?:for\s+)?(?:that|you to do that|this))?[\s,.!:;-]*"
    r"(?:i\s+(?:meant|wanted|asked(?:\s+for)?)\s*)?(?P<rest4>.*)$",
    re.I)


def read_correction(text: str) -> Optional[str]:
    """None if `text` is not a correction; else what it says was meant ('' if
    it does not say)."""
    t = (text or "").strip()
    if not t or len(t) > 300:
        return None
    m = _CORRECTION.match(t)
    if not m:
        return None
    bare = re.match(r"^\s*(?:no+|nope|nah)\b", t, re.I)
    rest = next((g for g in (m.group("rest"), m.group("rest2"), m.group("rest3"), m.group("rest4")) if g),
                "").strip(" .!")
    if bare:
        # "no, I meant X" / "no, search my files" are corrections; "no",
        # "no thanks", "no, that's fine" and "no I don't think so" are answers.
        lead = re.match(r"^\s*(?:no+|nope|nah)\b[\s,.!:;-]*(?P<x>.*)$", t, re.I).group("x")
        if not re.match(r"(?:i\s+(?:meant|mean|wanted|asked|said)|not that|wrong|that'?s not)\b", lead, re.I) \
                and not re.match(r"(?:search|find|open|show|check|look|tell|set|play|read|calculate|convert|"
                                 r"remind|what|how|when|where|who)\b", lead, re.I):
            return None
    return re.sub(r"^(?:to\s+|for\s+)", "", rest, flags=re.I).strip()


def _tokens(text: str) -> List[str]:
    words = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).replace("'", "").split()
    return [w for w in words if w not in _STOP and (len(w) > 1 or w.isdigit())]


def similarity(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class Correction:
    said: str                 # what the user said, cut short
    wrong: str                # the intent that was wrong
    right: str                # the intent that was meant ("" if not known: just "not that")
    t: float
    times: int = 1


class CorrectionMemory:
    """The corrections, and the nudge they give the classifier."""

    def __init__(self, path: Optional[str] = None, max_items: int = 300,
                 nudge: float = 0.25, min_similarity: float = 0.6):
        self.path = path
        self.max_items = max_items
        self.nudge = nudge
        self.min_similarity = min_similarity
        self.items: List[Correction] = []
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.items = [Correction(**{k: v for k, v in d.items() if k in Correction.__dataclass_fields__})
                          for d in raw.get("items", [])]
        except Exception:
            logger.exception("Corrections file unreadable; starting empty")
            try:
                os.replace(self.path, self.path + ".bad")
            except Exception:
                pass
            self.items = []

    def save(self):
        if not self.path:
            return
        with self._lock:
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                tmp = f"{self.path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"version": 1, "items": [asdict(c) for c in self.items]}, fh, indent=1)
                os.replace(tmp, self.path)
            except Exception:
                logger.exception("Could not save corrections")

    def record(self, said: str, wrong: str, right: str = "", now: Optional[float] = None) -> Optional[Correction]:
        """Note that `said` was not `wrong` (and was `right`, if known)."""
        said = re.sub(r"\s+", " ", (said or "")).strip()[:160]
        wrong, right = (wrong or "").strip(), (right or "").strip()
        if not said or not wrong or wrong == right or not _tokens(said):
            return None
        now = time.time() if now is None else now
        with self._lock:
            for c in self.items:
                if c.wrong == wrong and similarity(c.said, said) >= 0.9:
                    c.times += 1
                    c.t = now
                    if right:
                        c.right = right
                    self.save()
                    return c
            c = Correction(said, wrong, right, now)
            self.items.append(c)
            if len(self.items) > self.max_items:
                self.items.sort(key=lambda x: (x.times, x.t))
                del self.items[:len(self.items) - self.max_items]
            self.save()
            return c

    def adjust(self, query: str, scored: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Re-rank [(intent, score)] by what similar sentences were corrected to."""
        scored = list(scored)
        if not self.items or not query:
            return scored
        delta: Dict[str, float] = {}
        with self._lock:
            for c in self.items:
                s = similarity(c.said, query)
                if s < self.min_similarity:
                    continue
                w = self.nudge * s * min(2.0, 1.0 + 0.25 * (c.times - 1))
                delta[c.wrong] = delta.get(c.wrong, 0.0) - w
                if c.right:
                    delta[c.right] = delta.get(c.right, 0.0) + w
        if not delta:
            return scored
        have = {i for i, _ in scored}
        out = [(i, max(0.0, sc + delta.get(i, 0.0))) for i, sc in scored]
        for i, d in delta.items():                     # a meant intent the catalog ranked nowhere
            if d > 0 and i not in have and i != "conversation":
                out.append((i, d))
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out

    def forget(self):
        with self._lock:
            self.items = []
            self.save()

    def report(self) -> str:
        with self._lock:
            lines = [f"Corrections: {len(self.items)}"]
            for c in sorted(self.items, key=lambda c: -c.t)[:30]:
                lines.append(f"  x{c.times}  {c.wrong} -> {c.right or '(not that)'}   \"{c.said[:60]}\"")
            return "\n".join(lines)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"items": len(self.items), "path": self.path}


_DEFAULT = CorrectionMemory()


def default() -> CorrectionMemory:
    return _DEFAULT


def configure(path: Optional[str]) -> CorrectionMemory:
    """Point the process-wide store at a file (the daemon does this)."""
    global _DEFAULT
    _DEFAULT = CorrectionMemory(path)
    return _DEFAULT
