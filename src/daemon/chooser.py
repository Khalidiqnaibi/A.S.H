"""
src/daemon/chooser.py

"What kind of thing was that?" decided by meaning, and learned from being corrected -- not by a list of phrasings somebody has to keep extending.

Several small decisions in ASH were piles of regexes: is "later" a goodbye, is "whats your emotional status" a question about feelings or
about the CPU, is "i mean from your memory" naming a place to look, is "can i have today's trivia" a request for a game. Each worked for the
phrasings its author thought of, and was wrong with the next one (today: "emotional status" read as "status", "whats todays trivia" not
read as a request at all). The classifier ASH already has does this for tools; this is the same idea for these small decisions.

A Chooser holds labelled example sentences (seeds written once, plus whatever it was told since), embeds them with the sentence encoder
ASH already loads, and labels a new sentence by its nearest examples. A "none" label of ordinary sentences keeps it from reading
everything as one of the others. It answers only when it is clear: the best label must be close enough, and clearly ahead of "none" and of
the runner-up -- otherwise the caller falls back to what it did before. `learn()` adds a sentence under the label it should have had and
keeps it, so a correction teaches it for good.

No encoder (a bare runtime, a test without one): it says nothing, and the caller's rules stand.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

NONE = "none"


@dataclass
class Verdict:
    label: Optional[str]              # None: not clear enough to say
    score: float = 0.0                # similarity to the nearest example of the best label
    margin: float = 0.0               # ahead of the next label (or "none")
    scores: Dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.label is not None


def clean(text: str) -> str:
    t = re.sub(r"[^\w\s']", " ", str(text or "").lower().replace("’", "'"))
    t = re.sub(r"^\s*(?:hey\s+)?ash[\s,]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


class Chooser:
    def __init__(self, name: str, embed: Optional[Callable[[Sequence[str]], Any]], seeds: Dict[str, Sequence[str]],
                 path: Optional[str] = None, accept: float = 0.6, margin: float = 0.04, cap: int = 150):
        self.name = name
        self.embed = embed
        self.seeds = {k: list(v) for k, v in seeds.items()}
        self.learned: Dict[str, List[str]] = {}
        self.path = path
        self.accept = accept
        self.margin = margin
        self.cap = cap
        self._lock = threading.RLock()
        self._vecs: Dict[str, np.ndarray] = {}
        self._dirty = True
        self.stats = {"asked": 0, "answered": 0, "learned": 0}
        self._load()

    @property
    def ready(self) -> bool:
        return self.embed is not None

    # -- storage -------------------------------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.learned = {str(k): [str(x) for x in v][-self.cap:] for k, v in (d.get("learned") or {}).items()}
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"learned": self.learned}, fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # -- embedding ------------------------------------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if self.embed is None or not texts:
            return None
        try:
            v = np.asarray(self.embed(list(texts)), dtype=np.float32)
        except Exception:
            return None
        if v.ndim != 2 or v.shape[0] != len(texts):
            return None
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)

    def _build(self) -> bool:
        with self._lock:
            if not self._dirty:
                return bool(self._vecs)
            vecs: Dict[str, np.ndarray] = {}
            for label in set(self.seeds) | set(self.learned):
                texts = [clean(t) for t in self.seeds.get(label, []) + self.learned.get(label, [])]
                texts = [t for t in dict.fromkeys(texts) if t]
                v = self._encode(texts)
                if v is not None:
                    vecs[label] = v
            self._vecs = vecs
            self._dirty = bool(self.embed is None) or not vecs
            return bool(vecs)

    # -- deciding -----------------------------------------------------------------------------------------
    def classify(self, text: str) -> Verdict:
        self.stats["asked"] += 1
        t = clean(text)
        if not t or not self._build():
            return Verdict(None)
        q = self._encode([t])
        if q is None:
            return Verdict(None)
        scores = {label: float((m @ q[0]).max()) for label, m in self._vecs.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top, top_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        v = Verdict(None, top_s, top_s - second_s, scores)
        if top != NONE and top_s >= self.accept and top_s - second_s >= self.margin:
            v.label = top
            self.stats["answered"] += 1
        return v

    def learn(self, text: str, label: str) -> bool:
        """`text` should have been `label` (a correction, or a sentence that was handled well): kept, so it counts from now on."""
        t = str(text or "").strip()
        c = clean(t)
        if not c or len(c) > 160:
            return False
        with self._lock:
            known = {clean(x) for x in self.seeds.get(label, []) + self.learned.get(label, [])}
            if c in known:
                return False
            for other, items in self.learned.items():            # said under another label before: it was wrong, so move it
                if other != label:
                    self.learned[other] = [x for x in items if clean(x) != c]
            self.learned.setdefault(label, []).append(t)
            del self.learned[label][:-self.cap]
            self._dirty = True
            self.stats["learned"] += 1
            self.save()
            return True

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "ready": self.ready, **self.stats,
                "learned": {k: len(v) for k, v in self.learned.items() if v}}
