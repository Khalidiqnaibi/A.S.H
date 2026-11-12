"""
src/vision/memory.py

What ASH has been shown, and who and what it has seen: labels and embeddings, not pictures.

A person or a thing is taught by showing ("this is Omar", "this is my blue mug"). What is kept is a label, a handful
of embeddings (numbers describing how it looked), when it was taught, and when it was last seen -- never the frame it
came from, unless the user said "and save the photo" (then only a path is kept here; the file is the user's).

Matching is deliberately cautious. A match needs a similarity above a threshold AND a clear lead over the next best
different label; anything less is "not sure", never a guess. Faces are biometric data, so everything stays on this
machine, in one file the user can read, export and delete from.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ash.vision.memory")

KINDS = ("person", "thing")


def norm_label(label: str) -> str:
    """Case, articles and spacing do not make a different label: "My Blue Mug" is "blue mug"."""
    t = re.sub(r"[^a-z0-9' ]+", " ", (label or "").lower())
    t = re.sub(r"^(?:the|a|an|my|our|his|her|their)\s+", "", t.strip())
    return re.sub(r"\s+", " ", t).strip()


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else a


@dataclass
class Visual:
    key: str                      # the normalised label
    label: str                    # as it was said
    kind: str                     # person | thing
    embeds: List[List[float]] = field(default_factory=list)
    created: float = 0.0
    last_seen: float = 0.0
    seen: int = 0
    photo: str = ""               # a path the user asked to keep, or ""
    note: str = ""


@dataclass
class Match:
    label: str
    kind: str
    score: float                  # cosine similarity to the closest embedding of that label
    runner_up: float              # the best score of any other label (0 if there is none)


class VisualMemory:
    def __init__(self, path: Optional[str] = None, max_embeds: int = 12, clock: Callable[[], float] = time.time):
        self.path = path
        self.max_embeds = int(max_embeds)
        self.clock = clock
        self._lock = threading.RLock()
        self.items: Dict[str, Visual] = {}              # key -> Visual
        self._load()

    # -- keeping ----------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for d in raw.get("items", []):
                v = Visual(key=str(d["key"]), label=str(d["label"]), kind=str(d["kind"]),
                           embeds=[[float(x) for x in e] for e in d.get("embeds", [])],
                           created=float(d.get("created", 0)), last_seen=float(d.get("last_seen", 0)),
                           seen=int(d.get("seen", 0)), photo=str(d.get("photo", "")), note=str(d.get("note", "")))
                self.items[v.key] = v
        except Exception:
            logger.exception("Visual memory unreadable; starting empty")
            self.items = {}

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            rows = [dict(key=v.key, label=v.label, kind=v.kind, embeds=[[round(x, 5) for x in e] for e in v.embeds],
                         created=v.created, last_seen=v.last_seen, seen=v.seen, photo=v.photo, note=v.note)
                    for v in self.items.values()]
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "items": rows}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("Could not save visual memory")

    # -- teaching ---------------------------------------------------------------
    def enroll(self, label: str, kind: str, vec, photo: str = "") -> Optional[Visual]:
        """Add one more look at `label`. A new label starts it; an existing one gets another embedding (the oldest
        goes when there are too many, so it follows a haircut or a new mug lid). None if the label or kind is bad."""
        key = norm_label(label)
        if not key or kind not in KINDS or vec is None:
            return None
        v_ = _unit(vec)
        if not np.isfinite(v_).all() or float(np.linalg.norm(v_)) < 0.5:
            return None
        now = self.clock()
        with self._lock:
            cur = self.items.get(key)
            if cur is not None and cur.kind != kind:
                return None                                  # "Omar" the person is not "Omar" the mug
            if cur is None:
                cur = self.items[key] = Visual(key=key, label=label.strip(), kind=kind, created=now, last_seen=now)
            cur.embeds.append([float(x) for x in v_])
            cur.embeds = cur.embeds[-self.max_embeds:]
            cur.last_seen = now
            cur.seen += 1
            if photo:
                cur.photo = photo
            self.save()
            return cur

    # -- recognising ---------------------------------------------------------------
    def scores(self, vec, kind: Optional[str] = None) -> List[Tuple[str, float]]:
        """(label, best cosine similarity), highest first."""
        v_ = _unit(vec)
        out: List[Tuple[str, float]] = []
        with self._lock:
            for it in self.items.values():
                if kind and it.kind != kind:
                    continue
                best = -1.0
                for e in it.embeds:
                    if len(e) == v_.size:
                        best = max(best, float(np.dot(v_, np.asarray(e, dtype=np.float32))))
                if best > -1.0:
                    out.append((it.label, best))
        out.sort(key=lambda r: -r[1])
        return out

    def match(self, vec, kind: Optional[str] = None, threshold: float = 0.6, margin: float = 0.05) -> Optional[Match]:
        """The label this looks like, or None. Needs `threshold`, and a lead of `margin` over the next label."""
        sc = self.scores(vec, kind)
        if not sc or sc[0][1] < threshold:
            return None
        runner = sc[1][1] if len(sc) > 1 else 0.0
        if sc[0][1] - runner < margin:
            return None                                       # two labels look alike: not sure is the honest answer
        with self._lock:
            kind_ = self.items[norm_label(sc[0][0])].kind
        return Match(label=sc[0][0], kind=kind_, score=sc[0][1], runner_up=max(0.0, runner))

    def seen_now(self, label: str, now: Optional[float] = None):
        with self._lock:
            it = self.items.get(norm_label(label))
            if it is not None:
                it.last_seen = self.clock() if now is None else now
                it.seen += 1
                self.save()

    # -- asking --------------------------------------------------------------------
    def get(self, label: str) -> Optional[Visual]:
        with self._lock:
            return self.items.get(norm_label(label))

    def list(self, kind: Optional[str] = None) -> List[Visual]:
        with self._lock:
            rows = [v for v in self.items.values() if not kind or v.kind == kind]
        rows.sort(key=lambda v: (-v.last_seen, v.label))
        return rows

    # -- forgetting -----------------------------------------------------------------
    def forget(self, label: Optional[str] = None, kind: Optional[str] = None) -> int:
        """Delete one label, or every one of a kind, or everything. Returns how many went. A kept photo is the
        user's file and is left where it is -- only the pointer goes."""
        with self._lock:
            if label:
                gone = 1 if self.items.pop(norm_label(label), None) is not None else 0
            else:
                keys = [k for k, v in self.items.items() if not kind or v.kind == kind]
                for k in keys:
                    del self.items[k]
                gone = len(keys)
            if gone:
                self.save()
        return gone

    # -- taking it with you -------------------------------------------------------
    def export(self, path: str) -> int:
        """A readable copy without the embeddings: who and what is known, when taught and last seen."""
        rows = [dict(label=v.label, kind=v.kind, taught=time.strftime("%Y-%m-%d %H:%M", time.localtime(v.created)),
                     last_seen=time.strftime("%Y-%m-%d %H:%M", time.localtime(v.last_seen)), times_seen=v.seen,
                     looks_kept=len(v.embeds), photo=v.photo) for v in self.list()]
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"note": "Labels and counts only. The numbers ASH matches with are not included, and no pictures are kept "
                               "unless you asked for one.", "items": rows}, fh, indent=1)
        return len(rows)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"people": sum(1 for v in self.items.values() if v.kind == "person"),
                    "things": sum(1 for v in self.items.values() if v.kind == "thing")}
