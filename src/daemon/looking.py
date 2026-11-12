"""
src/daemon/looking.py

ASH with its eyes open -- on request, in rules, no language model.

    "look around" / "what's in front of the camera?" / "can you see me?"   what is in view
    "what am I holding?" / "look at this"                                  the thing nearest the middle

(bare "what do you see?" is the SCREEN's, and stays so.)

It takes one frame (eye.py), runs the detectors (backends.py) and answers from the labels. Nothing it sees is stored:
not the frame, not the labels. Never while privacy mode is on, and never for a guest's voice. When the models are not
installed it says so instead of guessing.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.looking")

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:so\s+)?(?:please\s+|can you\s+|could you\s+)?"
_Z = r"(?:\s+please)?\s*[.!?]*\s*$"

_RX: List[Tuple[str, "re.Pattern"]] = [(k, re.compile(p, re.I)) for k, p in (
    ("holding", _A + r"(?:what\s+(?:am\s+i|are\s+we)\s+holding|what(?:'s|\s+is)\s+(?:this|that)\s+(?:in\s+my\s+hand|i'?m\s+holding)"
                     r"|what(?:'s|\s+is)\s+in\s+my\s+hand|look\s+at\s+this(?:\s+(?:thing|object))?"
                     r"|what\s+is\s+this\s+(?:thing|object)(?:\s+i'?m\s+holding)?)" + _Z),
    ("see", _A + r"(?:look\s+around|what(?:'s|\s+is)\s+in\s+front\s+of\s+(?:you|the\s+camera|the\s+webcam)"
                 r"|what\s+(?:can|do)\s+you\s+see\s+(?:through|with|on|from)\s+(?:the\s+|your\s+)?(?:camera|webcam)"
                 r"|what(?:'s|\s+is)\s+(?:on|in\s+view\s+of)\s+(?:the\s+|your\s+)?(?:camera|webcam)"
                 r"|(?:use|check)\s+(?:the\s+|your\s+)?(?:camera|webcam)(?:\s+and\s+(?:tell\s+me\s+)?what\s+you\s+see)?"
                 r"|can\s+you\s+see\s+(?:me|anything))" + _Z),
)]


def parse(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """(kind, details) if this is a request to use the camera, else None."""
    t = (text or "").strip()
    if not t or len(t) > 140:
        return None
    for kind, rx in _RX:
        m = rx.match(t)
        if m:
            return kind, {k: v for k, v in m.groupdict().items() if v}
    return None


@dataclass
class LookingConfig:
    enabled: bool = False
    face_threshold: float = 0.40          # SFace cosine similarity for "the same person"
    face_margin: float = 0.05             # and a lead over the next person
    thing_threshold: float = 0.88         # the light embedder: same colour and shape
    thing_margin: float = 0.04
    object_conf: float = 0.35
    sure_conf: float = 0.60               # above this a label is stated; between this and object_conf it is hedged
    photo_dir: str = "~/Pictures/ASH"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LookingConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                try:
                    setattr(c, k, type(getattr(c, k))(v))
                except (TypeError, ValueError):
                    pass
        return c


def build(cfg: Optional[Dict[str, Any]], io: Optional[Dict[str, Any]], sensor: Callable[[], Any], paused: Callable[[], bool],
          clock: Callable[[], float], memory_path: Optional[str] = None) -> "Looking":
    """The production wiring: the real eye and models from the config -- or whatever `io` supplies instead (tests pass
    fakes for any of eye, faces, objects, embedder, memory). Models that are not on disk are simply unavailable."""
    import os
    from src.vision import models as vmodels
    from src.vision.backends import ONNXObjects, OpenCVFaces, ThingEmbedder
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    io = io or {}
    c = dict(cfg or {})
    folder = os.path.expanduser(str(c.get("models_dir") or vmodels.default_dir()))
    lc = LookingConfig.from_dict({**c, "enabled": True})
    eye = io.get("eye") or Eye(sensor=sensor, paused=paused, index=int(c.get("camera_index", 0)), clock=clock)
    faces = io["faces"] if "faces" in io else OpenCVFaces(vmodels.path_of("face_detector", folder), vmodels.path_of("face_recognizer", folder))
    objects = io["objects"] if "objects" in io else ONNXObjects(os.path.expanduser(str(c.get("objects_path") or os.path.join(folder, "objects.onnx"))),
                                                                  conf=lc.object_conf)
    memory = io["memory"] if "memory" in io else VisualMemory(memory_path, clock=clock)
    return Looking(eye, faces, objects, io.get("embedder") or ThingEmbedder(), memory, lc, clock=clock)


_PLURAL = {"person": "people", "sheep": "sheep"}
_BIG = frozenset({"person", "chair", "couch", "bed", "dining table", "tv", "laptop", "refrigerator", "oven", "sink", "toilet", "bench"})


def _a(label: str) -> str:
    return ("an " if label[:1] in "aeiou" else "a ") + label


def _count(label: str, n: int) -> str:
    return _a(label) if n == 1 else f"{n} {_PLURAL.get(label, label + 's')}"


def _list(xs: List[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


class Looking:
    """The camera, asked. `eye`, `faces`, `objects` and `embedder` are injected so tests never touch hardware."""

    def __init__(self, eye, faces, objects, embedder, memory=None, cfg: Optional[LookingConfig] = None,
                 clock: Callable[[], float] = time.time):
        self.eye = eye
        self.faces = faces
        self.objects = objects
        self.embedder = embedder
        self.memory = memory
        self.cfg = cfg or LookingConfig()
        self.clock = clock
        self.stats = {"looks": 0}

    # -- what can be done at all ----------------------------------------------------
    def can_see_things(self) -> bool:
        return bool(self.objects is not None and self.objects.available())

    def can_see_faces(self) -> bool:
        return bool(self.faces is not None and self.faces.available())

    def _no_models(self) -> str:
        return ("I got a picture, but I have no model installed to read it with. The face models install with "
                "\"python ashctl.py vision install\" (it asks first); an object model is any YOLOv8 ONNX file -- see the README.")

    # -- answering -------------------------------------------------------------------
    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        p = parse(text)
        if p is None or not self.cfg.enabled:
            return None
        kind, _d = p
        now = self.clock() if now is None else now
        frame, why = self.eye.look(now)
        if frame is None:
            return ("I'm paused for privacy, so I'm not looking." if why == "privacy"
                    else "I couldn't get a picture -- there's no camera I can open.")
        self.stats["looks"] += 1
        if kind == "holding":
            return self.holding(frame)
        return self.around(frame)

    def _things(self, frame) -> List[Any]:
        try:
            return [t for t in self.objects.detect(frame) if t.score >= self.cfg.object_conf] if self.can_see_things() else []
        except Exception:
            logger.exception("Object detection failed")
            return []

    def _faces(self, frame) -> List[Any]:
        try:
            return self.faces.detect(frame) if self.can_see_faces() else []
        except Exception:
            logger.exception("Face detection failed")
            return []

    def around(self, frame) -> str:
        """What is in view, from the labels."""
        if not (self.can_see_things() or self.can_see_faces()):
            return self._no_models()
        things, faces = self._things(frame), self._faces(frame)
        counts: Dict[str, int] = {}
        for t in things:
            counts[t.label] = counts.get(t.label, 0) + 1
        if faces:
            counts["person"] = max(counts.get("person", 0), len(faces))
        if not counts:
            return "I don't see anything I recognise."
        order = sorted(counts, key=lambda k: (-counts[k], k))
        sure = [t for t in things if t.score >= self.cfg.sure_conf]
        shown = [_count(k, counts[k]) for k in order[:5]]
        hedge = "" if (sure or faces) else " -- though I'm not very sure"
        extra = f", and {len(order) - 5} other things" if len(order) > 5 else ""
        return f"I see {_list(shown)}{extra}{hedge}."

    def holding(self, frame) -> str:
        """The thing nearest the middle that is not a person or furniture: what is held up to the camera."""
        if not self.can_see_things():
            return ("I can't tell what's in your hand without an object model installed. Any YOLOv8 ONNX file works -- see the README."
                    if not self.can_see_faces() else "I can tell people apart, but I have no object model to say what you're holding.")
        things = [t for t in self._things(frame) if t.label not in _BIG]
        if not things:
            return "I can't make out what you're holding -- try holding it closer to the camera."
        h, w = frame.shape[:2]

        def off(t):
            x, y, bw, bh = t.box
            return ((x + bw / 2 - w / 2) / w) ** 2 + ((y + bh / 2 - h / 2) / h) ** 2
        best = min(things, key=lambda t: (off(t) - 0.15 * t.score))
        if best.score >= self.cfg.sure_conf:
            return f"That looks like {_a(best.label)}."
        return f"That might be {_a(best.label)}, but I'm not sure."

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "faces": self.can_see_faces(), "objects": self.can_see_things(), **self.stats}
