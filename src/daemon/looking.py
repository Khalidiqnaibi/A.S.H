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

def _L(n: str = "") -> str:
    return r"(?P<label%s>[a-z][a-z0-9' \-]{0,38}?)" % n


def _P(n: str = "") -> str:
    return r"(?P<photo%s>\s*(?:,\s*)?(?:and\s+)?(?:also\s+)?save\s+(?:the\s+|a\s+)?(?:photo|picture))?" % n


_RX: List[Tuple[str, "re.Pattern"]] = [(k, re.compile(p, re.I)) for k, p in (
    # Teaching: by showing. "this is X" alone is ordinary talk, so every form says "remember" or "learn".
    ("teach_me", _A + r"(?:remember|learn)\s+my\s+face" + _P() + _Z),
    ("teach_person", _A + r"(?:(?:remember|learn)\s+this\s+(?:person|face)\s+(?:as|is)\s+" + _L() + _P()
                          + r"|(?:remember|learn)\s+" + _L("2") + r"'s\s+face" + _P("2")
                          + r"|this\s+(?:person\s+)?is\s+" + _L("3") + r",?\s*(?:please\s+)?(?:remember|learn)\s+(?:his|her|their)\s+face" + _P("3") + r")" + _Z),
    ("teach_thing", _A + r"(?:(?:remember|learn)\s+this\s+(?:thing\s+|object\s+)?(?:as|is)\s+" + _L() + _P()
                         + r"|this\s+is\s+" + _L("2") + r",?\s*(?:please\s+)?(?:remember|learn)\s+(?:it|that|this)" + _P("2") + r")" + _Z),
    # Open questions, for a vision model -- only if switched on (vision.open_questions + vision.vlm_model).
    ("read", _A + r"(?:what\s+does\s+(?:this|that)\s+(?:(?:label|sign|box|page)\s+)?(?:say|read)|what(?:'s|\s+is)\s+written\s+(?:on|here)(?:\s+(?:this|that|it))?"
                  r"|read\s+(?:this|that|the)\s+(?:label|sign|text|box|page|packaging)(?:\s+out)?|read\s+(?:this|that)\s+out)" + _Z),
    ("describe", _A + r"(?:describe\s+(?:what\s+you\s+see|this|the\s+scene|the\s+room)(?:\s+in\s+detail)?"
                      r"|tell\s+me\s+(?:in\s+detail\s+)?what\s+you\s+see\s+(?:through|on|with)\s+the\s+camera)" + _Z),
    ("ask", r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:look\s+and\s+tell\s+me|use\s+the\s+camera\s+to\s+(?:tell\s+me|check|see))\s+(?P<q>[^.?!]{4,120})\s*[.?!]*\s*$"),
    # Managing what it knows. "forget X" only counts when X is something it was taught (Looking.wants).
    ("list", _A + r"(?:what\s+(?:faces|people|things)\s+do\s+you\s+(?:know|recogni[sz]e)|who\s+(?:do\s+you\s+know\s+by\s+sight|have\s+you\s+(?:learned|been\s+(?:taught|shown)))"
                  r"|what\s+(?:have\s+you\s+(?:saved|learned\s+to\s+recogni[sz]e)|have\s+i\s+(?:shown|taught)\s+you)|what\s+do\s+you\s+(?:remember\s+seeing|recogni[sz]e)"
                  r"|list\s+(?:the\s+)?(?:faces|people|things)\s+you\s+know)" + _Z),
    ("export", _A + r"export\s+(?:the\s+)?(?:faces|visual\s+memory|what\s+you(?:'ve|\s+have)\s+(?:seen|saved|learned\s+to\s+recogni[sz]e))" + _Z),
    ("save_photo", _A + r"(?:save|keep)\s+(?:that|the|this)\s+(?:last\s+)?(?:photo|picture)" + _Z),
    ("forget_kind", _A + r"forget\s+(?:all\s+)?(?:the\s+)?(?:(?P<kind>faces|people|things)(?:\s+you\s+know)?|everything\s+you(?:'ve|\s+have)\s+(?:seen|learned\s+by\s+looking))" + _Z),
    ("forget", _A + r"forget\s+(?:(?P<label>[a-z][a-z0-9' \-]{0,38}?)'s\s+face|(?:the\s+)?(?P<label2>my\s+[a-z][a-z0-9' \-]{0,30}?)|(?:the\s+)?(?P<label3>[a-z][a-z0-9' \-]{0,38}?))" + _Z),
    # Recognising. "is this X" and "where is my X" are everyday sentences, so they only count when X is something
    # it was taught (Looking.wants) -- the pattern alone is not enough.
    ("who", _A + r"(?:who(?:'s|\s+is)\s+(?:this|that|there|here|in\s+front\s+of\s+(?:you|the\s+camera|the\s+webcam)|at\s+the\s+(?:door|camera))"
                 r"|who\s+(?:can|do)\s+you\s+see|do\s+you\s+(?:know|recogni[sz]e)\s+(?:me|this\s+(?:person|face)|who\s+this\s+is))" + _Z),
    ("is_this", _A + r"is\s+(?:this|that)\s+(?:(?P<label>my\s+[a-z][a-z0-9' \-]{0,30}?)|(?P<label2>(?-i:[A-Z][a-z]{1,19}(?:\s[A-Z][a-z]{1,19})?)))" + _Z),
    ("where", _A + r"(?:where(?:'s|\s+is|\s+are)\s+(?:my\s+)?(?P<label>[a-z][a-z0-9' \-]{1,30}?)|where\s+did\s+i\s+(?:leave|put)\s+(?:my\s+)?(?P<label2>[a-z][a-z0-9' \-]{1,30}?))"
                  r"(?:\s+right\s+now)?" + _Z),
    ("holding",_A + r"(?:what\s+(?:am\s+i|are\s+we)\s+holding|what(?:'s|\s+is)\s+(?:this|that)\s+(?:in\s+my\s+hand|i'?m\s+holding)"
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
            g = {k: v for k, v in m.groupdict().items() if v}
            label = next((g[k] for k in ("label", "label2", "label3") if k in g), "")
            out: Dict[str, str] = {}
            if label:
                out["label"] = re.sub(r"\s+", " ", label).strip(" -'")
            if any(k.startswith("photo") for k in g):
                out["photo"] = "1"
            if "kind" in g:
                out["kind"] = g["kind"].lower()
            if "q" in g:
                out["q"] = re.sub(r"\s+", " ", g["q"]).strip()
            return kind, out
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
    # Open questions ("what does this label say?"): a LOCAL multimodal Ollama model, switched on and named by you.
    open_questions: bool = False
    vlm_model: str = ""
    vlm_url: str = ""                     # empty: the language model's own Ollama address
    # Who is here, from the camera sensor's own frames every few seconds (presence.py). Off until switched on.
    presence: bool = False
    presence_interval_s: float = 8.0
    away_after_s: float = 90.0            # not seen for this long: gone
    stranger_hold_s: float = 120.0        # after an unknown face, private things are not read aloud for this long
    hold_when_stranger: bool = True
    announce_known: bool = False          # say "Omar just came in" -- off: it only shows on the console and answers "who's here?"
    announce_strangers: bool = False      # say "someone I don't know came into view"

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
          clock: Callable[[], float], memory_path: Optional[str] = None, user: Callable[[], str] = lambda: "",
          ollama_url: str = "") -> "Looking":
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
    vlm = io.get("vlm")
    if vlm is None and lc.open_questions and lc.vlm_model:
        from src.vision.vlm import OllamaVision
        vlm = OllamaVision(lc.vlm_model, lc.vlm_url or ollama_url or "http://127.0.0.1:11434/api/chat")
    return Looking(eye, faces, objects, io.get("embedder") or ThingEmbedder(), memory, lc, clock=clock, user=user, vlm=vlm)


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
                 clock: Callable[[], float] = time.time, user: Callable[[], str] = lambda: "", vlm=None):
        self.user = user
        self.vlm = vlm
        self.eye = eye
        self.faces = faces
        self.objects = objects
        self.embedder = embedder
        self.memory = memory
        self.cfg = cfg or LookingConfig()
        self.clock = clock
        from src.vision.presence import Presence
        self.presence = Presence(self.cfg.away_after_s, self.cfg.stranger_hold_s)
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
    def wants(self, text: str) -> bool:
        """A request for the camera -- and for "is this X" / "where is my X", only when X is something it was taught."""
        p = parse(text)
        if p is None or not self.cfg.enabled:
            return False
        if p[0] in ("is_this", "where", "forget"):
            return self.memory is not None and self.memory.get(p[1].get("label", "")) is not None
        return True

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self.wants(text):
            return None
        kind, _d = parse(text)
        now = self.clock() if now is None else now
        if kind in ("read", "describe", "ask") and not self.can_open_questions():
            return self._no_vlm()                            # before any frame is taken
        if kind in ("list", "export", "save_photo", "forget", "forget_kind"):
            return self.manage(kind, _d, now)                # no new picture for any of these
        if kind == "who" and self.cfg.presence and self.presence.fresh(now, 3 * self.cfg.presence_interval_s):
            return self.who_from_state(now)                  # already watching: answered from that, no new capture
        frame, why = self.eye.look(now)
        if frame is None:
            return ("I'm paused for privacy, so I'm not looking." if why == "privacy"
                    else "I couldn't get a picture -- there's no camera I can open.")
        self.stats["looks"] += 1
        if kind in ("teach_me", "teach_person", "teach_thing"):
            return self.teach(kind, _d, frame, now)
        if kind in ("read", "describe", "ask"):
            return self.open_question(kind, _d, frame)
        if kind == "who":
            return self.who(frame, now)
        if kind == "is_this":
            return self.is_this(_d["label"], frame, now)
        if kind == "where":
            return self.where(_d["label"], frame, now)
        if kind == "holding":
            return self.holding(frame, now)
        return self.around(frame, now)

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

    # -- recognising -----------------------------------------------------------------------
    @staticmethod
    def _yours(label: str, kind: str = "thing") -> str:
        """How a taught label is said back: "my blue mug" -> "your blue mug"; a bare thing gets "the"; a person is the name."""
        if kind == "person":
            return label
        low = label.lower()
        if low.startswith("my "):
            return "your " + label[3:]
        return label if low.startswith(("the ", "a ", "an ", "your ")) else "the " + label

    def _people(self, frame) -> List[Tuple[Optional[str], Optional[str], Any]]:
        """For each face in view (nearest first, at most five): (who, who it might be, the face). who is None when
        it is nobody taught; 'might be' is a close second, for a hedged answer."""
        if not self.can_see_faces():
            return []
        out: List[Tuple[Optional[str], Optional[str], Any]] = []
        for f in sorted(self._faces(frame), key=lambda f: -(f.box[2] * f.box[3]))[:5]:
            vec = self.faces.embed(frame, f)
            who, near = None, None
            if vec is not None and self.memory is not None:
                m = self.memory.match(vec, kind="person", threshold=self.cfg.face_threshold, margin=self.cfg.face_margin)
                if m is not None:
                    who = m.label
                else:
                    sc = self.memory.scores(vec, "person")
                    if sc and sc[0][1] >= self.cfg.face_threshold - 0.08:
                        near = sc[0][0]
            out.append((who, near, f))
        return out

    def _thing_match(self, frame, box):
        """The taught thing that crop looks like, or None."""
        if self.memory is None or not self.memory.list("thing"):
            return None
        vec = self.embedder.embed(frame, box)
        if vec is None:
            return None
        return self.memory.match(vec, kind="thing", threshold=self.cfg.thing_threshold, margin=self.cfg.thing_margin)

    def around(self, frame, now: Optional[float] = None) -> str:
        """What is in view: people by name where they are known, taught things by name, the rest by what the detector says."""
        if not (self.can_see_things() or self.can_see_faces()):
            return self._no_models()
        now = self.clock() if now is None else now
        things = self._things(frame)
        people = self._people(frame)
        names: List[str] = []
        for who, _near, _f in people:
            if who and who not in names:
                names.append(who)
                self.memory.seen_now(who, now)
        det_people = sum(1 for t in things if t.label == "person")
        total_people = max(len(people), det_people)
        unknown = max(0, total_people - len(names))
        yours: List[str] = []
        counts: Dict[str, int] = {}
        for t in things:
            if t.label == "person":
                continue
            m = self._thing_match(frame, t.box) if len(yours) + len(counts) < 8 else None
            if m is not None:
                label = self._yours(m.label)
                if label not in yours:
                    yours.append(label)
                    self.memory.seen_now(m.label, now)
            else:
                counts[t.label] = counts.get(t.label, 0) + 1
        pieces: List[str] = []
        if names and unknown:
            pieces.append(f"{_list(names)} and " + ("someone I don't know" if unknown == 1 else f"{unknown} people I don't know"))
        elif names:
            pieces.append(_list(names))
        elif unknown:
            pieces.append(_count("person", unknown))
        pieces += yours
        pieces += [_count(k, counts[k]) for k in sorted(counts, key=lambda k: (-counts[k], k))]
        if not pieces:
            return "I don't see anything I recognise."
        sure = [t for t in things if t.score >= self.cfg.sure_conf]
        hedge = "" if (sure or people or yours) else " -- though I'm not very sure"
        extra = f", and {len(pieces) - 5} other things" if len(pieces) > 5 else ""
        return f"I see {_list(pieces[:5])}{extra}{hedge}."

    def _held(self, frame):
        """What is held up to the camera: the detection nearest the middle that is not a person or furniture."""
        things = [t for t in self._things(frame) if t.label not in _BIG]
        if not things:
            return None
        h, w = frame.shape[:2]

        def off(t):
            x, y, bw, bh = t.box
            return ((x + bw / 2 - w / 2) / w) ** 2 + ((y + bh / 2 - h / 2) / h) ** 2
        return min(things, key=lambda t: (off(t) - 0.15 * t.score))

    def holding(self, frame, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        best = self._held(frame)
        m = self._thing_match(frame, best.box if best is not None else None)
        if m is not None:                                           # what it was taught beats what a detector calls it
            self.memory.seen_now(m.label, now)
            return f"That's {self._yours(m.label)}."
        if not self.can_see_things():
            return ("I can't tell what's in your hand without an object model installed. Any YOLOv8 ONNX file works -- see the README."
                    if not self.can_see_faces() else "I can tell people apart, but I have no object model to say what you're holding.")
        if best is None:
            return "I can't make out what you're holding -- try holding it closer to the camera."
        if best.score >= self.cfg.sure_conf:
            return f"That looks like {_a(best.label)}."
        return f"That might be {_a(best.label)}, but I'm not sure."

    def who(self, frame, now: Optional[float] = None) -> str:
        """"Who's this?": the people in view, by name where they are known."""
        if not self.can_see_faces():
            return ("I can't tell faces apart yet -- the face models aren't installed. \"python ashctl.py vision install\" "
                    "fetches them, and asks first.")
        now = self.clock() if now is None else now
        people = self._people(frame)
        if not people:
            return "I don't see anyone."
        for who, _n, _f in people:
            if who:
                self.memory.seen_now(who, now)
        if len(people) == 1:
            who, near, _f = people[0]
            if who:
                return f"That's {who}."
            if near:
                return f"That might be {near}, but I'm not sure."
            return "I don't know them. If you tell me who it is -- say \"remember this person as <name>\" -- I will."
        names = [w for w, _n, _f in people if w]
        unknown = len(people) - len(names)
        if not names:
            return f"I see {len(people)} people and I don't know any of them."
        return f"I see {_list(names)}" + (f", and {'someone' if unknown == 1 else str(unknown) + ' people'} I don't know." if unknown else ".")

    def is_this(self, label: str, frame, now: Optional[float] = None) -> str:
        """"Is this Omar?" / "is this my mug?": checked against what it was taught, and honest about close calls."""
        item = self.memory.get(label)
        now = self.clock() if now is None else now
        if item.kind == "person":
            if not self.can_see_faces():
                return "I can't tell faces apart yet -- the face models aren't installed."
            people = self._people(frame)
            if not people:
                return "I can't see a face."
            who, near, _f = people[0]
            if who and who.lower() == item.label.lower():
                self.memory.seen_now(item.label, now)
                return f"Yes, that's {item.label}."
            if who:
                return f"No -- that looks like {who}."
            if near and near.lower() == item.label.lower():
                return f"It might be {item.label}, but I'm not sure."
            return f"No, that doesn't look like {item.label}."
        best = self._held(frame)
        m = self._thing_match(frame, best.box if best is not None else None)
        if m is not None and m.label.lower() == item.label.lower():
            self.memory.seen_now(item.label, now)
            return f"Yes, that's {self._yours(item.label)}."
        if m is not None:
            return f"No -- that looks like {self._yours(m.label)}."
        return f"I don't think so -- it doesn't look like {self._yours(item.label)}."

    @staticmethod
    def _when(ts: float, now: float) -> str:
        d = max(0.0, now - ts)
        ts, now = max(ts, 86400.0), max(now, 86400.0)                # a clock that has not been set must not crash the answer
        if d < 60:
            return "just now"
        if d < 3600:
            return f"{int(d // 60)} minutes ago"
        clock = time.strftime("%H:%M", time.localtime(ts))
        today = time.strftime("%Y-%m-%d", time.localtime(now))
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        yday = time.strftime("%Y-%m-%d", time.localtime(now - 86400))
        return (f"at {clock} today" if day == today else f"yesterday at {clock}" if day == yday
                else "on " + time.strftime("%d %b", time.localtime(ts)) + f" at {clock}")

    def where(self, label: str, frame, now: Optional[float] = None) -> str:
        """"Where are my keys?": in view right now, or when it was last seen. ASH has no map of the room: it can only say
        what the camera has shown it."""
        item = self.memory.get(label)
        now = self.clock() if now is None else now
        if item.kind == "person":
            if self.can_see_faces() and any(w and w.lower() == item.label.lower() for w, _n, _f in self._people(frame)):
                self.memory.seen_now(item.label, now)
                return f"{item.label} is in view right now."
        else:
            boxes = [t.box for t in self._things(frame) if t.label != "person"][:8]
            found = False
            for box in [None] + boxes:
                m = self._thing_match(frame, box)
                if m is not None and m.label.lower() == item.label.lower():
                    found = True
                    break
            if found:
                self.memory.seen_now(item.label, now)
                name = self._yours(item.label)
                return f"{name[0].upper() + name[1:]} {'are' if name.endswith('s') else 'is'} in view right now."
        name = self._yours(item.label, item.kind)
        if item.seen <= 1:
            return (f"I only know {name} from when you showed me, {self._when(item.created, now)} -- I haven't seen "
                    f"{'them' if name.endswith('s') or item.kind == 'person' else 'it'} since.")
        return (f"I last saw {name} {self._when(item.last_seen, now)}. I can only say what the camera has shown me, "
                f"not where things are in the room.")

    # -- teaching: by showing -------------------------------------------------------------
    _NOT_A_NAME = frozenset("this that it here there me you him her them everything something someone somebody".split())

    def teach(self, kind: str, d: Dict[str, str], frame, now: float) -> str:
        from src.vision.memory import norm_label
        label = (d.get("label") or "").strip()
        if kind == "teach_me":
            label = (self.user() or "").strip()
            if not label:
                return "I don't know your name yet -- say \"remember <your name>'s face\"."
        if not label or not norm_label(label) or norm_label(label) in self._NOT_A_NAME or len(label) > 40:
            return "I need a name for it -- say, for example, \"remember this as my blue mug\"."
        if self.memory is None:
            return "I have nowhere to keep that."
        say = self._teach_person(label, frame, now) if kind in ("teach_me", "teach_person") else self._teach_thing(label, frame, now)
        if d.get("photo") and self.memory.get(label) is not None and not say.startswith(("I can", "I couldn't", "I need", "That looks like")):
            path = self.eye.save(frame, self.cfg.photo_dir, label, now)
            if path:
                self.memory.get(label).photo = path
                self.memory.save()
                say += f" I saved the photo as {path.replace(chr(92), '/').rsplit('/', 1)[-1]}."
        return say

    def _teach_person(self, label: str, frame, now: float) -> str:
        from src.vision.memory import norm_label
        if not self.can_see_faces():
            return ("I can't tell faces apart yet -- the face models aren't installed. \"python ashctl.py vision install\" "
                    "fetches them, and asks first.")
        faces = sorted(self._faces(frame), key=lambda f: -(f.box[2] * f.box[3]))
        if not faces:
            return "I can't see a face. Face the camera and say it again."
        if len(faces) > 1 and faces[0].box[2] * faces[0].box[3] < 1.5 * faces[1].box[2] * faces[1].box[3]:
            return "I can see more than one face -- I need just one person in view to learn it."
        vec = self.faces.embed(frame, faces[0])
        if vec is None:
            return "I couldn't read that face -- try again with the face lit and straight on."
        m = self.memory.match(vec, kind="person", threshold=self.cfg.face_threshold, margin=0.0)
        if m is not None and norm_label(m.label) != norm_label(label):
            return f"That looks like {m.label}, who I already know. If I've got that wrong, say \"forget {m.label}\" first."
        v = self.memory.enroll(label, "person", vec)
        if v is None:
            return f"I can't keep {label} as a person: that name is already something else I know."
        if len(v.embeds) == 1:
            return (f"Okay, I'll recognise {label}. I keep numbers describing the face, not the picture, and only on this computer. "
                    f"Show me once more from another angle and I'll be surer.")
        return f"Got another look at {label} ({len(v.embeds)} now)."

    def _teach_thing(self, label: str, frame, now: float) -> str:
        from src.vision.memory import norm_label
        box, note = None, ""
        if self.can_see_things():
            cands = [t for t in self._things(frame) if t.label not in _BIG]
            if cands:
                h, w = frame.shape[:2]
                best = min(cands, key=lambda t: ((t.box[0] + t.box[2] / 2 - w / 2) / w) ** 2 + ((t.box[1] + t.box[3] / 2 - h / 2) / h) ** 2 - 0.15 * t.score)
                box, note = best.box, best.label
        vec = self.embedder.embed(frame, box)
        if vec is None:
            return "I couldn't get a clear look at it -- hold it up to the camera, in the middle."
        m = self.memory.match(vec, kind="thing", threshold=self.cfg.thing_threshold, margin=0.0)
        if m is not None and norm_label(m.label) != norm_label(label):
            return f"That looks like {m.label}, which I already know. If I've got that wrong, say \"forget {m.label}\" first."
        v = self.memory.enroll(label, "thing", vec)
        if v is None:
            return f"I can't keep {label} as a thing: that name is already a person I know."
        if len(v.embeds) == 1:
            seen = f" (it looks like {_a(note)})" if note else ""
            return f"Okay, I'll recognise {label}{seen}. Show it to me again in a different position and I'll be surer."
        return f"Got another look at {label} ({len(v.embeds)} now)."

    # -- open questions ---------------------------------------------------------------------
    def can_open_questions(self) -> bool:
        return bool(self.cfg.open_questions and self.vlm is not None and self.vlm.available())

    def _no_vlm(self) -> str:
        return ("I can name things and tell people apart, but reading a label or describing a scene needs a vision model, and that "
                "isn't set up. Set vision.open_questions and vision.vlm_model (a local Ollama model such as llava) in daemon.json.")

    def open_question(self, kind: str, d: Dict[str, str], frame) -> str:
        question = {"read": "Read out the text in the picture, exactly as written. If there is none, say so.",
                    "describe": "Describe what you see: the objects, the setting, and anything that looks written or labelled."}.get(kind) or d.get("q", "")
        text = self.vlm.ask(frame, question)
        if not text:
            return "I couldn't get an answer from the vision model."
        text = re.sub(r"\s+", " ", text)[:420].rstrip()
        return text + (" " if text.endswith((".", "!", "?")) else ". ") + "That's the vision model's reading, so check it if it matters."

    # -- managing what it knows -----------------------------------------------------------
    def manage(self, kind: str, d: Dict[str, str], now: float) -> str:
        mem = self.memory
        if kind == "save_photo":
            frame = self.eye.recent(now)
            if frame is None:
                return "I don't have a picture from the last minute. Say \"look around\" first, and then ask me to save it."
            path = self.eye.save(frame, self.cfg.photo_dir, "", now)
            self.eye.drop()
            if not path:
                return "I couldn't save the picture."
            return f"Saved as {path.replace(chr(92), '/').rsplit('/', 1)[-1]} in {self.cfg.photo_dir.replace(chr(92), '/').rstrip('/').rsplit('/', 1)[-1]}."
        if mem is None:
            return "I have nowhere to keep anything."
        if kind == "export":
            import os
            folder = os.path.join(os.environ.get("ASH_STATE_DIR", os.path.join(os.getcwd(), "state")), "exports")
            path = os.path.join(folder, time.strftime("vision-%Y%m%d-%H%M%S.json", time.localtime(now)))
            n = mem.export(path)
            return f"Saved {n} entries to {path}. It lists who and what I know and when -- not the numbers I recognise them by."
        if kind == "forget":
            item = mem.get(d["label"])
            if item is None:
                return ""
            had_photo = bool(item.photo)
            mem.forget(d["label"])
            self.presence.present.pop(item.label, None)
            them = "them" if item.kind == "person" else "it"
            say = f"Forgotten: {item.label}. I've deleted the numbers I recognised {them} by."
            if item.kind == "person":
                say += f" What you've told me about {item.label} is separate -- \"forget what I said about {item.label}\" removes that."
            if had_photo:
                say += " The photo you asked me to save is still where it was -- that file is yours to delete."
            return say
        if kind == "forget_kind":
            k = {"faces": "person", "people": "person", "things": "thing"}.get(d.get("kind", ""), None)
            photos = any(v.photo for v in mem.list(k))
            n = mem.forget(kind=k)
            if k in (None, "person"):
                self.presence.clear()
            noun = {"person": ("person", "people"), "thing": ("thing", "things"), None: ("entry", "entries")}[k]
            if not n:
                return f"I don't have any {noun[1]} to forget."
            say = f"Forgotten: {n} {noun[0] if n == 1 else noun[1]} -- the numbers I recognised them by are gone."
            if photos:
                say += " Photos you asked me to save are still where they were."
            return say
        return self.listing(now)

    def listing(self, now: float) -> str:
        mem = self.memory
        people, things = mem.list("person"), mem.list("thing")
        if not people and not things:
            return "I haven't been taught anyone or anything yet. Say \"remember this as my blue mug\" or \"remember Omar's face\" while showing me."
        bits = []
        for rows, kind, one, many in ((people, "person", "person", "people"), (things, "thing", "thing", "things")):
            if rows:
                shown = [f"{self._yours(v.label, kind)} (last seen {self._when(v.last_seen, now)})" for v in rows[:8]]
                more = f", and {len(rows) - 8} more" if len(rows) > 8 else ""
                bits.append(f"{len(rows)} {one if len(rows) == 1 else many}: {_list(shown)}{more}")
        return "I know " + "; ".join(bits) + ". I keep numbers describing them, not pictures."

    # -- who is here (presence) --------------------------------------------------------------
    def watch(self, frame, now: Optional[float] = None) -> List[Tuple[str, str]]:
        """One reading for the presence state: who is in this frame. Returns arrivals, departures, strangers. The frame is
        not kept and nothing but names and times is."""
        if not (self.cfg.enabled and self.cfg.presence) or not self.can_see_faces():
            return []
        now = self.clock() if now is None else now
        people = self._people(frame)
        for who, _near, _f in people:
            if who and self.memory is not None:
                self.memory.seen_now(who, now)
        return self.presence.update([who for who, _n, _f in people], now)

    def stranger_present(self, now: Optional[float] = None) -> bool:
        return bool(self.cfg.presence and self.cfg.hold_when_stranger and self.presence.stranger(self.clock() if now is None else now))

    def announcement(self, kind: str, who: str, user: str = "") -> str:
        """What to say aloud about a change, if the user switched announcing on; "" otherwise."""
        if kind == "stranger":
            return "Someone I don't know just came into view." if self.cfg.announce_strangers else ""
        if not self.cfg.announce_known:
            return ""
        if kind == "arrived":
            return "Welcome back." if user and who.lower() == user.lower() else f"{who} just came in."
        return ""

    def who_from_state(self, now: float) -> str:
        """"Who's here?" answered from the presence state: no new picture."""
        names = self.presence.here(now)
        stranger = self.presence.stranger(now)
        if not names and not stranger:
            return "I don't see anyone."
        if names and stranger:
            return f"I see {_list(names)}, and someone I don't know."
        if names:
            return f"{names[0]} is here." if len(names) == 1 else f"I see {_list(names)}."
        return "I see someone I don't know."

    def status(self) -> Dict[str, Any]:
        now = self.clock()
        return {"enabled": self.cfg.enabled, "faces": self.can_see_faces(), "objects": self.can_see_things(),
                "presence": bool(self.cfg.presence), "here": self.presence.here(now) if self.cfg.presence else [],
                "stranger": self.presence.stranger(now) if self.cfg.presence else False, **self.stats}
