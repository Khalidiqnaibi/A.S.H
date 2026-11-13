"""
src/vision/scene.py

What the camera has seen, as labels -- never as pictures.

    plain_label()    ImageNet's verbose class names into plain words ("digital watch" -> "watch").
    pick_held()      which detection is the thing in someone's hand: nearest the middle, else the biggest.
    DetectionLog     the last ~300 detections in memory (time, kind, label, confidence, box, source), and the same
                     lines appended to a file with a cap. Labels, numbers and boxes only: no pixels, no names, ever.
    ChangeWatcher    "you picked up a watch", "the watch is gone", "someone came in": from a stream of readings, with
                     a few readings of agreement before anything is claimed, and a cool-down per label.

Pure Python and numpy-free; nothing here touches a camera, a model or the network.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("ash.vision.scene")

# ----------------------------------------------------------------------------------------------------- plain words
# ImageNet-1k class names are the first synonym of a comma list ("cellular telephone, cellular phone, cellphone, ...").
# These are the ones whose first synonym is not what a person says.
PLAIN: Dict[str, str] = {
    "digital watch": "watch", "analog clock": "clock", "digital clock": "clock", "wall clock": "clock",
    "stopwatch": "stopwatch", "hourglass": "hourglass", "sundial": "sundial",
    "cellular telephone": "phone", "cellphone": "phone", "mobile phone": "phone", "dial telephone": "phone",
    "pay-phone": "phone", "ipod": "music player", "hand-held computer": "handheld computer",
    "notebook": "laptop", "laptop": "laptop", "desktop computer": "computer", "monitor": "monitor", "screen": "screen",
    "television": "tv", "computer keyboard": "keyboard", "space bar": "keyboard", "mouse": "mouse",
    "remote control": "remote", "joystick": "joystick", "modem": "modem", "hard disc": "hard drive",
    "ballpoint": "pen", "fountain pen": "pen", "rubber eraser": "eraser", "pencil box": "pencil case",
    "pencil sharpener": "pencil sharpener", "coffee mug": "mug", "coffeepot": "coffee pot", "cup": "cup",
    "water bottle": "water bottle", "pop bottle": "bottle", "wine bottle": "bottle", "beer bottle": "bottle",
    "pill bottle": "pill bottle", "water jug": "jug", "whiskey jug": "jug", "thermos": "flask",
    "sunglass": "sunglasses", "sunglasses": "sunglasses", "loupe": "magnifying glass", "lens cap": "lens cap",
    "reflex camera": "camera", "polaroid camera": "camera", "binoculars": "binoculars", "tripod": "tripod",
    "book jacket": "book", "comic book": "comic book", "envelope": "envelope", "packet": "packet", "carton": "box",
    "plastic bag": "plastic bag", "paper towel": "paper towel", "toilet tissue": "toilet paper", "wallet": "wallet",
    "purse": "purse", "backpack": "backpack", "umbrella": "umbrella", "lipstick": "lipstick", "hair spray": "hair spray",
    "band aid": "plaster", "windsor tie": "tie", "bow tie": "bow tie", "teddy": "teddy bear", "toilet seat": "toilet seat",
    "padlock": "padlock", "combination lock": "lock", "safety pin": "safety pin", "screwdriver": "screwdriver",
    "hammer": "hammer", "wrench": "wrench", "scissors": "scissors", "spatula": "spatula", "ladle": "ladle",
    "frying pan": "frying pan", "soup bowl": "bowl", "mixing bowl": "bowl", "measuring cup": "measuring cup",
    "tennis ball": "tennis ball", "golf ball": "golf ball", "ping-pong ball": "ping pong ball", "soccer ball": "football",
    "table lamp": "lamp", "lampshade": "lampshade", "desk": "desk", "dining table": "table", "studio couch": "sofa",
    "park bench": "bench", "rocking chair": "chair", "folding chair": "chair", "barber chair": "chair",
    "electric fan": "fan", "space heater": "heater", "washer": "washing machine", "iron": "iron", "vacuum": "vacuum cleaner",
    "gas pump": "petrol pump", "street sign": "street sign", "traffic light": "traffic light", "flagpole": "flagpole",
}

# Things ImageNet names when what is really in front of it is a person: it has no "person" class. A reading is turned
# into "person" -- at a discount -- so "can you see me?" is not answered "I see a sweatshirt".
WORN = frozenset((
    "jersey", "suit", "sweatshirt", "cardigan", "groom", "ballplayer", "scuba diver", "wig", "bonnet", "abaya", "academic gown",
    "cloak", "pajama", "poncho", "military uniform", "lab coat", "trench coat", "fur coat", "kimono", "sarong", "stole", "mask",
    "hair slide", "face powder", "bulletproof vest", "crash helmet", "sombrero", "cowboy hat", "bearskin", "neck brace",
    "stethoscope", "maillot", "bikini", "swimming trunks", "brassiere", "mortarboard", "bathing cap", "shower cap"))


def plain_label(raw: str) -> str:
    """"digital watch" -> "watch"; "cellular telephone, cellular phone, cellphone" -> "phone"; unknown names are kept
    as their first synonym, lower-case, with hyphens and underscores read as spaces."""
    s = str(raw or "").strip().lower().replace("_", " ")
    if not s:
        return ""
    first = s.split(",")[0].strip()
    if first in PLAIN:
        return PLAIN[first]
    for syn in s.split(",")[1:]:                      # a later synonym may be the plain one ("cell", "cellphone")
        syn = syn.strip()
        if syn in PLAIN and first not in PLAIN:
            return PLAIN[syn]
    return first


def is_worn(raw: str) -> bool:
    return str(raw or "").split(",")[0].strip().lower() in WORN


# ----------------------------------------------------------------------------------------------------- the held thing
def _centre_off(box: Sequence[int], w: int, h: int) -> float:
    x, y, bw, bh = box
    return ((x + bw / 2 - w / 2) / max(w, 1)) ** 2 + ((y + bh / 2 - h / 2) / max(h, 1)) ** 2


def pick_held(things: Iterable[Any], w: int, h: int, skip: Iterable[str] = ()) -> Optional[Any]:
    """The detection most likely to be what someone is holding up to the camera: not a person, not furniture
    (`skip`), and nearest the middle -- a little help from confidence. When nothing is near the middle (about 0.35 of
    the frame away), the biggest box wins instead: a thing held close to the lens fills the frame wherever its
    centre ended up. None when there is nothing."""
    skip = set(skip)
    cands = [t for t in things if t.label != "person" and t.label not in skip]
    if not cands:
        return None
    near = [t for t in cands if _centre_off(t.box, w, h) <= 0.12]
    if near:
        return min(near, key=lambda t: _centre_off(t.box, w, h) - 0.15 * t.score)
    return max(cands, key=lambda t: t.box[2] * t.box[3] * max(t.score, 0.01))


def in_centre(box: Sequence[int], w: int, h: int, frac: float = 0.3) -> bool:
    """Whether the middle of a box is within `frac` of the frame's width/height from the middle of the frame."""
    x, y, bw, bh = box
    return abs((x + bw / 2) / max(w, 1) - 0.5) <= frac and abs((y + bh / 2) / max(h, 1) - 0.5) <= frac


# ----------------------------------------------------------------------------------------------------- the log
class DetectionLog:
    """The last `size` detections, newest last, and optionally a file of them. A detection is
    {t, kind: object|face|person|event, label, conf, box, source}. Nothing else is ever written."""

    def __init__(self, path: Optional[str] = None, size: int = 300, max_lines: int = 2000, max_age_s: float = 14 * 86400.0,
                 clock=time.time):
        self.path = path
        self.size = max(1, int(size))
        self.max_lines = max(10, int(max_lines))
        self.max_age_s = float(max_age_s)
        self.clock = clock
        self._ring: deque = deque(maxlen=self.size)
        self._lock = threading.Lock()
        self._lines: Optional[int] = None
        self.total = 0
        if path:
            self._preload()

    @staticmethod
    def clean(kind: str, label: str, conf: float, box: Optional[Sequence[int]], source: str, now: float) -> Dict[str, Any]:
        b = [int(v) for v in box][:4] if box is not None and len(list(box)) >= 4 else None
        return {"t": round(float(now), 2), "kind": str(kind)[:12], "label": str(label)[:60], "conf": round(max(0.0, min(1.0, float(conf))), 3),
                "box": b, "source": str(source)[:12]}

    def _preload(self):
        try:
            if not os.path.isfile(self.path):
                return
            with open(self.path, encoding="utf-8") as fh:
                rows = fh.readlines()
            self._lines = len(rows)
            for line in rows[-self.size:]:
                try:
                    d = json.loads(line)
                    if isinstance(d, dict) and "label" in d and "t" in d:
                        self._ring.append(self.clean(d.get("kind", "object"), d["label"], d.get("conf", 0.0), d.get("box"),
                                                     d.get("source", ""), d["t"]))
                except (ValueError, TypeError):
                    continue
        except Exception:
            logger.debug("could not read the detection log", exc_info=True)

    def add(self, kind: str, label: str, conf: float, box: Optional[Sequence[int]] = None, source: str = "asked",
            now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        rec = self.clean(kind, label, conf, box, source, now)
        with self._lock:
            self._ring.append(rec)
            self.total += 1
            if self.path:
                self._append(rec, now)
        return rec

    def _append(self, rec: Dict[str, Any], now: float):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            if self._lines is None:
                self._lines = 0
                if os.path.isfile(self.path):
                    with open(self.path, encoding="utf-8") as fh:
                        self._lines = sum(1 for _ in fh)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._lines += 1
            if self._lines > int(self.max_lines * 1.25):
                self._trim(now)
        except Exception:
            logger.debug("could not write the detection log", exc_info=True)

    def _trim(self, now: float):
        """Keep the newest `max_lines` that are not older than `max_age_s`; write it atomically."""
        with open(self.path, encoding="utf-8") as fh:
            rows = fh.readlines()
        keep: List[str] = []
        for line in rows[-self.max_lines:]:
            try:
                if now - float(json.loads(line).get("t", 0)) <= self.max_age_s:
                    keep.append(line if line.endswith("\n") else line + "\n")
            except (ValueError, TypeError):
                continue
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(keep)
        os.replace(tmp, self.path)
        self._lines = len(keep)

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Newest first."""
        with self._lock:
            return list(self._ring)[-max(0, int(n)):][::-1]

    def counts(self) -> List[Dict[str, Any]]:
        """Per label over what is in the ring: how many times and the average confidence, most often first."""
        agg: Dict[Tuple[str, str], List[float]] = {}
        with self._lock:
            for r in self._ring:
                if r["kind"] == "event":
                    continue
                agg.setdefault((r["kind"], r["label"]), []).append(r["conf"])
        rows = [{"kind": k, "label": l, "count": len(v), "avg": round(sum(v) / len(v), 3)} for (k, l), v in agg.items()]
        rows.sort(key=lambda r: (-r["count"], r["label"]))
        return rows

    def clear(self):
        with self._lock:
            self._ring.clear()
            self._lines = 0
            if self.path:
                try:
                    os.remove(self.path)
                except OSError:
                    pass

    def __len__(self) -> int:
        return len(self._ring)


# ----------------------------------------------------------------------------------------------------- change watching
class ChangeWatcher:
    """Readings in, events out: ("appeared", label, conf), ("gone", label, conf), ("person_in", "", 1), ("person_out", "", 1).

    A thing has to be in the middle of the frame for `confirm` readings in a row to have "appeared", and missing for
    `miss` in a row to be "gone" (and only things that appeared are announced gone: the desk lamp is not news). The first
    `warmup` readings are the scene as it was -- nothing in them is an event. Each label has a cool-down, so a thing
    flickering in and out is said once. People are counted, not named (names are presence.py's)."""

    def __init__(self, confirm: int = 3, miss: int = 3, warmup: int = 3, min_conf: float = 0.30, cooldown_s: float = 120.0,
                 person_cooldown_s: float = 300.0, region: float = 0.3, skip: Iterable[str] = ()):
        self.confirm, self.miss, self.warmup = max(1, int(confirm)), max(1, int(miss)), max(0, int(warmup))
        self.min_conf, self.cooldown_s, self.person_cooldown_s, self.region = float(min_conf), float(cooldown_s), float(person_cooldown_s), float(region)
        self.skip = set(skip)
        self.readings = 0
        self._seen: Dict[str, int] = {}            # label -> readings in a row it was in the middle
        self._gone: Dict[str, int] = {}            # label -> readings in a row it was not
        self.present: Dict[str, float] = {}        # label -> confidence, for things currently held
        self.announced: set = set()
        self._said_at: Dict[str, float] = {}
        self._people_on = False
        self._people_in = 0
        self._people_out = 0

    def reset(self):
        self.__init__(self.confirm, self.miss, self.warmup, self.min_conf, self.cooldown_s, self.person_cooldown_s, self.region, self.skip)

    def _cool(self, key: str, now: float, secs: float) -> bool:
        last = self._said_at.get(key)
        return last is not None and now - last < secs

    def observe(self, things: Sequence[Any], people: int, shape: Tuple[int, int], now: float) -> List[Tuple[str, str, float]]:
        h, w = shape
        self.readings += 1
        warm = self.readings <= self.warmup
        out: List[Tuple[str, str, float]] = []
        here: Dict[str, float] = {}
        for t in things:
            if t.label == "person" or t.label in self.skip or t.score < self.min_conf or not in_centre(t.box, w, h, self.region):
                continue
            here[t.label] = max(here.get(t.label, 0.0), float(t.score))
        for label, conf in here.items():
            self._gone[label] = 0
            self._seen[label] = self._seen.get(label, 0) + 1
            if label in self.present:
                self.present[label] = conf
                continue
            if warm:
                if self._seen[label] >= min(self.confirm, max(1, self.warmup)):
                    self.present[label] = conf            # part of the scene as it was: not news
                continue
            if self._seen[label] >= self.confirm:
                self.present[label] = conf
                if not self._cool("+" + label, now, self.cooldown_s):
                    self._said_at["+" + label] = now
                    self.announced.add(label)
                    out.append(("appeared", label, conf))
        for label in list(self._seen):
            if label not in here:
                self._seen[label] = 0
                if label in self.present:
                    self._gone[label] = self._gone.get(label, 0) + 1
                    if self._gone[label] >= self.miss:
                        conf = self.present.pop(label)
                        self._gone[label] = 0
                        if label in self.announced and not warm:
                            self.announced.discard(label)
                            if not self._cool("-" + label, now, self.cooldown_s):
                                self._said_at["-" + label] = now
                                out.append(("gone", label, conf))
        # people: counted
        if people > 0:
            self._people_out = 0
            self._people_in += 1
            if not self._people_on and self._people_in >= self.confirm:
                self._people_on = True
                if not warm and not self._cool("person_in", now, self.person_cooldown_s):
                    self._said_at["person_in"] = now
                    out.append(("person_in", "", 1.0))
        else:
            self._people_in = 0
            self._people_out += 1
            if self._people_on and self._people_out >= self.miss * 2:
                self._people_on = False
                if not warm and not self._cool("person_out", now, self.person_cooldown_s):
                    self._said_at["person_out"] = now
                    out.append(("person_out", "", 1.0))
        return out


def event_line(kind: str, label: str, conf: float, sure: float = 0.5) -> str:
    """What to say about a change. Hedged when the detector was not sure; a person is never named here."""
    a = ("an " if label[:1] in "aeiou" else "a ") + label if label else ""
    if kind == "appeared":
        return f"You picked up {a}." if conf >= sure else f"Is that {a} you picked up? I'm not sure."
    if kind == "gone":
        return f"The {label} is gone." if conf >= sure else f"I don't see the {label} any more."
    if kind == "person_in":
        return "Someone just came into view."
    if kind == "person_out":
        return "Nobody is in view now."
    return ""
