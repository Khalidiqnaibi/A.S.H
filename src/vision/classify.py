"""
src/vision/classify.py

An object detector that needs no download: the ImageNet classifier the brain already uses (google/vit-base-patch16-224).

ImageNet-1k has about a thousand everyday things -- watches, clocks, mugs, bottles, phones, keyboards, scissors -- but it
is a CLASSIFIER, not a detector: it says what a whole picture is of, not where. So this one looks at the picture the way
a person squints at a table: the middle of it (where a held thing is), and nine overlapping tiles. Each crop is
classified; each confident-enough answer becomes a Thing(label, score, box) with the crop's box, once per label.

    * the score is the model's own probability for that class -- honest and usually lower than a detector's: a clear
      shot of a watch is perhaps 0.6-0.9, a murky one 0.2. `min_conf` (0.15) is the floor below which nothing is said;
      `sure_conf` (0.50) is where "it's a watch" replaces "I think it's a watch". The detector-style bars in LookingConfig
      (object_conf 0.35 / sure_conf 0.60) are for YOLO and do not apply here (Looking asks the backend for its own).
    * ImageNet has no "person" class. A reading that is clothing or something worn is reported as "person", discounted.
    * on demand: ten crops in one batch, about a second on a CPU. `detect_light` (the middle only) is what background
      watching uses.
    * the model is loaded lazily, only from the local Hugging Face cache (local_files_only=True): nothing is ever
      downloaded by looking at something. When it is not there, `available()` is False and `why_not()` says so.
    * the YOLO ONNX file, when there is one, is preferred: FallbackObjects tries it first.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .backends import ObjectBackend, Thing, as_bgr
from .scene import is_worn, plain_label

logger = logging.getLogger("ash.vision.classify")

MODEL_NAME = "google/vit-base-patch16-224"

Classifier = Callable[[List[np.ndarray]], List[List[Tuple[str, float]]]]     # BGR crops -> per crop [(raw label, prob)] best first


def hf_cache_dirs() -> List[str]:
    out = []
    for env in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(env):
            out.append(os.environ[env])
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    out.append(os.path.join(home, "hub"))
    return out


def cached_model_present(name: str = MODEL_NAME) -> bool:
    """Whether the model's files are in the local Hugging Face cache (no network is touched)."""
    folder = "models--" + name.replace("/", "--")
    for base in hf_cache_dirs():
        if glob.glob(os.path.join(base, folder, "snapshots", "*", "config.json")):
            return True
    return False


class ViTClassifier:
    """The real thing: loaded on first use from the local cache. Raises if it cannot be loaded."""
    _lock = threading.Lock()
    _loaded: dict = {}

    def __init__(self, name: str = MODEL_NAME):
        self.name = name

    def _load(self):
        with self._lock:
            got = self._loaded.get(self.name)
            if got is None:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                proc = AutoImageProcessor.from_pretrained(self.name, local_files_only=True)
                model = AutoModelForImageClassification.from_pretrained(self.name, local_files_only=True)
                model.eval()
                got = self._loaded[self.name] = (proc, model)
            return got

    def __call__(self, crops: List[np.ndarray]) -> List[List[Tuple[str, float]]]:
        import torch
        from PIL import Image
        proc, model = self._load()
        imgs = [Image.fromarray(np.ascontiguousarray(c[:, :, ::-1])) for c in crops]       # BGR -> RGB
        with torch.inference_mode():
            logits = model(**proc(images=imgs, return_tensors="pt")).logits
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=3, dim=-1)
        names = model.config.id2label
        return [[(names[int(i)], float(p)) for p, i in zip(top.values[r], top.indices[r])] for r in range(len(crops))]


def crops_of(bgr: np.ndarray, tiles: bool = True) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], str]]:
    """The crops to classify: the middle (the same box ThingEmbedder uses, so a thing taught by its box is found by it)
    first, then -- if `tiles` -- a 3x3 grid of half-size tiles overlapping by half. [(pixels, (x, y, w, h), "centre"|"tile")]."""
    h, w = bgr.shape[:2]
    cx, cy, cw, ch = int(w * 0.2), int(h * 0.15), int(w * 0.6), int(h * 0.7)
    out = [(bgr[cy:cy + ch, cx:cx + cw], (cx, cy, cw, ch), "centre")]
    if tiles:
        tw, th = w // 2, h // 2
        for r in range(3):
            for c in range(3):
                x, y = c * (w - tw) // 2, r * (h - th) // 2
                out.append((bgr[y:y + th, x:x + tw], (x, y, tw, th), "tile"))
    return [o for o in out if o[0].size and min(o[0].shape[:2]) >= 16]


class ClassifierObjects(ObjectBackend):
    """Things from an ImageNet classifier over crops. `classify` is injectable (tests pass a scripted one)."""

    min_conf = 0.30
    sure_conf = 0.55
    tile_conf = 0.45

    def __init__(self, classify: Optional[Classifier] = None, name: str = MODEL_NAME, min_conf: float = 0.30,
                 sure_conf: float = 0.55, probe_s: float = 60.0, clock: Callable[[], float] = time.time):
        self._classify = classify
        self.name = name
        self.min_conf, self.sure_conf = float(min_conf), float(sure_conf)
        self.probe_s = float(probe_s)
        self.clock = clock
        self._probe_at = -1e18
        self._probe_ok = False
        self._broken = ""
        self.calls = 0
        self.last_crops = 0

    # -- can it work at all --------------------------------------------------------
    def available(self) -> bool:
        if self._classify is not None:
            return True
        if self._broken:
            return False
        now = self.clock()
        if now - self._probe_at >= self.probe_s:
            self._probe_at = now
            try:
                import importlib.util
                self._probe_ok = (importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None
                                  and cached_model_present(self.name))
            except Exception:
                self._probe_ok = False
        return self._probe_ok

    def why_not(self) -> str:
        """In words, why there is no ImageNet classifier -- "" when there is."""
        if self.available():
            return ""
        if self._broken:
            return f"the ImageNet classifier failed to load ({self._broken})"
        try:
            import importlib.util
            if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
                return "transformers and torch are not installed, so the ImageNet classifier cannot run"
        except Exception:
            pass
        return f"{self.name} is not in the local Hugging Face cache (it is never downloaded by looking at something)"

    # -- looking -------------------------------------------------------------------------
    def _run(self, bgr: np.ndarray, tiles: bool) -> List[Thing]:
        crops = crops_of(bgr, tiles)
        self.last_crops = len(crops)
        if not crops:
            return []
        if self._classify is None:
            self._classify = ViTClassifier(self.name)
        t0 = time.time()
        answers = self._classify([c[0] for c in crops])
        logger.debug("classified %d crops in %.0f ms", len(crops), (time.time() - t0) * 1000)
        best: dict = {}
        for (_pix, box, src), top in zip(crops, answers):
            for rank, (raw, p) in enumerate(top[:2]):
                if p < self.min_conf or (rank == 1 and src != "centre"):
                    continue                                   # a second guess only for the middle, where it matters
                if src != "centre" and p < self.tile_conf:
                    continue                                   # the edges of a room: a tile must be sure, or it is clutter (harmonica, ski mask...)
                person = is_worn(raw)
                label = "person" if person else plain_label(raw)
                score = float(p) * (0.8 if person else 1.0)
                if not label or score < self.min_conf:
                    continue
                cur = best.get(label)
                if cur is None or score > cur.score or (src == "centre" and score >= cur.score - 0.05):
                    best[label] = Thing(label=label, score=score, box=tuple(int(v) for v in box), src=src)
        return sorted(best.values(), key=lambda t: -t.score)

    def detect(self, img: Any) -> List[Thing]:
        return self._guarded(img, True)

    def detect_light(self, img: Any) -> List[Thing]:
        """The middle of the frame only: one crop, for watching in the background."""
        return self._guarded(img, False)

    def _guarded(self, img: Any, tiles: bool) -> List[Thing]:
        bgr = as_bgr(img)
        if bgr is None or not self.available():
            return []
        try:
            self.calls += 1
            out = self._run(bgr, tiles)
            self.last_error = None
            return out
        except Exception as e:
            logger.exception("ImageNet classification failed")
            self.last_error = "object detection failed"
            if self._classify is None or isinstance(self._classify, ViTClassifier):
                self._broken, self._classify = f"{type(e).__name__}", None       # do not retry a model that cannot load, every time
            return []


class FallbackObjects(ObjectBackend):
    """The YOLO ONNX detector when its file is there, else the ImageNet classifier."""

    def __init__(self, primary: ObjectBackend, secondary: ObjectBackend):
        self.primary, self.secondary = primary, secondary

    @property
    def _use(self) -> ObjectBackend:
        return self.primary if self.primary.available() else self.secondary

    @property
    def kind(self) -> str:
        return "yolo" if self.primary.available() else ("imagenet" if self.secondary.available() else "none")

    @property
    def min_conf(self):
        return getattr(self._use, "min_conf", None)

    @property
    def sure_conf(self):
        return getattr(self._use, "sure_conf", None)

    @property
    def last_error(self):
        return getattr(self.primary, "last_error", None) or getattr(self.secondary, "last_error", None)

    @last_error.setter
    def last_error(self, v):
        for b in (self.primary, self.secondary):
            if hasattr(b, "last_error"):
                b.last_error = v

    def available(self) -> bool:
        return self.primary.available() or self.secondary.available()

    def why_not(self) -> str:
        return "" if self.available() else (getattr(self.secondary, "why_not", lambda: "")() or "no object model is installed")

    def detect(self, img: Any) -> List[Thing]:
        return self._use.detect(img)

    def detect_light(self, img: Any) -> List[Thing]:
        b = self._use
        return getattr(b, "detect_light", b.detect)(img)
