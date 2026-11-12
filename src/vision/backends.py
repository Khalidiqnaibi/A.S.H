"""
src/vision/backends.py

The models, behind small interfaces so everything above them can be tested with fakes.

    Faces    OpenCV's own YuNet (find faces) and SFace (a 128-number description of one), both ONNX, CPU, a few
             hundred KB and ~37 MB. `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`.
    Objects  any YOLOv8/YOLO11-format ONNX detector (80 COCO classes by default), through onnxruntime.
    Things   a light embedder for "my blue mug": colour, layout and edge description of a crop. No model, so it
             learns from one or two looks -- and is honest about what it can do: it tells things that differ in colour
             and shape apart, not two mugs of the same colour.

Frames are PIL RGB images (what the camera sensor emits) or BGR arrays (what OpenCV gives); `as_bgr` is the one place
that settles which. None of this touches the camera or the disk.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("ash.vision.backends")

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant stop-sign parking-meter bench "
        "bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis "
        "snowboard sports-ball kite baseball-bat baseball-glove skateboard surfboard tennis-racket bottle wine-glass cup "
        "fork knife spoon bowl banana apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch "
        "potted-plant bed dining-table toilet tv laptop mouse remote keyboard cell-phone microwave oven toaster sink "
        "refrigerator book clock vase scissors teddy-bear hair-drier toothbrush").split()
COCO_LABELS = tuple(w.replace("-", " ") for w in COCO)


def as_bgr(img: Any) -> Optional[np.ndarray]:
    """A uint8 HxWx3 BGR array from a PIL image (RGB) or an ndarray (taken to be BGR already); None if it is neither."""
    if img is None:
        return None
    try:
        if hasattr(img, "convert"):                              # PIL
            return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8)[:, :, ::-1])
        a = np.asarray(img)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.ndim != 3 or a.shape[2] < 3:
            return None
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
        return np.ascontiguousarray(a[:, :, :3])
    except Exception:
        return None


@dataclass
class Face:
    box: Tuple[int, int, int, int]            # x, y, w, h
    score: float
    row: Optional[np.ndarray] = None          # the detector's raw row, which the recogniser aligns from


@dataclass
class Thing:
    label: str
    score: float
    box: Tuple[int, int, int, int]


# ======================================================================================================================
# Faces
# ======================================================================================================================
class FaceBackend:
    """detect(img) -> [Face]; embed(img, face) -> a unit vector. `available()` says whether it can work at all."""

    def available(self) -> bool:
        return False

    def detect(self, img: Any) -> List[Face]:
        return []

    def embed(self, img: Any, face: Face) -> Optional[np.ndarray]:
        return None


class OpenCVFaces(FaceBackend):
    def __init__(self, detector_path: str, recognizer_path: str, score_threshold: float = 0.85):
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.score_threshold = float(score_threshold)
        self._det = None
        self._rec = None

    def available(self) -> bool:
        try:
            import cv2
            return (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
                    and os.path.isfile(self.detector_path) and os.path.isfile(self.recognizer_path))
        except Exception:
            return False

    def _load(self):
        import cv2
        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(self.detector_path, "", (320, 320), self.score_threshold, 0.3, 10)
        if self._rec is None:
            self._rec = cv2.FaceRecognizerSF.create(self.recognizer_path, "")

    def detect(self, img: Any) -> List[Face]:
        bgr = as_bgr(img)
        if bgr is None or not self.available():
            return []
        try:
            self._load()
            h, w = bgr.shape[:2]
            self._det.setInputSize((w, h))
            _, rows = self._det.detect(bgr)
            out = []
            for r in (rows if rows is not None else []):
                x, y, fw, fh = (int(round(float(v))) for v in r[:4])
                out.append(Face(box=(x, y, fw, fh), score=float(r[-1]), row=np.asarray(r, dtype=np.float32)))
            out.sort(key=lambda f: -(f.box[2] * f.box[3]))
            return out
        except Exception:
            logger.exception("Face detection failed")
            return []

    def embed(self, img: Any, face: Face) -> Optional[np.ndarray]:
        bgr = as_bgr(img)
        if bgr is None or face.row is None or not self.available():
            return None
        try:
            self._load()
            aligned = self._rec.alignCrop(bgr, face.row)
            feat = np.asarray(self._rec.feature(aligned), dtype=np.float32).ravel()
            n = float(np.linalg.norm(feat))
            return feat / n if n > 1e-9 else None
        except Exception:
            logger.exception("Face embedding failed")
            return None


# ======================================================================================================================
# Objects
# ======================================================================================================================
def letterbox(bgr: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping the aspect ratio onto a grey square. Returns (image, ratio, (pad_x, pad_y))."""
    import cv2
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, r, (px, py)


def parse_yolov8(output: np.ndarray, ratio: float, pad: Tuple[int, int], shape: Tuple[int, int],
                 labels: Sequence[str] = COCO_LABELS, conf: float = 0.35, iou: float = 0.45) -> List[Thing]:
    """A YOLOv8/YOLO11 head, [1, 4+classes, anchors], into boxes in the original image. Pure: no model needed."""
    import cv2
    a = np.asarray(output)
    if a.ndim == 3:
        a = a[0]
    channels = 4 + len(labels)
    if a.shape[0] == channels and a.shape[1] != channels:
        a = a.T                                                    # [4+classes, anchors] -> [anchors, 4+classes]
    elif a.shape[1] != channels and a.shape[0] < a.shape[1]:
        a = a.T                                                    # a model with another class list: fewer channels than anchors
    if a.shape[1] < 5:
        return []
    boxes, scores = a[:, :4], a[:, 4:]
    cls = scores.argmax(axis=1)
    conf_ = scores[np.arange(len(scores)), cls]
    keep = conf_ >= conf
    if not keep.any():
        return []
    boxes, cls, conf_ = boxes[keep], cls[keep], conf_[keep]
    h, w = shape
    px, py = pad
    xs = (boxes[:, 0] - boxes[:, 2] / 2 - px) / ratio
    ys = (boxes[:, 1] - boxes[:, 3] / 2 - py) / ratio
    ws, hs = boxes[:, 2] / ratio, boxes[:, 3] / ratio
    rects = [[float(x), float(y), float(bw), float(bh)] for x, y, bw, bh in zip(xs, ys, ws, hs)]
    idx = cv2.dnn.NMSBoxes(rects, [float(c) for c in conf_], conf, iou)
    out: List[Thing] = []
    for i in (np.asarray(idx).ravel() if len(idx) else []):
        x, y, bw, bh = rects[int(i)]
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
        if x1 <= x0 or y1 <= y0:
            continue
        k = int(cls[int(i)])
        out.append(Thing(label=labels[k] if k < len(labels) else f"class {k}", score=float(conf_[int(i)]),
                         box=(x0, y0, x1 - x0, y1 - y0)))
    out.sort(key=lambda t: -t.score)
    return out


class ObjectBackend:
    def available(self) -> bool:
        return False

    def detect(self, img: Any) -> List[Thing]:
        return []


class ONNXObjects(ObjectBackend):
    def __init__(self, model_path: str, labels: Sequence[str] = COCO_LABELS, size: int = 640,
                 conf: float = 0.35, iou: float = 0.45):
        self.model_path = model_path
        self.labels, self.size, self.conf, self.iou = labels, int(size), float(conf), float(iou)
        self._sess = None

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return os.path.isfile(self.model_path)
        except Exception:
            return False

    def detect(self, img: Any) -> List[Thing]:
        bgr = as_bgr(img)
        if bgr is None or not self.available():
            return []
        try:
            if self._sess is None:
                import onnxruntime as ort
                self._sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            canvas, ratio, pad = letterbox(bgr, self.size)
            x = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            name = self._sess.get_inputs()[0].name
            out = self._sess.run(None, {name: np.ascontiguousarray(x)})[0]
            return parse_yolov8(out, ratio, pad, bgr.shape[:2], self.labels, self.conf, self.iou)
        except Exception:
            logger.exception("Object detection failed")
            return []


# ======================================================================================================================
# Things: a light embedder
# ======================================================================================================================
class ThingEmbedder:
    """Colour, layout and edge description of a crop, as one unit vector. No model: it needs one or two looks."""

    DIM = 12 + 4 * 3 + 4 * 4 * 3 + 2 * 2 * 8

    def embed(self, img: Any, box: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        import cv2
        bgr = as_bgr(img)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        if box is not None:
            x, y, bw, bh = box
            crop = bgr[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        else:                                                      # no detector: the middle of the frame, where it is held
            crop = bgr[int(h * 0.15):int(h * 0.85), int(w * 0.2):int(w * 0.8)]
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Hue counted by how colourful a pixel is, so a grey desk behind a blue mug does not dilute "blue"; how
        # dark/bright/washed-out it is is kept separately, unweighted, for the things that have no hue (a black mug).
        sat, val = hsv[:, :, 1].astype(np.float32) / 255.0, hsv[:, :, 2].astype(np.float32) / 255.0
        hue = np.histogram(hsv[:, :, 0], bins=12, range=(0, 180), weights=sat * val)[0].astype(np.float32)
        sv = cv2.calcHist([hsv], [1, 2], None, [4, 3], [0, 256, 0, 256]).ravel().astype(np.float32)
        hist = np.concatenate([hue, 0.5 * sv / max(float(sv.sum()), 1.0) * max(float(np.linalg.norm(hue)), 1e-6)])
        grid = cv2.resize(crop, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        cells = []
        for r in range(2):
            for c in range(2):
                m = mag[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32]
                a = ang[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32]
                cells.append(np.histogram(a, bins=8, range=(0, 1), weights=m)[0].astype(np.float32))
        edges = np.concatenate(cells)

        def u(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else v
        v = np.concatenate([1.0 * u(hist), 0.7 * u(grid - grid.mean() + 1e-6), 0.5 * u(edges)])
        return u(v)
