"""
src/vision/models.py

Where the model files live, and how they get there -- only when the user says so.

Nothing is downloaded by running ASH, by turning vision on, or by asking it to look. `plan()` says what would be fetched
(name, source, rough size); `install(..., yes=True)` fetches it. The two face models come from OpenCV's own model zoo.
The object detector is not downloaded for you: any YOLOv8/YOLO11-format ONNX works, and `yolo export model=yolov8n.pt
format=onnx` makes one -- put it at `objects_path` and it is used.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.vision.models")

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS: Dict[str, Dict[str, Any]] = {
    "face_detector": {"file": "face_detection_yunet_2023mar.onnx", "url": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                      "approx_mb": 0.3, "what": "finds faces (OpenCV YuNet)"},
    "face_recognizer": {"file": "face_recognition_sface_2021dec.onnx", "url": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                        "approx_mb": 37.0, "what": "describes a face as 128 numbers (OpenCV SFace)"},
}


def default_dir() -> str:
    return os.path.join(os.environ.get("ASH_STATE_DIR", os.path.join(os.getcwd(), "state")), "models")


def path_of(name: str, folder: Optional[str] = None) -> str:
    return os.path.join(folder or default_dir(), MODELS[name]["file"])


def missing(folder: Optional[str] = None) -> List[str]:
    return [n for n in MODELS if not os.path.isfile(path_of(n, folder))]


def plan(folder: Optional[str] = None) -> List[Dict[str, Any]]:
    """What `install` would fetch: only what is not already there."""
    return [dict(name=n, **{k: MODELS[n][k] for k in ("url", "approx_mb", "what")}, to=path_of(n, folder)) for n in missing(folder)]


def _fetch(url: str, to: str) -> None:
    import urllib.request
    os.makedirs(os.path.dirname(to), exist_ok=True)
    tmp = to + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, to)


def install(folder: Optional[str] = None, yes: bool = False, fetch: Callable[[str, str], None] = _fetch) -> Dict[str, Any]:
    """Fetch the missing face models -- but only with `yes=True`. Returns {"plan": [...], "done": [...], "failed": [...]}."""
    todo = plan(folder)
    out: Dict[str, Any] = {"plan": todo, "done": [], "failed": []}
    if not yes:
        return out
    for item in todo:
        try:
            fetch(item["url"], item["to"])
            if os.path.getsize(item["to"]) < 1000:
                raise ValueError("the file is too small to be a model")
            out["done"].append(item["name"])
        except Exception as e:
            logger.warning("Could not fetch %s: %s", item["name"], e)
            try:
                os.remove(item["to"])
            except OSError:
                pass
            out["failed"].append(f"{item['name']}: {e}")
    return out
