"""
src/daemon/photobooth.py

The fun camera (BMO): a photo booth with a countdown and silly filters.

    take a silly photo / photo booth / take a fun picture            a random filter
    take a photo with the cartoon filter / take a sketch of me       cartoon, sketch, mirror, retro, comic, BMO frame
    send it to my phone                                              the photo just taken (phone.py)

"Three, two, one -- cheese!" is said, the picture is taken a couple of seconds later (one frame, from the camera ASH
already uses, or opened just for that and let go), the filter is applied here with OpenCV, and the result is saved
to the photos folder (media.photo_dir). Only because it was asked, never in privacy mode, never for a guest; the
unfiltered frame is not kept.
"""

from __future__ import annotations

import os
import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

FILTERS = ("cartoon", "sketch", "mirror", "retro", "comic", "bmo")
_ALIAS = {"cartoon": "cartoon", "toon": "cartoon", "sketch": "sketch", "pencil": "sketch", "drawing": "sketch",
          "mirror": "mirror", "twin": "mirror", "twins": "mirror", "retro": "retro", "old": "retro", "vintage": "retro",
          "sepia": "retro", "comic": "comic", "pop": "comic", "pop art": "comic", "bmo": "bmo", "frame": "bmo", "framed": "bmo"}
_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|let'?s)[,\s]+)*"
_BOOTH = re.compile(_LEAD + r"(?:(?:take|snap|make)\s+(?:a|an|another|my|us\s+a)?\s*(?:silly|fun|funny|cool|goofy|crazy)\s+(?:photo|picture|pic|selfie)"
                    r"|photo\s*booth(?:\s+time)?|do\s+the\s+photo\s*booth"
                    r"|(?:take|snap|make)\s+(?:a|an|my|us\s+a)?\s*(?:photo|picture|pic|selfie)\s+(?:of\s+(?:me|us)\s+)?(?:with|in)\s+(?:a|an|the)?\s*(?P<f>[a-z ]{3,12}?)\s*(?:filter|style|effect|frame)?"
                    r"|(?:make|draw|take)\s+(?:a\s+)?(?P<f2>cartoon|sketch|comic)\s+(?:of\s+)?(?:me|us))"
                    r"(?:\s+(?:of\s+(?:me|us)|please|now))*\s*[.!]*\s*$", re.I)


def apply(frame, name: str):
    """The frame with the filter `name` applied (a new BGR uint8 array the same size)."""
    import cv2
    import numpy as np
    img = np.ascontiguousarray(frame[:, :, :3]).astype(np.uint8)
    if name == "cartoon":
        color = cv2.bilateralFilter(img, 9, 120, 120)
        gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
        return cv2.bitwise_and(color, color, mask=edges)
    if name == "sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sk = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)
    if name == "mirror":
        out = img.copy()
        w = img.shape[1]
        out[:, w - w // 2:] = cv2.flip(img[:, : w // 2], 1)
        return out
    if name == "retro":
        k = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        sep = np.clip(img.astype(np.float32) @ k.T, 0, 255)
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        vig = 1 - 0.55 * (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2
        return np.clip(sep * vig[..., None], 0, 255).astype(np.uint8)
    if name == "comic":
        q = (img // 64) * 64 + 32
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        q[cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0] = 0
        return q.astype(np.uint8)
    if name == "bmo":
        h, w = img.shape[:2]
        pad = max(12, min(h, w) // 12)
        teal = (170, 200, 60)                                 # BGR: a BMO-ish teal
        out = cv2.copyMakeBorder(img, pad, pad * 2, pad, pad, cv2.BORDER_CONSTANT, value=teal)
        out = cv2.resize(out, (w, h))
        cv2.putText(out, "A.S.H", (pad, h - pad // 2), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, h / 500), (40, 40, 40), 2, cv2.LINE_AA)
        return out
    return img.copy()


def filter_of(text: str) -> Optional[str]:
    """The filter named in a photo-booth request ("" = any), or None if it isn't one."""
    m = _BOOTH.match((text or "").strip())
    if not m:
        return None
    raw = (m.group("f") or m.group("f2") or "").strip().lower()
    if not raw:
        return ""
    for word in (raw, raw.split()[-1], raw.split()[0]):
        if word in _ALIAS:
            return _ALIAS[word]
    return None if m.group("f") else ""


class PhotoBooth:
    def __init__(self, grab: Callable[[], Any], folder: str = "~/Pictures/ASH", clock: Callable[[], float] = time.time,
                 rng: Optional[random.Random] = None, write: Optional[Callable[[str, Any], bool]] = None):
        self.grab = grab
        self.folder = folder
        self.clock = clock
        self.rng = rng or random.Random()
        self.write = write
        self.last_path: Optional[str] = None
        self.last_at = 0.0
        self.stats = {"taken": 0, "failed": 0}

    def countdown(self, name: str) -> str:
        what = {"cartoon": "cartoon", "sketch": "sketch", "mirror": "mirror twin", "retro": "old-timey", "comic": "comic book",
                "bmo": "framed"}.get(name, name)
        return f"Photo booth! A {what} one. Three... two... one... cheese!"

    def pick(self, name: str) -> str:
        return name if name in FILTERS else self.rng.choice(FILTERS)

    def take(self, name: str, now: Optional[float] = None) -> str:
        """Take one frame, apply the filter, save it. What to say."""
        now = self.clock() if now is None else now
        try:
            frame = self.grab()
        except Exception:
            frame = None
        if frame is None:
            self.stats["failed"] += 1
            return "I couldn't get a picture -- is the camera free?"
        try:
            out = apply(frame, name)
        except Exception:
            self.stats["failed"] += 1
            return "Something went wrong with the filter -- try another one."
        folder = os.path.expanduser(self.folder)
        path = os.path.join(folder, datetime.fromtimestamp(now).strftime("booth_%Y-%m-%d_%H%M%S") + f"_{name}.jpg")
        try:
            os.makedirs(folder, exist_ok=True)
            if self.write is not None:
                ok = self.write(path, out)
            else:
                import cv2
                ok = bool(cv2.imwrite(path, out))
        except Exception:
            ok = False
        if not ok:
            self.stats["failed"] += 1
            return "I took it, but couldn't save it."
        self.last_path = path
        self.last_at = now
        self.stats["taken"] += 1
        return f"Got it! Saved as {os.path.basename(path)} in your pictures. Say “send it to my phone” to have it there."

    def status(self) -> Dict[str, Any]:
        return dict(self.stats)
