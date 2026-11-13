"""
src/vision/eye.py

One frame from the camera, when asked for -- and only then.

    look()    a frame for someone who said "look" / "what's this?" / "who's that?". If the camera sensor is running it
              hands over its own freshest frame (a second capture of the same device would fail); otherwise the camera
              is opened for a moment, a few frames are skipped so the exposure settles, one is kept, and the device is
              let go again so its light goes out.
    recent()  that frame, for a minute and in memory only, so "save that photo" after "who's this?" means the photo that
              was just taken. It is dropped when used, when it is old, and when privacy mode goes on.
    save()    the only way a frame reaches the disk, and only because the user said so.

Never while privacy mode is on. Never a stored frame otherwise.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger("ash.vision.eye")


def capture_once(index: int = 0, warmup: int = 4) -> Optional[np.ndarray]:
    """Open the camera, let the exposure settle, take one frame, let go. A BGR array, or None."""
    try:
        import cv2
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                return None
            frame = None
            for _ in range(max(1, warmup)):
                ok, f = cap.read()
                if ok:
                    frame = f
            return frame
        finally:
            cap.release()
    except Exception:
        logger.exception("Camera capture failed")
        return None


class Eye:
    def __init__(self, sensor: Callable[[], Any] = lambda: None, grab: Optional[Callable[[], Optional[np.ndarray]]] = None,
                 paused: Callable[[], bool] = lambda: False, index: int = 0,
                 clock: Callable[[], float] = time.time, keep_s: float = 60.0):
        self.sensor = sensor
        self._grab = grab
        self.paused = paused
        self.index = int(index)
        self.clock = clock
        self.keep_s = float(keep_s)
        self._frame: Optional[np.ndarray] = None
        self._at = 0.0
        self.looks = 0

    def look(self, now: Optional[float] = None) -> Tuple[Optional[np.ndarray], str]:
        """(frame, "sensor" | "camera") or (None, "privacy" | "busy" | "no camera"). "busy": the camera sensor has the
        device open and gave no frame, so a second open would fail too."""
        now = self.clock() if now is None else now
        if self.paused():
            self.drop()
            return None, "privacy"
        frame, how, held = None, "camera", False
        try:
            s = self.sensor()
            if s is not None and getattr(s, "state", None) is not None and getattr(s.state, "value", "") == "active":
                held = True                                   # the camera sensor has the device open
                frame, how = s.grab_now(), "sensor"
        except Exception:
            logger.debug("sensor frame failed", exc_info=True)
        if frame is None:
            frame, how = (self._grab() if self._grab is not None else capture_once(self.index)), "camera"
        if frame is None:
            return None, ("busy" if held else "no camera")
        self._frame, self._at = frame, now
        self.looks += 1
        return frame, how

    def recent(self, now: Optional[float] = None) -> Optional[np.ndarray]:
        now = self.clock() if now is None else now
        if self._frame is None or now - self._at > self.keep_s or self.paused():
            self.drop()
            return None
        return self._frame

    def drop(self) -> None:
        self._frame, self._at = None, 0.0

    def save(self, frame: np.ndarray, folder: str, label: str = "", now: Optional[float] = None) -> Optional[str]:
        """Write a frame as a JPEG. Called only because the user asked; returns the path."""
        try:
            import cv2
            now = self.clock() if now is None else now
            os.makedirs(os.path.expanduser(folder), exist_ok=True)
            safe = "".join(c for c in label.lower() if c.isalnum() or c in "-_ ").strip().replace(" ", "-")[:30]
            name = time.strftime("ASH-%Y%m%d-%H%M%S", time.localtime(now)) + (f"-{safe}" if safe else "") + ".jpg"
            path = os.path.join(os.path.expanduser(folder), name)
            return path if cv2.imwrite(path, frame) else None
        except Exception:
            logger.exception("Could not save the photo")
            return None
