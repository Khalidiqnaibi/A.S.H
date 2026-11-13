"""
src/daemon/visionwatch.py

ASH noticing what the camera sees change -- "you picked up a watch", "the watch is gone" -- by rule, no language model.

Only from the camera SENSOR's own latest frame (never a new capture, so the camera is never opened for this), which means
only while `sensors.camera` is on; and never while privacy mode is on, in kids mode, or while a stranger is in view. One
reading every `vision.watch_interval_s` on a worker thread (a classifier can take a second on a CPU; the loop never waits
for it), the answer picked up on a later step. What a reading finds is recorded -- labels, confidences and boxes, in the
ring and in state/vision/detections.jsonl -- and, when `vision.announce_objects` is on, said once (a cool-down per label).
Nothing of a frame is kept.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("ash.daemon.visionwatch")


class SceneWatcher:
    def __init__(self, rt):
        self.rt = rt
        self._job = None
        self._pool = None
        self._at = 0.0
        self._job_at = 0.0
        self.stats = {"readings": 0, "spoken": 0, "skipped": 0}

    # -- when not to look at all -----------------------------------------------------------------
    def _blocked(self) -> str:
        rt = self.rt
        lk = getattr(rt, "looking", None)
        if lk is None or not lk.cfg.enabled or not lk.cfg.react:
            return "off"
        if rt.privacy.paused:
            return "privacy"
        kids = getattr(rt, "kids", None)
        if kids is not None and kids.active:
            return "kids"
        return ""

    def _frame(self):
        cam = getattr(self.rt.periphery, "sensors", {}).get("camera")
        if cam is None or getattr(getattr(cam, "state", None), "value", "") != "active":
            return None
        ts = float(getattr(cam, "last_frame_ts", 0.0) or 0.0)
        frame = getattr(cam, "last_frame", None)
        if frame is None or time.time() - ts > 10.0:
            return None
        try:
            return frame.copy()                               # the camera thread keeps writing
        except Exception:
            return None

    # -- the loop's step ---------------------------------------------------------------------------
    def step(self, now: float) -> None:
        rt = self.rt
        self._collect()
        why = self._blocked()
        if why:
            if why in ("privacy", "kids"):
                self._drop()                                  # a reading in flight is not used
            return
        lk = rt.looking
        if now - self._at < lk.cfg.watch_interval_s:
            return
        self._at = now
        if self._job is not None:
            if time.time() - self._job_at > 3 * lk.cfg.watch_interval_s + 30:
                logger.error("A camera-change reading took too long; abandoning it")
                self._drop()
            return
        snap = self._frame()
        if snap is None:
            self.stats["skipped"] += 1
            return
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ash-vision-watch")
        self._job = self._pool.submit(lk.watch_scene, snap, now)
        self._job_at = time.time()
        self.stats["readings"] += 1

    def _drop(self):
        self._job = None
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _collect(self) -> None:
        job = self._job
        if job is None or not job.done():
            return
        self._job = None
        try:
            events = job.result()
        except Exception:
            logger.exception("Camera-change reading failed")
            return
        if self._blocked():                                   # privacy / kids went on while it was reading
            return
        self.say(events)

    def say(self, events) -> None:
        """Count each change, and say it aloud when switched on and it is polite to: not muted, not quiet, not with a
        stranger in view."""
        rt, lk = self.rt, self.rt.looking
        for kind, label, conf in events:
            rt.stats["vision_" + kind] = rt.stats.get("vision_" + kind, 0) + 1
            line = lk.change_line(kind, label, conf)
            if not line:
                continue
            if rt.gate.muted or rt.privacy.paused or getattr(rt, "_quiet_until", None) is not None:
                continue
            try:
                if lk.stranger_present(rt.clock()):
                    continue
                rt.response_sink(line, {"addressed": False, "source": "vision", "attention": None, "pathway": "wonder",
                                        "outcome": "vision"})
                self.stats["spoken"] += 1
            except Exception:
                logger.exception("Saying a camera change failed")

    def close(self) -> None:
        self._drop()
