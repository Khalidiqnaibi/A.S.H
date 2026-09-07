"""
src/face/player.py

The playback engine.

Three things are happening simultaneously on every frame, and keeping them
separate is what makes the face feel alive rather than scripted:

  1. THE ACTIVE ANIMATION -- whatever was last requested, sampled at its
     current time, crossfaded from whatever it replaced.

  2. THE AUTO-BLINKER -- an independent layer that multiplies `open`. Because
     it is a layer rather than part of each animation, every animation blinks
     correctly without its author thinking about it, and `blink_suppress`
     turns it off for the ones where blinking would break the read.

  3. DRIVE MODULATION -- a continuous bias from ASH's homeostatic drives.
     Between animations the face is not a static neutral: fatigue lowers the
     lids, curiosity widens the eyes, caution narrows them. This is why the
     face still looks like it belongs to something when nothing is playing.

Blink timing
------------
Human inter-blink intervals are not uniform; they cluster around 2-6 seconds
with a long tail, and blinks arrive in occasional doubles. `_next_blink_delay`
samples a log-normal, which costs one line and is the difference between "a
robot that blinks" and "a robot on a timer".

Scheduling
----------
`play()` is priority-based with an interruptible flag. A request that loses
is dropped rather than queued -- a face that works through a backlog of stale
reactions is worse than one that misses a few, because by the time the queue
drains the reaction no longer matches anything.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .library import BLINK, DOUBLE_BLINK, BUILTINS, load_library
from .model import Animation, FaceParams, REST, blend

logger = logging.getLogger("ash.face.player")

CROSSFADE_S = 0.18


class FacePlayer:
    """Owns the current pose and the thread that advances it."""

    def __init__(self, renderer=None, driver=None, fps: int = 30,
                 library: Optional[Dict[str, Animation]] = None,
                 auto_blink: bool = True, animations_dir: str = "animations"):
        self.renderer = renderer
        self.driver = driver
        self.fps = fps
        self.library = library if library is not None else load_library(animations_dir)
        self.auto_blink = auto_blink

        self.current: Optional[Animation] = None
        self.current_t: float = 0.0
        self._started: float = 0.0

        # Crossfade state.
        self._from_pose: Optional[FaceParams] = None
        self._fade_start: float = 0.0

        # Blink layer.
        self._blink: Optional[Animation] = None
        self._blink_t: float = 0.0
        self._next_blink: float = time.time() + 2.0

        # Drive modulation.
        self._drive_bias: Dict[str, float] = {}

        self.pose: FaceParams = REST.copy()
        self.idle_name = "idle"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames = 0
        self.dropped = 0
        self.on_frame: Optional[Callable[[FaceParams, Any], None]] = None

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def play(self, name: str, force: bool = False) -> bool:
        """Request an animation. Returns whether it won."""
        anim = self.library.get(name)
        if anim is None:
            logger.warning("No animation named %r", name)
            return False
        return self.play_animation(anim, force=force)

    def play_animation(self, anim: Animation, force: bool = False) -> bool:
        with self._lock:
            cur = self.current
            if cur is not None and not force:
                # A running animation holds the screen if it outranks the
                # request, or ties but refuses to be cut.
                if cur.priority > anim.priority:
                    self.dropped += 1
                    return False
                if cur.priority == anim.priority and not cur.interruptible:
                    if not cur.finished(self.current_t):
                        self.dropped += 1
                        return False

            self._from_pose = self.pose.copy()
            self._fade_start = time.time()
            self.current = anim
            self.current_t = 0.0
            self._started = time.time()
            return True

    def stop(self):
        with self._lock:
            self.current = None
            self.current_t = 0.0

    def set_idle(self, name: str):
        self.idle_name = name

    # ------------------------------------------------------------------
    # Drive modulation
    # ------------------------------------------------------------------
    def set_drives(self, drives):
        """Continuous bias from the homeostatic modulator.

        Small numbers on purpose. These ride on top of every animation, so
        anything large would fight the authored keyframes; the goal is a
        detectable baseline mood, not an expression.
        """
        try:
            self._drive_bias = {
                # Tired eyes sit lower and narrower.
                "lid_top": 0.22 * float(drives.fatigue),
                # Curiosity opens them up.
                "h": 3.0 * (float(drives.curiosity) - 0.4),
                "w": 1.5 * (float(drives.curiosity) - 0.4),
                # Caution narrows, urgency brightens.
                "lid_bottom": 0.12 * max(0.0, float(drives.caution) - 0.5),
                "brightness": 0.85 + 0.15 * float(drives.urgency),
            }
        except Exception:
            logger.debug("Bad drives object", exc_info=True)

    def _apply_bias(self, pose: FaceParams) -> FaceParams:
        b = self._drive_bias
        if not b:
            return pose
        for side in ("left", "right"):
            eye = getattr(pose, side)
            eye.lid_top = min(0.85, eye.lid_top + b.get("lid_top", 0.0))
            eye.lid_bottom = min(0.85, eye.lid_bottom + b.get("lid_bottom", 0.0))
            eye.h = max(4.0, eye.h + b.get("h", 0.0))
            eye.w = max(4.0, eye.w + b.get("w", 0.0))
        pose.brightness = max(0.05, min(1.0, pose.brightness * b.get("brightness", 1.0)))
        return pose

    # ------------------------------------------------------------------
    # Blink layer
    # ------------------------------------------------------------------
    @staticmethod
    def _next_blink_delay() -> float:
        """Log-normal inter-blink interval, ~2-8s with a long tail."""
        return min(14.0, max(1.2, random.lognormvariate(1.15, 0.45)))

    def _tick_blink(self, now: float, suppressed: bool):
        if not self.auto_blink or suppressed:
            # Push the schedule forward so blinks don't all fire the instant
            # suppression lifts.
            self._next_blink = max(self._next_blink, now + 0.8)
            return
        if self._blink is not None:
            self._blink_t = now - self._blink_start
            if self._blink.finished(self._blink_t):
                self._blink = None
                self._next_blink = now + self._next_blink_delay()
            return
        if now >= self._next_blink:
            self._blink = DOUBLE_BLINK if random.random() < 0.12 else BLINK
            self._blink_start = now
            self._blink_t = 0.0

    def _blink_factor(self) -> float:
        if self._blink is None:
            return 1.0
        p = self._blink.sample(self._blink_t)
        # Both eyes blink together; take the left as representative.
        return max(0.0, min(1.0, p.left.open))

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------
    def advance(self, now: Optional[float] = None) -> FaceParams:
        """Compute the pose for this instant. Pure: safe to call from a test
        without a thread or a driver."""
        now = now or time.time()

        with self._lock:
            anim = self.current
            if anim is None:
                anim = self.library.get(self.idle_name)
                if anim is not None:
                    self.current = anim
                    self._started = now

            if anim is None:
                pose = REST.copy()
            else:
                self.current_t = now - self._started
                if anim.finished(self.current_t) and not anim.hold_last:
                    self.current = None
                    idle = self.library.get(self.idle_name)
                    if idle is not None:
                        self._from_pose = self.pose.copy()
                        self._fade_start = now
                        self.current = idle
                        self._started = now
                        anim = idle
                        self.current_t = 0.0
                pose = anim.sample(self.current_t)

            # Crossfade out of whatever was on screen before.
            if self._from_pose is not None:
                u = (now - self._fade_start) / CROSSFADE_S
                if u >= 1.0:
                    self._from_pose = None
                else:
                    pose = blend(self._from_pose, pose, u)

            suppressed = bool(anim and anim.blink_suppress)
            self._tick_blink(now, suppressed)

            pose = self._apply_bias(pose)

            f = self._blink_factor()
            if f < 1.0:
                pose.left.open *= f
                pose.right.open *= f

            self.pose = pose
            return pose

    def render_frame(self, now: Optional[float] = None):
        pose = self.advance(now)
        img = None
        if self.renderer is not None:
            img = self.renderer.render(pose)
        if self.driver is not None and img is not None:
            try:
                self.driver.show(img, pose)
            except Exception:
                logger.exception("Driver failed")
        if self.on_frame is not None:
            try:
                self.on_frame(pose, img)
            except Exception:
                logger.exception("on_frame callback failed")
        self.frames += 1
        return pose, img

    # ------------------------------------------------------------------
    # Thread
    # ------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ash-face")
        self._thread.start()
        logger.info("Face player started at %d fps", self.fps)

    def _loop(self):
        period = 1.0 / max(1, self.fps)
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    self.render_frame()
                except Exception:
                    logger.exception("Face frame failed")
                self._stop.wait(max(0.0, period - (time.perf_counter() - t0)))
        finally:
            # Close on this same thread, not whichever thread calls
            # shutdown() -- some drivers (WindowDriver) hold a Tk/Tcl
            # interpreter that only the thread which drove show() may touch.
            if self.driver is not None:
                try:
                    self.driver.close()
                except Exception:
                    logger.exception("Driver close failed")

    def shutdown(self, play_outro: bool = True, timeout: float = 2.0):
        if play_outro and "shutdown" in self.library:
            self.play("shutdown", force=True)
            time.sleep(min(timeout, (self.library["shutdown"].duration or 1.0) + 0.1))
        self._stop.set()
        if self._thread:
            # _loop()'s finally block closes the driver on its own thread
            # once it observes _stop; join just waits for that to happen.
            self._thread.join(timeout=2.0)
        elif self.driver is not None:
            # No render thread was ever started (e.g. the CLI drives
            # render_frame() directly on the calling thread) -- safe to
            # close directly here since no other thread touched the driver.
            try:
                self.driver.close()
            except Exception:
                pass

    def status(self) -> Dict[str, Any]:
        return {
            "current": self.current.name if self.current else None,
            "t": round(self.current_t, 2),
            "frames": self.frames,
            "dropped_requests": self.dropped,
            "blinking": self._blink is not None,
            "animations": len(self.library),
            "driver": type(self.driver).__name__ if self.driver else None,
        }
