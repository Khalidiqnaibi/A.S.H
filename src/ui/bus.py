"""
src/ui/bus.py

One stream of everything the console shows.

Two kinds of message travel on it, and they are kept apart on purpose:

  RECORDS   chat lines, cues ("listening", "thinking", "ran launch"), status.
            Numbered and kept in a bounded history, so a window opened late
            -- or reopened after being closed -- shows the conversation so far
            instead of an empty pane.

  POSES     the face, 30 times a second. Never stored: only the latest one
            matters, and replaying a backlog of stale poses would make the
            face stutter through old expressions on connect.

Subscribers get a bounded queue each. A window that stops reading (minimised
browser tab, hung renderer) loses old poses rather than growing memory
without limit; records are never dropped for a live subscriber because the
queue holds far more than a conversation produces between two reads.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

HISTORY = 400


class UIBus:
    def __init__(self, history: int = HISTORY):
        self._history: deque = deque(maxlen=history)
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self.pose: Optional[Dict[str, Any]] = None
        self.state: Dict[str, Any] = {}     # sticky cues: listening, thinking, speaking

    # ------------------------------------------------------------------
    def publish(self, kind: str, **data) -> Dict[str, Any]:
        """Send a record: stored in history and delivered to every window."""
        rec = {"id": next(self._seq), "kind": kind, "ts": time.time(), **data}
        with self._lock:
            self._history.append(rec)
            subs = list(self._subs)
        for q in subs:
            self._offer(q, rec)
        return rec

    def cue(self, name: str, on: bool, **data):
        """A state that is either happening or not (thinking, speaking).
        Remembered so a window that connects mid-thought shows it."""
        if on:
            self.state[name] = {"since": time.time(), **data}
        else:
            self.state.pop(name, None)
        return self.publish("cue", cue=name, on=on, **data)

    def publish_pose(self, pose: Dict[str, Any]):
        self.pose = pose
        self.broadcast("pose", **pose)

    def broadcast(self, kind: str, **data):
        """Live-only and lossy: poses and periodic status, never replayed."""
        with self._lock:
            subs = list(self._subs)
        msg = {"kind": kind, **data}
        for q in subs:
            self._offer(q, msg, lossy=True)

    # ------------------------------------------------------------------
    def subscribe(self, maxsize: int = 600) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def history(self, after: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            return [r for r in self._history if r["id"] > after]

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._subs)

    # ------------------------------------------------------------------
    @staticmethod
    def _offer(q: queue.Queue, msg: Dict[str, Any], lossy: bool = False):
        try:
            q.put_nowait(msg)
            return
        except queue.Full:
            pass
        if lossy:
            return
        # A record must get through: make room by dropping the oldest item,
        # which in a backed-up queue is almost always a stale pose.
        try:
            q.get_nowait()
            q.put_nowait(msg)
        except (queue.Empty, queue.Full):
            pass
