"""
Maintenance-phase scheduler for ASH.

The README describes a "sleep mode" that runs heavier upkeep
(episodic pruning, index rebuilds, etc.) during charging/WiFi/idle
periods, but no such scheduler existed in the codebase yet -- this
is that hook, wired in the simplest way that fits without inventing
hardware APIs this project doesn't have.

ASH currently has no charging/WiFi sensor layer (it's a software
runtime, not hardware-integrated yet per the README's own "Future
Extensions" section), so the practical proxy used here is IDLE TIME:
no query processed for `idle_seconds` triggers a maintenance sweep.
Swap `is_idle()` for a real charging/WiFi check later without
touching anything else -- that's the only function this depends on.

Usage (call once, near where `ash = ASH()` is instantiated):

    from src.memory.maintenance import MaintenanceScheduler
    scheduler = MaintenanceScheduler(ash_instance, idle_seconds=1800)
    scheduler.start()

The scheduler runs on a daemon background thread and does not block
the main app/request-handling thread.
"""

from __future__ import annotations
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("ash.memory.maintenance")


class MaintenanceScheduler:
    def __init__(self, ash_instance, idle_seconds: float = 30 * 60,
                 check_interval: float = 60, prune_min_age_seconds: float = 7 * 86400):
        self.ash = ash_instance
        self.idle_seconds = idle_seconds
        self.check_interval = check_interval
        self.prune_min_age_seconds = prune_min_age_seconds
        self._last_activity = time.time()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ran_since_last_activity = False

    def note_activity(self):
        """Call this whenever a real user turn happens (e.g. from
        ASH.run()) so is_idle() has an accurate last-activity time."""
        self._last_activity = time.time()
        self._ran_since_last_activity = False

    def is_idle(self) -> bool:
        """Placeholder trigger -- see module docstring. Replace with a
        real charging/WiFi check when ASH gets a hardware layer."""
        return (time.time() - self._last_activity) >= self.idle_seconds

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ash-maintenance")
        self._thread.start()
        logger.info("Maintenance scheduler started (idle threshold=%.0fs)", self.idle_seconds)

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.check_interval)
            if self.is_idle() and not self._ran_since_last_activity:
                self.run_sweep()
                self._ran_since_last_activity = True  # don't re-run every check_interval while still idle

    def run_sweep(self):
        """Runs once per idle period, not on every check_interval tick."""
        logger.info("Maintenance sweep starting (idle >= %.0fs)", self.idle_seconds)
        try:
            episodic = getattr(self.ash, "episodic_memory", None)
            if episodic is not None:
                pruned = episodic.prune(min_age_seconds=self.prune_min_age_seconds)
                logger.info("Maintenance sweep: pruned %d unused episode(s)", pruned)
            else:
                logger.warning("Maintenance sweep: ash_instance has no episodic_memory -- skipping prune")
        except Exception:
            logger.exception("Maintenance sweep failed")

        # Brain consolidation ("sleep"): replay-train the predictive forward
        # model, repay homeostatic fatigue, flush critic weights to disk.
        # Deliberately runs after pruning and is fully isolated -- a failure
        # here must never take down the maintenance thread. See
        # src/brain/consolidation.py.
        try:
            brain = getattr(self.ash, "brain", None)
            if brain is not None:
                idle_for = time.time() - self._last_activity
                report = brain.sleep(seconds_idle=idle_for)
                logger.info("Maintenance sweep: brain consolidation -> %s", report.get("replay"))
        except Exception:
            logger.exception("Brain consolidation failed during maintenance sweep")