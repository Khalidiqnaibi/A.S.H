"""
src/daemon/ambient.py

The always-on loop.

One thread, one queue, one clock. Sensors are polled on their own intervals
from a single scheduler rather than each running its own thread, because a
24h process with fifteen independent timer threads is a process whose failure
modes you cannot reason about. The two exceptions are sensors that are
inherently push-driven -- the microphone runs its own capture thread and
hands complete utterances to the queue -- and even those are *drained* by the
main loop, so ordering and backpressure stay in one place.

Per tick:

    1. Poll every sensor that is due and not privacy-paused.
    2. Journal each event (redacted, TTL-bounded).
    3. Ask the attention gate: IGNORE / OBSERVE / RESPOND.
    4. OBSERVE  -> brain.observe(). Sub-millisecond. No LLM.
       RESPOND -> brain.think() with ambient context, then deliver.
    5. Housekeeping: watchdog, idle detection, sleep phase.

Backpressure
------------
The queue is bounded. If sensors outpace processing -- which happens when the
LLM is mid-response and a burst of events arrives -- the loop drops the
*lowest-salience* events rather than the oldest. Losing forty window switches
to keep one spoken sentence is the correct trade, and a plain ring buffer
would make the opposite one.

Delivery
--------
`response_sink` is how ASH speaks. Default is a logger; the service wires it
to notification + TTS. Keeping it injectable means the same runtime drives a
headless box, a desktop with a tray icon, or a test harness with no side
effects at all.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from .attention import AttentionGate, Disposition
from .journal import Journal
from .privacy import PrivacyPolicy

logger = logging.getLogger("ash.daemon.ambient")


@dataclass
class AmbientConfig:
    tick_seconds: float = 1.0
    queue_limit: int = 256
    sleep_after_idle_s: float = 25 * 60
    sleep_min_interval_s: float = 60 * 60
    watchdog_stall_s: float = 120.0
    context_window_s: float = 1800.0
    max_response_chars: int = 600


class AmbientRuntime:
    """The 24h loop."""

    def __init__(self, ash, periphery, journal: Optional[Journal] = None,
                 gate: Optional[AttentionGate] = None,
                 privacy: Optional[PrivacyPolicy] = None,
                 config: Optional[AmbientConfig] = None,
                 response_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.ash = ash
        self.brain = ash.brain
        self.periphery = periphery
        self.journal = journal or Journal()
        self.gate = gate or AttentionGate()
        self.privacy = privacy or PrivacyPolicy()
        self.cfg = config or AmbientConfig()
        self.response_sink = response_sink or self._default_sink

        self._queue: Deque = deque()
        self._qlock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.started_at: Optional[float] = None
        self.last_tick: float = 0.0
        self.last_sleep: float = 0.0
        self.stats = {
            "ticks": 0, "events": 0, "dropped": 0,
            "observed": 0, "responded": 0, "ignored": 0, "errors": 0,
        }
        self._user_busy = 0.0
        self._last_activity = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ash-ambient")
        self._thread.start()
        logger.info("Ambient runtime started")

    def stop(self, timeout: float = 5.0):
        logger.info("Ambient runtime stopping")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.journal.flush()
        self.periphery.close_all()
        logger.info("Ambient runtime stopped. %s", self.stats)

    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------
    def submit(self, event):
        """Sensors and external callers push events here."""
        with self._qlock:
            if len(self._queue) >= self.cfg.queue_limit:
                # Drop the least salient rather than the oldest.
                victim = min(range(len(self._queue)),
                             key=lambda i: (self._queue[i].addressed,
                                            self._queue[i].urgency,
                                            self._queue[i].salience))
                if (self._queue[victim].salience < event.salience
                        or event.addressed or event.urgency > 0.5):
                    del self._queue[victim]
                    self.stats["dropped"] += 1
                else:
                    self.stats["dropped"] += 1
                    return
            self._queue.append(event)

    def _drain(self) -> List:
        with self._qlock:
            out = list(self._queue)
            self._queue.clear()
        # Addressed speech first, then by urgency. A queued question should
        # not wait behind eleven window switches.
        return sorted(out, key=lambda e: (not e.addressed, -e.urgency, e.ts))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick()
            except Exception:
                self.stats["errors"] += 1
                logger.exception("Ambient tick failed")
            self.last_tick = time.time()
            self.stats["ticks"] += 1

            elapsed = time.time() - t0
            self._stop.wait(max(0.05, self.cfg.tick_seconds - elapsed))

    def _tick(self):
        now = time.time()

        # 1. Poll due sensors.
        for sensor in list(self.periphery.sensors.values()):
            if not sensor.due(now):
                continue
            if not self.privacy.allows(sensor):
                sensor.last_read = now   # keep it from stampeding on resume
                continue
            for ev in sensor.poll():
                self.privacy.scrub_event(ev)
                self.submit(ev)

        # 2. Process.
        events = self._drain()
        for ev in events:
            self.stats["events"] += 1
            self._handle(ev)

        # 3. Housekeeping.
        self._update_busy()
        self._maybe_sleep(now)

    # ------------------------------------------------------------------
    def _handle(self, ev):
        self.journal.append(ev)

        novelty = 0.5
        try:
            drives = self.brain.homeostasis.drives
            decision = self.gate.decide(ev, drives, novelty=novelty,
                                        user_busy=self._user_busy)
        except Exception:
            logger.exception("Attention gate failed; defaulting to observe")
            decision = None

        if decision is None or decision.disposition == Disposition.IGNORE:
            self.stats["ignored"] += 1
            return

        if decision.disposition == Disposition.OBSERVE:
            self.stats["observed"] += 1
            try:
                trace = self.brain.observe(
                    text=ev.text,
                    image=(ev.data if getattr(ev.modality, "value", ev.modality) == "vision" else None),
                    source=ev.source, salience=ev.salience,
                )
                # Feed real novelty back so the next decision on a similar
                # event is better informed than the 0.5 placeholder above.
                self._last_novelty = trace.percepts.get("novelty", 0.5)
            except Exception:
                logger.exception("brain.observe failed")
            return

        # RESPOND
        self.stats["responded"] += 1
        self._last_activity = time.time()
        self._respond(ev, decision)

    def _respond(self, ev, decision):
        """Full cognitive cycle, with ambient context attached."""
        context = self.journal.context_block(self.cfg.context_window_s)

        if ev.addressed:
            prompt = ev.text
        else:
            # Unprompted remarks get an explicit framing so the narrator knows
            # it is volunteering rather than answering. Without this the LLM
            # reliably writes as though it had been asked a question.
            prompt = (
                f"[UNPROMPTED OBSERVATION -- the user did not ask you anything. "
                f"You noticed: {ev.text}. Decide whether this is worth one short "
                f"remark. If it isn't clearly useful, say nothing of substance.]"
            )

        if context:
            prompt = f"{prompt}\n\n[RECENT AMBIENT CONTEXT]\n{context}"

        try:
            image = ev.data if getattr(ev.modality, "value", ev.modality) == "vision" else None
            response = self.ash.think(prompt, image=image)
            text = (response.text or "").strip()[: self.cfg.max_response_chars]
        except Exception:
            logger.exception("Response cycle failed")
            return

        if not text:
            return

        try:
            self.response_sink(text, {
                "addressed": ev.addressed,
                "source": ev.source,
                "attention": decision.as_dict(),
                "pathway": response.pathway.value,
            })
        except Exception:
            logger.exception("Response sink failed")

        if ev.addressed:
            self.gate.note_user_turn()

    # ------------------------------------------------------------------
    def _update_busy(self):
        """0..1 estimate of how interruptible the user is."""
        idle_sensor = self.periphery.sensors.get("idle")
        if idle_sensor is None:
            self._user_busy = 0.35
            return
        try:
            idle = idle_sensor.seconds_idle()
        except Exception:
            return
        if idle < 5:
            self._user_busy = 1.0        # actively typing
        elif idle < 60:
            self._user_busy = 0.6
        elif idle < 300:
            self._user_busy = 0.25
        else:
            self._user_busy = 0.0        # away

    def _maybe_sleep(self, now: float):
        """Trigger the consolidation phase when the machine goes quiet.

        Unlike the original idle scheduler this has real evidence to work
        with: it knows whether the user is at the keyboard, and (via the
        system sensor) whether the machine is on mains power -- which is what
        the README always wanted 'sleep mode' to key off.
        """
        idle_for = now - self._last_activity
        if idle_for < self.cfg.sleep_after_idle_s:
            return
        if now - self.last_sleep < self.cfg.sleep_min_interval_s:
            return
        if self._user_busy > 0.3:
            return
        self.last_sleep = now
        threading.Thread(target=self._sleep_phase, daemon=True,
                         name="ash-sleep").start()

    def _sleep_phase(self):
        logger.info("Sleep phase starting")
        report: Dict[str, Any] = {}
        try:
            report["brain"] = self.brain.sleep(seconds_idle=self.cfg.sleep_after_idle_s)
        except Exception:
            logger.exception("Brain consolidation failed")
        try:
            report["rollup"] = self.journal.rollup_pending(
                episodic=getattr(self.ash, "episodic_memory", None))
        except Exception:
            logger.exception("Journal rollup failed")
        try:
            from ..senses.screen import SCREEN_DIR
            report["privacy_sweep"] = self.privacy.sweep(self.journal.dir, SCREEN_DIR)
        except Exception:
            logger.exception("Privacy sweep failed")
        logger.info("Sleep phase complete: %s", report)
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _default_sink(text: str, meta: Dict[str, Any]):
        tag = "ANSWER" if meta.get("addressed") else "REMARK"
        logger.info("[%s] %s", tag, text)
        print(f"\nASH ({tag.lower()}): {text}\n", flush=True)

    # ------------------------------------------------------------------
    def healthy(self) -> bool:
        """Watchdog predicate: has the loop ticked recently?"""
        if not self.alive():
            return False
        return (time.time() - self.last_tick) < self.cfg.watchdog_stall_s

    def status(self) -> Dict[str, Any]:
        up = time.time() - self.started_at if self.started_at else 0
        return {
            "alive": self.alive(),
            "healthy": self.healthy(),
            "uptime_h": round(up / 3600, 2),
            "stats": dict(self.stats),
            "queue": len(self._queue),
            "user_busy": round(self._user_busy, 2),
            "attention": self.gate.status(),
            "privacy": self.privacy.status(),
            "journal": self.journal.status(),
            "periphery": self.periphery.status(),
            "brain": self.brain.status(),
        }
