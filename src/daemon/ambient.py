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

from src.senses import address
from .attention import AttentionGate, Disposition
from .journal import Journal
from .privacy import PrivacyPolicy
from .social import (FlowDetector, ReactionTracker, SocialLearner, classify_reply,
                     compose_check_in, friendly_app, is_generic_checkin, where_phrase)

try:
    from src.brain.memory_engine import is_failure_text
except Exception:                                        # pragma: no cover
    def is_failure_text(text) -> bool:
        return isinstance(text, str) and text.strip().startswith(("[LLM]", "Error calling"))

logger = logging.getLogger("ash.daemon.ambient")

# Said out loud, to someone who asked a real question, so: what is wrong and
# nothing else. No apology (it is not ASH's fault and saying sorry twice for
# the same outage is worse than the outage) and no jargon dressed up as one.
NO_MODEL_TEXT = "My model isn't answering right now, so I've got nothing for you on that."


@dataclass
class AmbientConfig:
    tick_seconds: float = 1.0
    queue_limit: int = 256
    sleep_after_idle_s: float = 25 * 60
    sleep_min_interval_s: float = 60 * 60
    watchdog_stall_s: float = 120.0
    context_window_s: float = 1800.0
    max_response_chars: int = 600
    follow_plans: bool = True        # carry on a multi-step plan between ticks
    learn_social: bool = True        # learn from how volunteered remarks land
    flow_offers: bool = True         # offer a chat in free moments, help when stuck
    social_path: Optional[str] = None    # where the learned record lives (None: in memory)
    reaction_window_s: float = 120.0     # how long to wait for a reaction to a remark
    no_reply_notice_s: float = 600.0     # how often to say the model is down, at most


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
            "plan_steps": 0,
        }
        self._user_busy = 0.0
        self._last_activity = time.time()
        self.clock: Callable[[], float] = time.time

        # The social loop: what was said, how it landed, and when to speak.
        self.learner: Optional[SocialLearner] = None
        self.reactions: Optional[ReactionTracker] = None
        self.flow: Optional[FlowDetector] = FlowDetector() if self.cfg.flow_offers else None
        if self.cfg.learn_social:
            self.learner = getattr(self.gate, "learner", None) or SocialLearner(
                path=self.cfg.social_path)
            self.gate.learner = self.learner
            self.learner.clock = lambda: self.clock()
            self.reactions = ReactionTracker(self.learner, window_s=self.cfg.reaction_window_s)
        self._reaction = None            # the reaction the event being handled carried
        self.stats["offers"] = 0
        self.stats["no_reply"] = 0
        self._last_no_reply_notice = -1e12

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        # Clear the stop flag *before* the already-alive check, not only in
        # the branch that spawns a new thread. If the previous ash-ambient
        # thread is still blocked inside a slow synchronous call (an
        # in-flight LLM request -- see the module docstring: _respond() runs
        # on this thread directly, nothing is spawned per response) then a
        # watchdog restart's stop() already set this flag and timed out its
        # join() without actually stopping anything. Leaving it set here
        # silently arms a time bomb: the stuck thread is still running the
        # exact same loop with the exact same flag object, so the moment its
        # current tick finally finishes and it re-checks the flag, it would
        # see "stop" and exit for good -- which is exactly what looked like
        # "ashd answered, then the whole process just exited": the daemon
        # was not killed, its one and only ambient thread quietly honored a
        # restart request that was issued minutes earlier.
        self._stop.clear()
        if self._thread and self._thread.is_alive():
            logger.warning(
                "Ambient runtime start() called while the previous ash-ambient "
                "thread is still running -- most likely still blocked in a slow "
                "call (e.g. an LLM request) that a watchdog restart could not "
                "actually interrupt (Python cannot force-kill a thread). Leaving "
                "it in place rather than spawning a second one; it will resume "
                "ticking normally once that call returns."
            )
            return
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
        responded_before = self.stats["responded"]
        for ev in events:
            self.stats["events"] += 1
            self._handle(ev)

        # 2b. Carry on a plan, one step per tick, but never on a tick where
        # the brain just answered someone: that answer already replaced it.
        if self.stats["responded"] == responded_before:
            self._advance_plan()

        # 2c. Speak up at a free moment or when stuck, and count the remarks
        # nobody reacted to.
        self._social_step(self.clock())

        # 3. Housekeeping.
        self._update_busy()
        self._maybe_sleep(now)

    # ------------------------------------------------------------------
    def _idle_seconds(self) -> Optional[float]:
        sensor = self.periphery.sensors.get("idle")
        if sensor is None:
            return None
        try:
            return float(sensor.seconds_idle())
        except Exception:
            return None

    def _social_step(self, now: float, idle_s: Optional[float] = None):
        try:
            if self.reactions is not None:
                self.reactions.expire(now)
            if self.flow is None:
                return
            if idle_s is None:
                idle_s = self._idle_seconds()
            if idle_s is None:
                return                   # no way to tell when they pause: say nothing
            for offer in self.flow.poll(idle_s, now):
                self.stats["offers"] += 1
                self.submit(offer)
        except Exception:
            self.stats["errors"] += 1
            logger.exception("Social step failed")

    def _handle(self, ev):
        self.journal.append(ev)
        now = self.clock()

        if getattr(ev, "addressed", False):
            # Holds the exchange open from the user's side, so a second
            # sentence arriving before ASH has answered the first still reads
            # as part of the same conversation.
            address.conversation.note_user_turn(now)

        self._reaction = None
        if self.flow is not None:
            self.flow.observe(ev, now)
        if self.reactions is not None:
            try:
                self._reaction = self.reactions.on_user_event(ev, now)
            except Exception:
                logger.exception("Reading the reaction failed")

        novelty = 0.5
        try:
            drives = self.brain.homeostasis.drives
            decision = self.gate.decide(ev, drives, novelty=novelty,
                                        user_busy=self._user_busy, now=now)
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
        meta = getattr(ev, "meta", None) or {}
        # Certainly addressed, or route 1b's "probably". Both are replies to
        # the user and must be phrased as replies -- framing "what time is
        # it" as an unprompted observation gets a remark about the question
        # rather than an answer to it.
        to_ash = bool(getattr(ev, "addressed", False) or getattr(decision, "likely", False))
        offer = None if to_ash else meta.get("offer")
        react = self._reaction if to_ash else None
        say = None                       # a line that needs no model to say

        if to_ash:
            from src.senses.mic import MicrophoneSensor
            prompt = MicrophoneSensor.strip_address(ev.text)
            if react is not None and react.outcome in ("rejected", "deferred"):
                # "Not now" after an offer is answered as a person would, and
                # the offer's kind has already been snoozed.
                say = ("Okay, I'll stay out of your way." if react.outcome == "rejected"
                       else "Okay, I'll check back later.")
            elif react is not None and react.offer.accept_context:
                context = (f"{context}\n[{react.offer.accept_context} "
                           f"The user's message is their answer to that.]").strip()
        elif offer == "stuck_help" and meta.get("say"):
            say = str(meta["say"])
            prompt = ""
        elif offer == "free_moment":
            # The subject ("brain.py") says more about the stretch of work
            # than the app ("VS Code") ever does, so it leads.
            where = where_phrase(str(meta.get("process", "")),
                                 str(meta.get("subject", "")),
                                 str(meta.get("project", ""))).strip()
            prompt = (
                f"[FREE MOMENT -- the user did not ask you anything. Fact: {ev.text}"
                f"{' (' + where + ')' if where else ''}. Reply with ONE short sentence, under 25 "
                f"words, that mentions what they were doing or for how long -- a natural remark "
                f"from a colleague who noticed, e.g. about the stretch of work or a break. Do NOT "
                f"ask how you can help, do NOT say 'hey there', do NOT offer assistance in "
                f"general. At most one question.]")
        else:
            # Unprompted remarks get an explicit framing so the narrator knows
            # it is volunteering rather than answering. Without this the LLM
            # reliably writes as though it had been asked a question.
            prompt = (
                f"[UNPROMPTED OBSERVATION -- the user did not ask you anything. "
                f"You noticed: {ev.text}. Decide whether this is worth one short "
                f"remark. If it isn't clearly useful, say nothing of substance.]"
            )

        pathway = "offer"
        if say is not None:
            text = say[: self.cfg.max_response_chars]
        else:
            try:
                image = ev.data if getattr(ev.modality, "value", ev.modality) == "vision" else None
                response = self.ash.think(prompt, image=image, context=context)
                text = (response.text or "").strip()[: self.cfg.max_response_chars]
                pathway = response.pathway.value
            except Exception:
                logger.exception("Response cycle failed")
                return

        # A plan the brain adopted while volunteering a remark is not
        # followed: the remark was spent against the hourly budget, and the
        # rest of the steps would speak with none.
        if not to_ash:
            self._drop_plan("unprompted")

        degraded = is_failure_text(text)
        if offer == "free_moment" and (degraded or is_generic_checkin(text)):
            # Needs no model: a plain line from how long they worked and in what.
            text = compose_check_in(float(meta.get("minutes", 5) or 5),
                                    str(meta.get("process", "")),
                                    time.localtime(self.clock()).tm_hour,
                                    subject=str(meta.get("subject", "")),
                                    project=str(meta.get("project", "")))
            pathway, degraded = "offer", False
        elif degraded:
            self._model_unavailable(ev, decision)
            return

        if not text:
            return

        try:
            self.response_sink(text, {
                "addressed": to_ash,
                "likely": bool(getattr(decision, "likely", False)),
                "source": ev.source,
                "attention": decision.as_dict(),
                "pathway": pathway,
            })
        except Exception:
            logger.exception("Response sink failed")

        if to_ash:
            self.gate.note_user_turn()
            # The exchange is now open: the next sentence is read as the other
            # half of it, with no need to say the name again.
            address.conversation.note_ash_spoke(self.clock())
        elif self.reactions is not None:
            # Something was volunteered: watch what the user does with it.
            self.reactions.register(
                getattr(decision, "kind", "") or ev.source, getattr(decision, "ctx", ""),
                text, self.clock(),
                topic=str(meta.get("topic", "")),
                accept_context=str(meta.get("accept_context") or f"You had just said: {text}"))

    def _model_unavailable(self, ev, decision):
        """The brain had no real reply (the model is down or said nothing).

        Never spoken, and never about a remark nobody asked for. Someone who did
        ask is told once, quietly, then left alone until the notice interval has
        passed: repeating "I couldn't put together a response" aloud for every
        sentence of a conversation is worse than the outage.
        """
        self.stats["no_reply"] += 1
        now = self.clock()
        if not ev.addressed or now - self._last_no_reply_notice < self.cfg.no_reply_notice_s:
            logger.warning("No reply to give (%s)", "addressed" if ev.addressed else "unprompted")
            if ev.addressed:
                self.gate.note_user_turn()
            return
        self._last_no_reply_notice = now
        try:
            self.response_sink(NO_MODEL_TEXT, {
                "addressed": True, "source": ev.source, "attention": decision.as_dict(),
                "pathway": "degraded", "degraded": True,
            })
        except Exception:
            logger.exception("Response sink failed")
        self.gate.note_user_turn()

    # ------------------------------------------------------------------
    def _plan_active(self) -> bool:
        f = getattr(self.brain, "follower", None)
        return bool(f is not None and getattr(f, "active", False)
                    and hasattr(self.brain, "advance_plan"))

    def _drop_plan(self, reason: str):
        if self._plan_active():
            try:
                self.brain.follower.drop(reason)
            except Exception:
                logger.exception("Could not drop plan")

    def _advance_plan(self):
        """Take the next step of the brain's active plan and say what came of it."""
        if not self.cfg.follow_plans or not self._plan_active():
            return
        if self._user_busy >= 1.0:
            return                       # typing right now: wait, the plan keeps
        try:
            response = self.brain.advance_plan()
        except Exception:
            self.stats["errors"] += 1
            logger.exception("Plan step failed; dropping the plan")
            self._drop_plan("error")
            return
        if response is None:
            return
        self.stats["plan_steps"] += 1
        self._last_activity = time.time()
        text = (response.text or "").strip()[: self.cfg.max_response_chars]
        if not text:
            return
        try:
            self.response_sink(text, {
                "addressed": True,
                "source": "plan",
                "attention": None,
                "pathway": response.pathway.value,
                "plan": dict(getattr(response.trace, "plan", None) or {}),
            })
        except Exception:
            logger.exception("Response sink failed")

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
            "social": ({"learner": self.learner.status() if self.learner else None,
                        "reactions": self.reactions.status() if self.reactions else None,
                        "flow": self.flow.status() if self.flow else None}),
            "privacy": self.privacy.status(),
            "journal": self.journal.status(),
            "periphery": self.periphery.status(),
            "brain": self.brain.status(),
        }
