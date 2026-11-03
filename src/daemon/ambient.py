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
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

from src.senses import address
from src.senses.base import Modality, SensorEvent
from .attention import AttentionGate, Disposition
from .journal import Journal
from .privacy import PrivacyPolicy
from .freemoment import FreeMoments, ProviderActivity, default_moments
from .wondering import Environment, Knowledge, Wonderer, WonderingConfig
from .social import (FlowDetector, ReactionTracker, SocialLearner,
                     compose_check_in, is_generic_checkin, where_phrase)

try:
    from src.brain.memory_engine import is_failure_text
except Exception:                                        # pragma: no cover
    def is_failure_text(text) -> bool:
        return isinstance(text, str) and text.strip().startswith(("[LLM]", "Error calling"))

logger = logging.getLogger("ash.daemon.ambient")

# Said out loud, to someone who asked a real question, so: what is wrong and
# nothing else. No apology (it is not ASH's fault and saying sorry twice for
# the same outage is worse than the outage) and no jargon dressed up as one.
def clip_reply(text: str, limit: int) -> str:
    """Cut a reply to `limit` characters at the end of a sentence.

    This used to be text[:limit], which ended a long answer in the middle of
    a word -- "...represent the en" -- and that string was then spoken aloud.
    A sentence boundary if one falls in the back half of the allowance, else
    the last space with an ellipsis, so it never stops mid-word.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    # A list marker ("1.") is a full stop that is not the end of anything.
    ends = [m.end() for m in re.finditer(r"[.!?](?:[\"')\]]*)(?=\s|$)", head)
            if not re.search(r"(?:^|\s)\d{1,2}$", head[:m.start()])]
    if ends and ends[-1] >= limit * 0.4:
        return head[:ends[-1]].rstrip()
    return head.rsplit(" ", 1)[0].rstrip(" ,;:") + "..."


NO_MODEL_TEXT = "My model isn't answering right now, so I've got nothing for you on that."


@dataclass
class AmbientConfig:
    tick_seconds: float = 1.0
    queue_limit: int = 256
    sleep_after_idle_s: float = 25 * 60
    sleep_min_interval_s: float = 60 * 60
    # Longer than the model is allowed to take. At 120 this fired "ambient loop
    # unhealthy -- restarting" a dozen times in a morning, every time the
    # brain was simply inside a 60-100s generation on a CPU-bound model; the
    # restart could not do anything (the old thread was still running) and
    # the log filled with errors about a loop that was working.
    watchdog_stall_s: float = 360.0
    context_window_s: float = 1800.0
    max_response_chars: int = 600
    follow_plans: bool = True        # carry on a multi-step plan between ticks
    learn_social: bool = True        # learn from how volunteered remarks land
    flow_offers: bool = True         # offer a chat in free moments, help when stuck
    social_path: Optional[str] = None    # where the learned record lives (None: in memory)
    reaction_window_s: float = 120.0     # how long to wait for a reaction to a remark
    no_reply_notice_s: float = 600.0     # how often to say the model is down, at most
    # Say the free-moment check-in with the language model, rather than from
    # the facts. Off: it was a ~60s CPU generation that froze the whole
    # ambient loop (the watchdog restarted it at 13:16, 13:37 and 13:57 --
    # exactly the three check-ins that day) to produce lines like "Just
    # finished a bit of work in Opera, if you needed me." The composed ones
    # are specific, vary, and cost nothing.
    check_in_with_model: bool = False
    # How long a stretch of continuous activity has to be before a pause after
    # it is worth remarking on. It was 120s, which made "about 2 minutes of
    # work" a thing ASH said aloud four times in an hour and a half -- two
    # minutes in a chat window, two minutes of a song. Ten minutes is a stretch
    # of work. (A watchdog restart is not one of the costs; the check-in is
    # composed now, with no model call to stall the loop.)
    flow_min_burst_s: float = 600.0
    # Wondering: notice what is not known and ask, at a good moment. Off here
    # so a bare AmbientRuntime behaves as it always did; the daemon turns it
    # on from daemon.json ("wondering").
    wonder: bool = False
    wonder_cfg: Optional[Dict[str, Any]] = None
    wonder_path: Optional[str] = None        # state/wonder/wonders.json
    knowledge_path: Optional[str] = None     # state/wonder/knowledge.json
    # Free moments: choose among gap closure, a joke, a mood check, a note on
    # how ASH is running... Needs wonder for the first. Off on a bare runtime.
    moments: bool = False
    moments_cfg: Optional[Dict[str, Any]] = None


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
        self.flow: Optional[FlowDetector] = (
            FlowDetector(min_burst_s=self.cfg.flow_min_burst_s) if self.cfg.flow_offers else None)
        if self.cfg.learn_social:
            self.learner = getattr(self.gate, "learner", None) or SocialLearner(
                path=self.cfg.social_path)
            self.gate.learner = self.learner
            self.learner.clock = lambda: self.clock()
            self.reactions = ReactionTracker(self.learner, window_s=self.cfg.reaction_window_s)
        # Wondering: gaps noticed in what the user says, kept until a good moment.
        self.wonder: Optional[Wonderer] = None
        self.mood_fn: Optional[Callable[[], Optional[float]]] = None   # -1..1, or None
        self._free_until = 0.0
        self._wonder_inflight: Optional[str] = None
        if self.cfg.wonder:
            wc = WonderingConfig.from_dict(self.cfg.wonder_cfg)
            ent = getattr(self.ash, "entity_memory", None)
            user = str(getattr(self.brain, "user", "") or "")
            if user and user.lower() not in {x.lower() for x in wc.ignore_names}:
                wc.ignore_names = tuple(wc.ignore_names) + (user,)
            self.wonder = Wonderer(
                wc, Knowledge(self.cfg.knowledge_path, entity_manager=ent),
                path=self.cfg.wonder_path, clock=lambda: self.clock(),
                scrub=self.privacy.scrub)
        # Free moments: what to do with a pause, chosen among several things.
        self.moments: Optional[FreeMoments] = None
        self._moment_inflight: Optional[str] = None
        self._errors_told = 0
        if self.cfg.moments:
            mc = dict(self.cfg.moments_cfg or {})
            self.moments = default_moments(
                mc, wonderer=self.wonder, clock=lambda: self.clock(),
                jokes=bool(mc.get("jokes", True)), mood_check=bool(mc.get("mood_check", True)))
            self.moments.register(ProviderActivity(
                "system_assessment", self._system_note, cooldown_s=6 * 3600.0, weight=0.9))
        self.stats["wonders_asked"] = 0
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
            if idle_s is None:
                idle_s = self._idle_seconds()
            if self.flow is None:
                self._wonder_step(now, idle_s)
                return
            if idle_s is None:
                return                   # no way to tell when they pause: say nothing
            for offer in self.flow.poll(idle_s, now):
                if (offer.meta or {}).get("offer") == "free_moment":
                    # A pause after real work: the moment for whatever has
                    # been saved up. A question beats "you worked a while".
                    self._free_until = now + (self.wonder.cfg.free_moment_s if self.wonder else 180.0)
                    if self._moment_step(now, idle_s) or self._wonder_step(now, idle_s):
                        continue
                self.stats["offers"] += 1
                self.submit(offer)
            if not self._moment_step(now, idle_s):
                self._wonder_step(now, idle_s)
        except Exception:
            self.stats["errors"] += 1
            logger.exception("Social step failed")

    # ------------------------------------------------------------------
    # Free moments
    # ------------------------------------------------------------------
    def _system_note(self, env, now):
        """ASH's own state, as a thing to say in a free moment: only when
        something went wrong since it was last mentioned."""
        n = int(self.stats.get("errors", 0))
        if n <= self._errors_told:
            return None
        self._errors_told = n
        return ("I ran into some internal errors recently. Nothing you need to do, "
                "but the logs have the details.", 0.5)

    def _moment_step(self, now: float, idle_s: Optional[float]) -> bool:
        """In a free moment, pick the best thing to do with it and offer it.
        True if something was submitted."""
        m = self.moments
        if m is None or self._moment_inflight is not None or self._wonder_inflight is not None:
            return False
        try:
            env = self._wonder_env(now, idle_s)
            c = m.pick(env, now)
            if c is None:
                return False
            q = c.data.get("wonder")
            if q is not None:
                self._wonder_inflight = q.id
                self.stats["wonders_asked"] += 1
                meta = {"offer": "wonder", "offer_strength": round(0.7 + 0.15 * q.urgency, 3),
                        "say": c.text, "wonder_id": q.id, "topic": q.kind,
                        "accept_context": f"You had just asked the user: {c.text}"}
            else:
                self._moment_inflight = c.activity
                meta = {"offer": "moment", "offer_strength": round(0.5 + 0.2 * c.score, 3),
                        "say": c.text, "moment": c.activity, "topic": c.activity,
                        "accept_context": f"You had just said: {c.text}"}
            self.submit(SensorEvent(source="flow", modality=Modality.TEXT, text=c.text,
                                    salience=0.5, ts=now, persistable=False, meta=meta))
            return True
        except Exception:
            self.stats["errors"] += 1
            logger.exception("Free-moment step failed")
            return False

    # ------------------------------------------------------------------
    # Wondering
    # ------------------------------------------------------------------
    def _wonder_env(self, now: float, idle_s: Optional[float]) -> Environment:
        mood = None
        try:
            mood = self.mood_fn() if self.mood_fn else None
        except Exception:
            logger.exception("mood_fn failed")
        if mood is None and self.moments is not None:
            mood = self.moments.mood(now)       # what a mood check last heard
        flow = self.flow
        return Environment(
            idle_s=idle_s, muted=bool(self.gate.muted),
            quiet_hours=bool(self.gate._in_quiet_hours(now)), paused=bool(self.privacy.paused),
            conversation_open=address.conversation.openness(now) > 0.02,
            plan_active=self._plan_active(),
            free_moment=now < self._free_until, mood=mood,
            activity=(f"{getattr(flow, 'subject', '')} {getattr(flow, 'process', '')}".strip()
                      if flow is not None else ""))

    def _wonder_step(self, now: float, idle_s: Optional[float]) -> bool:
        """Submit the next question as an offer, if the moment allows. True if
        one was submitted."""
        w = self.wonder
        if w is None or self._wonder_inflight is not None:
            return False
        try:
            q = w.next_question(self._wonder_env(now, idle_s), now)
            if q is None:
                return False
            text = w.render(q, now)
            self._wonder_inflight = q.id
            self.stats["wonders_asked"] += 1
            self.submit(SensorEvent(
                source="flow", modality=Modality.TEXT, text=text, salience=0.6, ts=now,
                persistable=False,
                meta={"offer": "wonder", "offer_strength": round(0.7 + 0.15 * q.urgency, 3),
                      "say": text, "wonder_id": q.id, "topic": q.kind,
                      "accept_context": f"You had just asked the user: {text}"}))
            return True
        except Exception:
            self.stats["errors"] += 1
            logger.exception("Wonder step failed")
            return False

    def _wonder_hear(self, ev, now: float):
        """Read a user utterance. Returns the Reply if the event is spent."""
        w = self.wonder
        if w is None:
            return None
        meta = getattr(ev, "meta", None) or {}
        try:
            if meta.get("dismissed"):
                return w.hear("", now, dismissed=True)
            if getattr(ev, "source", "") == "flow" or not w.hears(getattr(ev, "source", "")):
                return None
            text = getattr(ev, "text", "") or ""
            rep = w.hear(text, now, addressed=bool(getattr(ev, "addressed", False)))
            if not rep.consumed:
                # Not an answer: learn what it states, note what it leaves open.
                rep.learned = w.observe(text, ev.source, now).learned
            self._wonder_learn(rep.learned)
            if rep.consumed:
                self._wonder_reaction(rep, now)
            return rep
        except Exception:
            logger.exception("Wondering failed on an utterance")
            return None

    def _wonder_reaction(self, rep, now: float):
        """The user's reply to a question is a reaction to it. An addressed
        reply was already read by the reaction tracker; one heard without
        the name was not, and would otherwise be counted as silence."""
        rt = self.reactions
        if rt is None or rep.outcome == "followup":
            return
        if rt.pending and rt.pending[-1].kind == "wonder":
            outcome = {"answered": "engaged", "unclear": "engaged", "deferred": "deferred",
                       }.get(rep.outcome, "rejected")
            rt._resolve(rt.pending.pop(), outcome, now)
        if rep.outcome in ("muted", "snoozed", "dismissed") and self.learner is not None:
            self.learner.snooze("wonder", now)

    def _wonder_learn(self, sentences):
        """What was learned also goes to the concept network, as a perception."""
        for sent in sentences or []:
            try:
                self.brain.observe(text=sent, source="wonder", salience=0.6)
            except Exception:
                logger.exception("brain.observe failed on a learned fact")

    def _wonder_say(self, ev, rep):
        try:
            self.response_sink(rep.say, {"addressed": True, "source": "wonder",
                                         "attention": None, "pathway": "wonder"})
            address.conversation.note_ash_spoke(self.clock(), asked=rep.outcome == "followup")
            self.gate.note_user_turn()
        except Exception:
            logger.exception("Response sink failed")

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

        # A reply to a question ASH asked, or a request about the questions
        # ("not now", "stop asking"), is spent here and not run as a new query.
        if self.wonder is not None:
            rep = self._wonder_hear(ev, now)
            if rep is not None and rep.consumed:
                self.stats["wonders_answered"] = self.stats.get("wonders_answered", 0) + 1
                if rep.say:
                    self._wonder_say(ev, rep)
                return

        # An answer to a free-moment remark (how are you?) is spent the same way.
        if self.moments is not None and getattr(ev, "source", "") != "flow":
            try:
                said = self.moments.hear(getattr(ev, "text", "") or "", now)
            except Exception:
                said = None
            if said:
                self._wonder_say(ev, type("R", (), {"say": said, "outcome": "moment"})())
                return

        novelty = 0.5
        try:
            drives = self.brain.homeostasis.drives
            decision = self.gate.decide(ev, drives, novelty=novelty,
                                        user_busy=self._user_busy, now=now)
        except Exception:
            logger.exception("Attention gate failed; defaulting to observe")
            decision = None

        if (ev.meta or {}).get("offer") == "moment" and (
                decision is None or decision.disposition != Disposition.RESPOND):
            self._moment_inflight = None         # not let through: nothing was said
        wid = (ev.meta or {}).get("wonder_id") if (ev.meta or {}).get("offer") == "wonder" else None
        if wid and (decision is None or decision.disposition != Disposition.RESPOND):
            # The gate would not let the question through: keep it, try later.
            self._wonder_inflight = None
            if self.wonder is not None:
                self.wonder.note_held(wid, now)

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
        try:
            self._respond(ev, decision)
        finally:
            if (ev.meta or {}).get("offer") == "wonder":
                self._wonder_inflight = None
            elif (ev.meta or {}).get("offer") == "moment":
                self._moment_inflight = None

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
        elif offer in ("wonder", "moment") and meta.get("say"):
            say = str(meta["say"])
            prompt = ""
        elif offer == "free_moment" and not self.cfg.check_in_with_model:
            # From the facts, no model: see AmbientConfig.check_in_with_model.
            say = compose_check_in(float(meta.get("minutes", 5) or 5),
                                   str(meta.get("process", "")),
                                   time.localtime(self.clock()).tm_hour,
                                   subject=str(meta.get("subject", "")),
                                   project=str(meta.get("project", "")))
            prompt = ""
        elif offer == "free_moment":
            # The subject ("brain.py") says more about the stretch of work
            # than the app ("VS Code") ever does, so it leads. It is not
            # repeated in the fact line, which already carries it.
            where = where_phrase(str(meta.get("process", "")),
                                 str(meta.get("subject", "")),
                                 str(meta.get("project", ""))).strip()
            fact = ev.text if (not where or where in ev.text) else f"{ev.text} ({where})"
            prompt = (
                f"[FREE MOMENT -- the user did not ask you anything. Fact: {fact}. "
                f"Reply with ONE short sentence, under 25 "
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
            text = clip_reply(say, self.cfg.max_response_chars)
        else:
            try:
                image = ev.data if getattr(ev.modality, "value", ev.modality) == "vision" else None
                if not to_ash and hasattr(self.ash, "volunteer"):
                    # Something ASH is saying on its own initiative is not a
                    # question from the user and must not be stored, routed
                    # or learned from as one. See Brain.volunteer.
                    response = self.ash.volunteer(prompt, context=context)
                else:
                    response = self.ash.think(prompt, image=image, context=context)
                text = clip_reply(response.text, self.cfg.max_response_chars)
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

        if offer == "wonder":
            self._wonder_inflight = None
            if self.wonder is not None:
                self.wonder.mark_asked(str(meta.get("wonder_id", "")), self.clock())
            if self.moments is not None:
                self.moments.mark_spoken("gap_closure", self.clock())
        elif offer == "moment":
            self._moment_inflight = None
            if self.moments is not None:
                self.moments.mark_spoken(str(meta.get("moment", "")), self.clock())

        if to_ash:
            self.gate.note_user_turn()
            # The exchange is now open: the next sentence is read as the other
            # half of it, with no need to say the name again.
            address.conversation.note_ash_spoke(self.clock(), asked=text.rstrip().endswith("?"))
        elif self.reactions is not None:
            # Something was volunteered: watch what the user does with it.
            # And if it asked something, the next "yeah" is an answer to it
            # rather than a noise made to show someone is listening.
            address.conversation.note_ash_spoke(
                self.clock(), asked=bool(offer) or text.rstrip().endswith("?"))
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

        "Someone who did ask" includes route 1b. This keyed on ev.addressed,
        which is only the *certain* half, so a probable question -- "what time
        is it", the whole reason that route exists -- got complete silence
        when the model was down. Being asked and saying nothing is the failure
        the route was added to stop; it should not come back the moment Ollama
        does not answer.
        """
        self.stats["no_reply"] += 1
        now = self.clock()
        asked = bool(getattr(ev, "addressed", False)
                     or getattr(decision, "likely", False))
        if not asked or now - self._last_no_reply_notice < self.cfg.no_reply_notice_s:
            logger.warning("No reply to give (%s)", "asked" if asked else "unprompted")
            if asked:
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
            "wondering": self.wonder.status() if self.wonder else None,
            "free_moments": self.moments.status() if self.moments else None,
            "privacy": self.privacy.status(),
            "journal": self.journal.status(),
            "periphery": self.periphery.status(),
            "brain": self.brain.status(),
        }
