"""
src/face/bridge.py

When each animation plays. This is the "used when it makes sense" half.

The face is driven by state ASH already produces -- no new signals were
invented for it. Three sources, in descending authority:

  EVENTS      discrete things that just happened: a veto, a tool failure, an
              addressed utterance, a critical alert. These map to one-shot
              animations and are the only ones that preempt.

  PATHWAY     which cognitive pathway a turn took. SLOW means the Executive
              refused to let a reflex through, which is *exactly* the moment a
              person would look away and think -- so `thinking` plays for the
              duration of the slow path and stops when the answer lands. FAST
              and REFLEX are too quick to animate; trying makes the face
              flicker.

  DRIVES      the continuous baseline, handled by FacePlayer.set_drives, plus
              a coarse mood animation when a drive sits at an extreme long
              enough to be a real state rather than a spike.

The hysteresis matters more than the mapping. Emotions in the engine move on
every telemetry update, and a face that re-triggers on each one is a strobe.
`_mood_hold` forces a mood to persist for a minimum time and to clear a
margin before switching, which turns a jittery signal into something that
looks like a temperament.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("ash.face.bridge")

# A mood must stay put this long before the face admits to a different one.
MOOD_MIN_HOLD_S = 6.0
# And must beat the incumbent by this margin to take over.
MOOD_SWITCH_MARGIN = 1.2


class FaceBridge:
    """Translates ASH's internal state into animation requests."""

    def __init__(self, player, enabled: bool = True, sounds: bool = False):
        from .sounds import FaceSounds
        self.player = player
        self.enabled = enabled
        self.sounds = FaceSounds(enabled=sounds)        # a chime with some expressions (off by default)
        self._mood: Optional[str] = None
        self._mood_since: float = 0.0
        self._thinking_since: Optional[float] = None
        self._quiet = False
        self._away = False
        self.counts: Dict[str, int] = {}
        from .moods import IdleLife
        self.idle_life = IdleLife()
        self._speaking = False
        self._owed = None
        self._felt = None
        self._last_user_t = time.time()

    # ------------------------------------------------------------------
    def _play(self, name: str, force: bool = False):
        if not self.enabled:
            return
        if self.player.play(name, force=force):
            self.counts[name] = self.counts.get(name, 0) + 1

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def on_boot(self):
        self._play("boot", force=True)

    def on_shutdown(self):
        self._play("shutdown", force=True)

    def on_addressed(self):
        """Someone spoke to ASH. Look up and hold still -- the stillness is
        what communicates attention."""
        self._last_user_t = time.time()
        self._play("listening")

    def on_thinking_start(self):
        self._thinking_since = time.time()
        self._play("thinking")

    def on_thinking_end(self, tool_success: Optional[bool] = None):
        self._thinking_since = None
        # `thinking` loops at a higher priority than either reaction below, so
        # it has to be ended before one can play -- otherwise the face keeps
        # thinking, lids half down, long after the answer has landed.
        self._end_loop("thinking", "listening", "processing")
        if tool_success is False:
            self._play("confused")
        else:
            self._play("acknowledge")

    def _end_loop(self, *names: str):
        cur = self.player.current
        if cur is not None and cur.name in names:
            self.player.stop()

    def shown_mood(self, now: Optional[float] = None) -> Optional[str]:
        """The feeling the face is wearing now, for the console's tint: a reaction
        (thanks -> love) for half a minute, else the slower drive/emotion mood."""
        now = time.time() if now is None else now
        if self._felt and now - self._felt[1] < 30.0:
            return self._felt[0]
        return self._mood

    # What the face does when something happens to ASH itself (a sense switched,
    # privacy mode): small, and only from animations that exist.
    _EVENTS = {
        ("microphone", True): "curious", ("microphone", False): "sleepy",
        ("camera", True): "surprised", ("camera", False): "sleepy",
        ("screenshot", True): "curious", ("screenshot", False): "sleepy",
        ("privacy", True): "sleepy", ("privacy", False): "happy",
    }

    def on_event(self, what: str, on: bool):
        anim = self._EVENTS.get((what, bool(on)))
        if anim:
            self._play(anim)
            self.idle_life.reset()

    def on_user_text(self, text: str):
        """The user said something with a feeling in it: the face answers before
        the voice does (thanks -> love, a laugh -> very_happy, "wow" -> surprised)."""
        from .moods import reaction_to
        anim = reaction_to(text)
        if anim:
            self._play(anim)
            self._felt = (anim, time.time())
            self._owed = (anim, time.time())     # and again once the reply lands, which would nod over it
            self.idle_life.reset()

    def on_idle_tick(self, now: float, idle_s: float, hour: Optional[int] = None):
        """Nobody is talking: now and then look around, double-blink, wink."""
        if not self.enabled or self._quiet or self._away:
            return
        busy = self._thinking_since is not None or self._speaking
        anim = self.idle_life.step(now, idle_s, busy=busy, hour=hour)
        if anim:
            self._play(anim)

    def on_speaking(self, on: bool):
        self._speaking = bool(on)
        if on:
            self._play("speaking")
        elif self.player.current and self.player.current.name == "speaking":
            self.player.stop()

    def on_veto(self, vetoed_by=None):
        """The Executive blocked an action. The face says no before the
        sentence does, which is the whole reason to have one."""
        logger.debug("Face: refuse (vetoed by %s)", vetoed_by)
        self._play("refuse", force=True)

    def on_tool_failure(self):
        self._play("confused")

    def on_error(self):
        self._play("error", force=True)

    def on_alert(self, urgency: float = 1.0):
        self._play("alert" if urgency >= 0.85 else "surprised", force=urgency >= 0.9)

    def on_deadlock(self):
        """Board couldn't decide, so ASH is about to ask a clarifying
        question. Look the part before it does."""
        self._play("confused")

    def on_sleep(self, entering: bool):
        self._play("sleeping" if entering else "boot")

    def on_user_away(self, away: bool):
        self._away = away
        self.player.set_idle("sleepy" if away or self._quiet else "idle")

    def on_quiet_hours(self, quiet: bool):
        """In quiet hours the face settles into sleepy, and wakes after."""
        if quiet == self._quiet:
            return
        self._quiet = quiet
        self.player.set_idle("sleepy" if quiet or self._away else "idle")

    def on_line(self, text: str, meta: Dict[str, Any]):
        """Something ASH just said: the face matches what kind of line it was.
        Questions look curious, alerts alert, good news happy, a win very
        happy, a correction confused, a plain acknowledgement a nod."""
        anim = self.expression_for(text, meta or {})
        owed, self._owed = self._owed, None
        if owed and (meta or {}).get("addressed") and time.time() - owed[1] < 60:
            anim = owed[0]                       # "thank you" is answered with love, not a nod
        if anim:
            self._play(anim)
            try:
                self.sounds.maybe(anim, volunteered=not (meta or {}).get("addressed"), quiet=self._quiet)
            except Exception:
                logger.debug("Face sound failed", exc_info=True)

    @staticmethod
    def expression_for(text: str, meta: Dict[str, Any]) -> Optional[str]:
        pathway = str(meta.get("pathway") or "")
        outcome = str(meta.get("outcome") or "")
        att = meta.get("attention") or {}
        kind = str(att.get("kind") or "") if isinstance(att, dict) else ""
        t = (text or "").strip()
        if pathway in ("reminder", "watch"):
            return "alert"
        if outcome == "play":
            if t.startswith("Yes!"):
                return "very_happy"
            if t.startswith(("Hm, I have", "Higher.", "Lower.")):
                return "skeptical"
            return "happy" if "game over" in t.lower() or "agreed on" in t else "curious"
        if outcome == "correction" or (outcome == "followup" and not t.endswith("?")):
            return "confused"
        if kind == "wonder" or outcome in ("followup", "batch") or (not meta.get("addressed") and t.endswith("?")):
            return "curious"
        if kind in ("briefing", "moment"):
            return "happy"
        if outcome in ("answered", "desktop", "research", "briefing"):
            return "acknowledge"
        return None

    # ------------------------------------------------------------------
    # Pathway
    # ------------------------------------------------------------------
    def on_pathway(self, pathway: str, tool_success: Optional[bool] = None,
                   vetoed_by=None, deadlocked: bool = False):
        """Called once per completed cognitive cycle."""
        # The cycle is over, so whatever it looped (thinking, listening) ends
        # here; every reaction below outranks idle but not those loops.
        self._end_loop("thinking", "listening", "processing")
        self._thinking_since = None            # the cycle is over, whatever path it took
        if vetoed_by:
            self.on_veto(vetoed_by)
            return
        if deadlocked:
            self.on_deadlock()
            return
        if tool_success is False:
            self.on_tool_failure()
            return

        # FAST and REFLEX finish in milliseconds. Animating them produces a
        # flicker, not an expression, so they get nothing and the idle or
        # mood animation continues undisturbed.
        if pathway in ("slow", "escalated"):
            self.on_thinking_end(tool_success)

    # ------------------------------------------------------------------
    # Drives and emotions
    # ------------------------------------------------------------------
    def on_drives(self, drives):
        """Continuous modulation plus, occasionally, a mood animation."""
        if not self.enabled:
            return
        self.player.set_drives(drives)

        candidate, strength = self._mood_from_drives(drives)
        now = time.time()

        if candidate is None:
            return
        if self._mood == candidate:
            return
        if self._mood is not None and (now - self._mood_since) < MOOD_MIN_HOLD_S:
            return
        if strength < MOOD_SWITCH_MARGIN:
            return

        self._mood = candidate
        self._mood_since = now
        self._play(candidate)

    @staticmethod
    def _mood_from_drives(drives):
        """Coarse mood from the drive vector. Returns (name, strength) where
        strength must clear MOOD_SWITCH_MARGIN to trigger a change."""
        try:
            f = float(drives.fatigue)
            c = float(drives.curiosity)
            ca = float(drives.caution)
            u = float(drives.urgency)
        except Exception:
            return None, 0.0

        scores = {
            "sleepy": 3.0 * f,
            "curious": 2.4 * max(0.0, c - 0.55),
            "skeptical": 2.4 * max(0.0, ca - 0.6),
            "annoyed": 2.2 * max(0.0, u - 0.7),
        }
        name = max(scores, key=scores.get)
        return (name, scores[name]) if scores[name] > 0 else (None, 0.0)

    def on_emotions(self, emotions: Dict[str, float]):
        """Optional: drive the face from the raw emotion vector instead of
        drives. Higher fidelity, more jitter -- use one or the other."""
        if not emotions:
            return
        g = lambda k: float(emotions.get(k, 0.0))  # noqa: E731
        scores = {
            "very_happy": g("happy") * 0.7 + g("excitement") * 0.5,
            "happy": g("happy") * 0.6 + g("trust") * 0.3,
            "sad": g("sad") * 0.9,
            "angry": g("anger") * 0.9 + g("frustration") * 0.6,
            "annoyed": g("frustration") * 0.8 + g("boredom") * 0.4,
            "surprised": g("surprise") * 1.0,
            "curious": g("curiosity") * 0.8,
            "skeptical": g("anxiety") * 0.6 + g("distrust") * 0.7,
            "love": g("love") * 1.0,
        }
        name = max(scores, key=scores.get)
        # 10-point emotion scale; below 4 there is no mood worth showing.
        if scores[name] < 4.0:
            return
        now = time.time()
        if self._mood == name or (now - self._mood_since) < MOOD_MIN_HOLD_S:
            return
        self._mood, self._mood_since = name, now
        self._play(name)

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mood": self._mood,
            "mood_age_s": round(time.time() - self._mood_since, 1) if self._mood else None,
            "triggered": dict(sorted(self.counts.items(), key=lambda kv: -kv[1])[:12]),
            "player": self.player.status(),
        }


# ----------------------------------------------------------------------
def attach_to_daemon(runtime, player, enabled: bool = True, sounds: bool = False) -> FaceBridge:
    """Wire a FaceBridge into a running AmbientRuntime.

    Done by wrapping rather than by editing the daemon: the ambient loop has
    no idea a face exists, and if the face throws it cannot take the loop with
    it. Every hook is individually guarded for the same reason -- an always-on
    process must not die because a display went away.
    """
    bridge = FaceBridge(player, enabled=enabled, sounds=sounds)

    original_handle = runtime._handle
    original_respond = runtime._respond
    original_sleep = runtime._sleep_phase
    original_sink = runtime.response_sink
    original_social = runtime._social_step

    def sink(text, meta):
        try:
            bridge.on_line(text, meta)
        except Exception:
            logger.exception("Face hook (line) failed")
        return original_sink(text, meta)

    def social_step(now, idle_s=None):
        try:
            bridge.on_quiet_hours(bool(runtime.gate._in_quiet_hours(now)) or bool(getattr(runtime.gate, "muted", False)))
        except Exception:
            logger.exception("Face hook (quiet hours) failed")
        try:
            # Quiet since someone last spoke to ASH -- not input idleness: a person
            # reading at the desk is present, and is who the face is alive for.
            bridge.on_idle_tick(now, now - bridge._last_user_t, hour=time.localtime(now).tm_hour)
        except Exception:
            logger.exception("Face hook (idle life) failed")
        return original_social(now, idle_s)

    def handle(ev):
        try:
            if getattr(ev, "addressed", False):
                bridge.on_addressed()
                bridge.on_user_text(str(getattr(ev, "text", "") or ""))
                bridge._last_user_t = runtime.clock()
            elif getattr(ev, "urgency", 0.0) >= 0.85:
                bridge.on_alert(ev.urgency)
            elif ev.source == "idle":
                away = bool((ev.meta or {}).get("away"))
                bridge.on_user_away(away)
        except Exception:
            logger.exception("Face hook (handle) failed")
        return original_handle(ev)

    def respond(ev, decision):
        try:
            bridge.on_thinking_start()
        except Exception:
            logger.exception("Face hook (thinking) failed")
        try:
            return original_respond(ev, decision)
        finally:
            try:
                trace = runtime.ash.explain_last() or {}
                verdict = trace.get("verdict") or {}
                bridge.on_pathway(
                    trace.get("pathway", "slow"),
                    tool_success=None,
                    vetoed_by=verdict.get("vetoed_by"),
                    deadlocked=verdict.get("deadlocked", False),
                )
                bridge.on_drives(runtime.brain.homeostasis.drives)
            except Exception:
                logger.exception("Face hook (pathway) failed")
            finally:
                bridge._thinking_since = None       # answered (or failed): the face is no longer "busy thinking"

    def sleep_phase():
        try:
            bridge.on_sleep(True)
        except Exception:
            pass
        try:
            return original_sleep()
        finally:
            try:
                bridge.on_sleep(False)
            except Exception:
                pass

    original_set_sensor = runtime.set_sensor
    original_pause, original_resume = runtime.privacy.pause, runtime.privacy.resume

    def set_sensor(name, on):
        res = original_set_sensor(name, on)
        try:
            if res and res[0]:
                bridge.on_event(name, on)
        except Exception:
            logger.exception("Face hook (sensor) failed")
        return res

    def pause(*a, **k):
        r = original_pause(*a, **k)
        try:
            bridge.on_event("privacy", True)
        except Exception:
            logger.exception("Face hook (privacy) failed")
        return r

    def resume(*a, **k):
        r = original_resume(*a, **k)
        try:
            bridge.on_event("privacy", False)
        except Exception:
            logger.exception("Face hook (privacy) failed")
        return r

    runtime.set_sensor = set_sensor
    runtime.privacy.pause = pause
    runtime.privacy.resume = resume
    runtime._handle = handle
    runtime._respond = respond
    runtime._sleep_phase = sleep_phase
    runtime.response_sink = sink
    runtime._social_step = social_step

    bridge._last_user_t = runtime.clock()
    bridge.on_boot()
    logger.info("Face bridge attached to ambient runtime")
    return bridge
