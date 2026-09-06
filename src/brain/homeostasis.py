"""
src/brain/homeostasis.py

Homeostatic Modulator  (the limbic box at the top of the diagram).

This sits *above* everything and broadcasts downward. It does not decide
anything; it changes the price of deciding.

The existing `EmotionEngine` (tools/emo.py) already does the hard part --
deterministic, telemetry-driven emotion with time-decay toward a slow mood
baseline. This module does not replace it. It wraps it and adds the piece the
diagram implies but the old runtime never had: a compression of that 23-dim
emotion vector into six **latent drives** that actually gate cognition.

Why the extra layer instead of feeding emotions straight in?

  ASH's founding constraint is "emotion influences tone, not decisions."
  Drives keep that true. `frustration` never selects a tool. But frustration
  raising `urgency` and lowering `effort_budget` -- so the brain reaches for
  the reflex path sooner and stops writing four-paragraph answers to someone
  who is clearly annoyed -- is exactly what a nervous system does, and it
  never touches *which* action is correct, only how much deliberation gets
  purchased before committing to it.

Homeostasis proper: each drive has a setpoint and is pulled back toward it
every cycle. Sustained load raises `fatigue`, which suppresses
`effort_budget` -- so a long hammering session degrades into terser, faster
responses rather than burning full deliberation on turn 200. Idle time
recovers it.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional

from .signals import Drives

logger = logging.getLogger("ash.brain.homeostasis")

# Setpoints each drive relaxes toward in the absence of stimulation.
SETPOINTS: Dict[str, float] = {
    "urgency": 0.30,
    "caution": 0.35,
    "curiosity": 0.40,
    "effort_budget": 0.55,
    "social": 0.50,
    "fatigue": 0.00,
}

# Per-cycle relaxation rate toward setpoint (0 = frozen, 1 = instant reset).
RELAX = 0.25

# Time constant for fatigue recovery during idle, in seconds.
FATIGUE_TAU = 900.0


def _c01(v: float) -> float:
    return max(0.0, min(1.0, v))


class HomeostaticModulator:
    """Owns the emotion engine and translates its output into drives."""

    def __init__(self, emotion_engine, drives: Optional[Drives] = None):
        self.emotion = emotion_engine
        self.drives = drives or Drives(**SETPOINTS)
        self._last_update: Optional[float] = None
        self._load_ema = 0.0          # rolling cost of recent cycles
        self.last_emotion_result = None

    # ------------------------------------------------------------------
    # Drive derivation
    # ------------------------------------------------------------------
    def _derive(self, emo: Dict[str, float], meta: Dict[str, Any]) -> Dict[str, float]:
        """Map the 0..10 emotion vector onto 0..1 drive targets.

        Every coefficient here is a deliberate, inspectable design choice --
        no learned weights, no LLM in the loop. Read it as a wiring diagram.
        """
        g = lambda k: emo.get(k, 0.0) / 10.0  # noqa: E731

        frustration = g("frustration")
        anxiety = g("anxiety")
        confidence = g("confidence")
        curiosity_e = g("curiosity")
        boredom = g("boredom")
        excitement = g("excitement")
        trust = g("trust")
        happy = g("happy")
        sad = g("sad")
        surprise = g("surprise")
        fear = g("fear")

        fail_streak = float(meta.get("fail_streak", 0) or 0)
        cadence = meta.get("cadence", "normal")

        # URGENCY -- pressure to answer now. Frustration and excitement push
        # it up; rapid-fire cadence pushes it up hard (the human is impatient).
        urgency = 0.20 + 0.45 * frustration + 0.20 * excitement + 0.15 * anxiety
        if cadence in ("rapid", "fast", "burst"):
            urgency += 0.20

        # CAUTION -- appetite for irreversible action. Anxiety and repeated
        # failure raise it; confidence and trust lower it. This is the drive
        # the VLA channel reads before touching an actuator.
        caution = 0.25 + 0.40 * anxiety + 0.30 * fear + 0.12 * min(fail_streak, 3.0) - 0.30 * confidence - 0.10 * trust

        # CURIOSITY -- explore vs. exploit. Boredom is the strongest driver:
        # a bored brain deliberately spends slow-path cycles it doesn't
        # strictly need, which is how it finds better policies.
        curiosity = 0.25 + 0.40 * curiosity_e + 0.30 * boredom + 0.15 * surprise - 0.20 * anxiety

        # EFFORT BUDGET -- how much compute this turn is worth. Confidence
        # buys the *right* to think less, not more. Frustration cuts it:
        # an annoyed user wants an answer, not a treatise.
        effort = 0.50 + 0.25 * curiosity_e + 0.15 * trust - 0.30 * frustration - 0.20 * boredom

        # SOCIAL -- how much of the response should attend to the human
        # rather than the task.
        social = 0.35 + 0.30 * happy + 0.30 * sad + 0.25 * g("love") + 0.20 * frustration

        return {
            "urgency": _c01(urgency),
            "caution": _c01(caution),
            "curiosity": _c01(curiosity),
            "effort_budget": _c01(effort),
            "social": _c01(social),
        }

    # ------------------------------------------------------------------
    # Fatigue: the actual homeostatic loop
    # ------------------------------------------------------------------
    def _update_fatigue(self, dt: Optional[float], cycle_cost: float) -> float:
        # Exponential recovery while idle.
        if dt is not None and dt > 0:
            self.drives.fatigue *= math.exp(-dt / FATIGUE_TAU)
        # Accumulation from work done. Slow-path turns cost far more.
        self._load_ema = 0.85 * self._load_ema + 0.15 * cycle_cost
        self.drives.fatigue = _c01(self.drives.fatigue + 0.04 * cycle_cost)
        return self.drives.fatigue

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def update(self, signals, cycle_cost: float = 0.5) -> Drives:
        """Run the emotion engine on this turn's telemetry, then relax the
        drive vector toward the emotion-derived targets.

        `cycle_cost` is 0..1: roughly, how expensive the *previous* cycle was
        (reflex ~0.05, fast ~0.2, slow ~1.0). It only feeds fatigue.
        """
        now = time.time()
        dt = None if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now

        try:
            result = self.emotion.update(signals)
        except Exception:
            logger.exception("EmotionEngine.update failed; holding previous drives")
            return self.drives

        self.last_emotion_result = result
        meta = {
            "fail_streak": getattr(result, "fail_streak", 0),
            "success_streak": getattr(result, "success_streak", 0),
            "cadence": getattr(result, "cadence", "normal"),
        }
        targets = self._derive(result.emotions, meta)

        # Relax current drives toward targets, then toward setpoints. Two-stage
        # so drives track emotion without ever pinning to an extreme.
        for k, target in targets.items():
            cur = getattr(self.drives, k)
            moved = cur + RELAX * (target - cur)
            moved = moved + 0.10 * (SETPOINTS[k] - moved)
            setattr(self.drives, k, _c01(moved))

        fatigue = self._update_fatigue(dt, cycle_cost)
        # Fatigue directly taxes the effort budget -- this is the feedback
        # loop that makes long sessions get terser on their own.
        self.drives.effort_budget = _c01(self.drives.effort_budget * (1.0 - 0.5 * fatigue))

        return self.drives

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "drives": self.drives.as_dict(),
            "load_ema": round(self._load_ema, 3),
            "emotions": dict(getattr(self.emotion, "emotion", {}) or {}),
        }

    def recover(self, seconds: float):
        """Called by the consolidation/sleep phase: idle time repays fatigue
        and lets drives settle all the way back to setpoint."""
        self.drives.fatigue = _c01(self.drives.fatigue * math.exp(-seconds / FATIGUE_TAU))
        for k, sp in SETPOINTS.items():
            if k == "fatigue":
                continue
            cur = getattr(self.drives, k)
            setattr(self.drives, k, _c01(cur + 0.5 * (sp - cur)))
        logger.info("Homeostasis: recovery pass, fatigue=%.3f", self.drives.fatigue)
