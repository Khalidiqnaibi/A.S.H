"""
src/concepts/neuromod.py

Neuromodulators: the signals that decide WHICH system learns, WHEN, and HOW
MUCH.

This is the piece that was missing. ASH had drives (how much thinking to buy)
and it had plasticity (how much to rewire), but nothing connecting them. In a
brain those are the same machinery: a handful of diffuse chemical signals
broadcast from small nuclei, each gating a different aspect of learning.

Four of them matter here, and each has a job that maps onto something ASH
already computes:

    ACETYLCHOLINE (ACh) -- encode vs consolidate.
        High ACh = the world is novel and uncertain, so favour ENCODING:
        hippocampus dominant, high learning rate, suppress cortical
        interference. Low ACh (quiet, familiar, asleep) = favour
        CONSOLIDATION: replay hippocampus into cortex.
        This is a genuine switch, not a dial: the same network cannot encode
        and consolidate at the same time without one corrupting the other.
        Driven by: novelty, and inversely by familiarity.

    DOPAMINE (DA) -- the third factor. Reward prediction error.
        Gates which associations get stamped in. Already present in ASH as
        the System 1 critic's TD error; this module just gives it a name and
        a decay so it can act as a broadcast signal rather than a
        per-update argument.

    NORADRENALINE (NE) -- gain and surprise.
        Multiplies plasticity magnitude globally. A surprising moment is
        learned more strongly than a boring one, at every synapse, regardless
        of reward. This is why you remember where you were during something
        shocking. Driven by: prediction error and event urgency.

    SEROTONIN (5-HT) -- patience and the discount rate.
        Low 5-HT biases toward immediate, habitual responses; high 5-HT
        supports waiting and deliberation. Maps onto ASH's fast/slow gate.
        Driven by: fatigue (inversely) and mood.

Why this matters more than it looks
-----------------------------------
Without a neuromodulatory layer, every experience is learned equally hard.
That is the main reason naive continual-learning systems degrade: they spend
the same plasticity budget on the ten-thousandth window switch as on the one
sentence that actually mattered. Gating by ACh/NE means ASH's limited
plasticity gets spent on the novel and the surprising, which is exactly where
the information is.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger("ash.concepts.neuromod")


@dataclass
class NeuroState:
    """Current levels. All 0..1 except dopamine, which is signed."""

    ach: float = 0.5        # encode (high) vs consolidate (low)
    da: float = 0.0         # reward prediction error, -1..1
    ne: float = 0.3         # global plasticity gain
    serotonin: float = 0.5  # patience / deliberation bias

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in asdict(self).items()}


class Neuromodulators:
    """Derives modulator levels from signals ASH already produces."""

    # Time constants, seconds. ACh tracks the environment over minutes;
    # dopamine is phasic and decays in seconds; NE sits between.
    TAU_ACH = 240.0
    TAU_DA = 8.0
    TAU_NE = 45.0
    TAU_5HT = 900.0

    def __init__(self):
        self.state = NeuroState()
        self._last = time.time()
        self._novelty_ema = 0.5
        self.history: list = []

    # ------------------------------------------------------------------
    def _decay_toward(self, current: float, target: float, tau: float,
                      dt: float) -> float:
        if tau <= 0:
            return target
        alpha = 1.0 - math.exp(-dt / tau)
        return current + alpha * (target - current)

    # Each update represents at least this much "modulator time", even when
    # several arrive in the same millisecond. Neuromodulator release is
    # driven by events, not only by the clock, and keying purely to wall time
    # meant a burst of cycles moved nothing at all.
    MIN_EVENT_DT = 1.5

    def update(self, novelty: float = 0.5, surprise: float = 0.0,
               urgency: float = 0.3, fatigue: float = 0.0,
               td_error: float = 0.0, asleep: bool = False,
               now: Optional[float] = None, dt: Optional[float] = None) -> NeuroState:
        now = now or time.time()
        if dt is None:
            dt = max(self.MIN_EVENT_DT, now - self._last)
        self._last = now
        s = self.state

        self._novelty_ema = 0.9 * self._novelty_ema + 0.1 * float(novelty)

        # --- ACh: encode vs consolidate -----------------------------------
        # Sleep drives it to floor. That is not a convenience -- low ACh is
        # what physically permits hippocampal output to drive cortex during
        # slow-wave sleep, and high ACh during waking is what prevents it.
        # The encode/consolidate switch IS the ACh level.
        target_ach = 0.05 if asleep else float(min(1.0, 0.25 + 0.75 * self._novelty_ema))
        s.ach = self._decay_toward(s.ach, target_ach, self.TAU_ACH if not asleep else 20.0, dt)

        # --- DA: phasic reward prediction error --------------------------
        s.da = self._decay_toward(s.da, 0.0, self.TAU_DA, dt)
        if abs(td_error) > 1e-3:
            s.da = float(max(-1.0, min(1.0, s.da + td_error)))

        # --- NE: gain from surprise and urgency ---------------------------
        target_ne = float(min(1.0, 0.15 + 0.6 * surprise + 0.35 * urgency))
        s.ne = self._decay_toward(s.ne, target_ne, self.TAU_NE, dt)

        # --- 5-HT: patience ------------------------------------------------
        target_5ht = float(max(0.0, min(1.0, 0.6 - 0.5 * fatigue)))
        s.serotonin = self._decay_toward(s.serotonin, target_5ht, self.TAU_5HT, dt)

        if len(self.history) < 5000:
            self.history.append((now, s.ach, s.da, s.ne))
        return s

    # ------------------------------------------------------------------
    # Gates. Everything downstream asks these rather than reading levels.
    # ------------------------------------------------------------------
    def encoding_gain(self) -> float:
        """Multiplier on hippocampal write strength.

        High ACh means the world is unfamiliar, so store aggressively. This
        is why the first day in a new place produces vivid memories and the
        hundredth produces almost none: the encoder is turned down, not the
        camera.
        """
        return 0.3 + 1.4 * self.state.ach

    def cortical_gain(self) -> float:
        """Multiplier on slow (cortical) plasticity.

        Deliberately the inverse of encoding gain. Cortex learning while ACh
        is high is how you get interference -- the new episode smearing into
        the existing semantic structure before anything has verified it was
        worth keeping.
        """
        return 0.15 + 0.85 * (1.0 - self.state.ach)

    def plasticity_gain(self) -> float:
        """Global multiplier from NE. Applies to every synapse."""
        return 0.4 + 1.6 * self.state.ne

    def reward_signal(self) -> float:
        return self.state.da

    def prefers_deliberation(self) -> float:
        """0..1 bias toward the slow path. Read by the Executive's gate."""
        return float(max(0.0, min(1.0, 0.5 * self.state.serotonin
                                  + 0.5 * self.state.ach)))

    def is_consolidating(self) -> bool:
        return self.state.ach < 0.2

    def status(self) -> Dict[str, Any]:
        return {
            "levels": self.state.as_dict(),
            "mode": "consolidate" if self.is_consolidating() else "encode",
            "encoding_gain": round(self.encoding_gain(), 3),
            "cortical_gain": round(self.cortical_gain(), 3),
            "plasticity_gain": round(self.plasticity_gain(), 3),
            "novelty_ema": round(self._novelty_ema, 3),
        }
