"""
src/daemon/attention.py

The attention gate: "only react when it feels needed or directly acknowledged."

This is the piece that makes an always-on assistant tolerable rather than
exhausting. Every ambient event gets one of three dispositions:

    IGNORE   -- beneath the noise floor. Not even remembered individually.
    OBSERVE  -- perceived and remembered. No response. This is the default,
                and it is what ~99% of a day's events get.
    RESPOND  -- run the full cognitive cycle and say something.

Two independent routes to RESPOND
---------------------------------
1. ADDRESSED. The user said ASH's name, or spoke a direct request. This is an
   unconditional override -- no threshold, no refractory period, no drive
   check. Being spoken to and ignored is the single worst failure mode an
   assistant has, so it is not subject to any of the machinery below.

2. VOLUNTEERED. Nobody asked, but something crossed the bar. The bar is not a
   constant; it is raised and lowered by:

     + event urgency        battery critical, a sensor alert
     + event salience       how much it stands out
     + novelty              never seen anything like it
     + curiosity drive      a bored ASH speaks up more
     - refractory decay     just spoke -> much higher bar, decays over minutes
     - interruption cost    user typing in a focused app -> higher bar
     - fatigue              a long day lowers willingness to volunteer
     - budget exhaustion    hard cap on unsolicited remarks per hour

The budget deserves emphasis. Everything else is a soft weighting that could,
under some unlucky combination of drives, produce a chatty ASH. The hourly
budget is a hard stop: once spent, nothing volunteers until the window rolls,
no matter how interesting the world gets. Addressed messages never consume it.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("ash.daemon.attention")


class Disposition(str, Enum):
    IGNORE = "ignore"
    OBSERVE = "observe"
    RESPOND = "respond"


@dataclass
class AttentionDecision:
    disposition: Disposition
    score: float = 0.0
    threshold: float = 0.0
    reasons: List[str] = field(default_factory=list)
    addressed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "score": round(self.score, 3),
            "threshold": round(self.threshold, 3),
            "addressed": self.addressed,
            "reasons": self.reasons,
        }


class AttentionGate:
    def __init__(
        self,
        base_threshold: float = 0.62,
        ignore_below: float = 0.12,
        refractory_seconds: float = 180.0,
        refractory_penalty: float = 0.45,
        max_unsolicited_per_hour: int = 6,
        quiet_hours: Optional[tuple] = None,     # (start_hour, end_hour), local
        interrupt_cost_weight: float = 0.20,
    ):
        self.base_threshold = base_threshold
        self.ignore_below = ignore_below
        self.refractory_seconds = refractory_seconds
        self.refractory_penalty = refractory_penalty
        self.max_unsolicited_per_hour = max_unsolicited_per_hour
        self.quiet_hours = quiet_hours
        self.interrupt_cost_weight = interrupt_cost_weight

        self._last_response: float = 0.0
        self._unsolicited: Deque[float] = deque(maxlen=64)
        self.muted = False

        # Counters for the status endpoint.
        self.counts = {d.value: 0 for d in Disposition}

    # ------------------------------------------------------------------
    def _refractory(self, now: float, urgency: float = 0.0) -> float:
        """Extra threshold from having recently spoken.

        Decays exponentially rather than cutting off at a boundary -- a hard
        window makes ASH chatty the instant it expires.

        Scaled down by the event's urgency: a system on fire should not have
        to wait out a politeness timer because ASH mentioned something two
        minutes ago. At urgency 1.0 the refractory penalty vanishes entirely.
        """
        if not self._last_response:
            return 0.0
        elapsed = now - self._last_response
        decay = math.exp(-elapsed / max(1.0, self.refractory_seconds / 2))
        return self.refractory_penalty * decay * (1.0 - min(1.0, urgency))

    def _budget_spent(self, now: float) -> int:
        while self._unsolicited and now - self._unsolicited[0] > 3600:
            self._unsolicited.popleft()
        return len(self._unsolicited)

    def _in_quiet_hours(self, now: float) -> bool:
        if not self.quiet_hours:
            return False
        h = time.localtime(now).tm_hour
        start, end = self.quiet_hours
        return (start <= h < end) if start < end else (h >= start or h < end)

    # ------------------------------------------------------------------
    def decide(self, event, drives, novelty: float = 0.5,
               user_busy: float = 0.0, now: Optional[float] = None) -> AttentionDecision:
        """`user_busy` is 0..1: 0 = away from the machine, 1 = actively typing
        in a focused application."""
        now = now or time.time()
        reasons: List[str] = []

        # ---- Route 1: directly addressed. Nothing below applies. --------
        #
        # Note what is deliberately NOT done here: the refractory clock is not
        # armed. Refractory exists to stop ASH volunteering repeatedly, and
        # answering a question is not volunteering. Arming it on an addressed
        # turn meant that asking ASH something would then suppress a genuine
        # alert for the next three minutes -- the exact opposite of useful.
        if getattr(event, "addressed", False):
            self.counts[Disposition.RESPOND.value] += 1
            return AttentionDecision(
                disposition=Disposition.RESPOND, score=1.0, threshold=0.0,
                addressed=True, reasons=["directly addressed"],
            )

        if self.muted:
            self.counts[Disposition.OBSERVE.value] += 1
            return AttentionDecision(Disposition.OBSERVE, reasons=["muted"])

        # ---- Score ------------------------------------------------------
        salience = float(getattr(event, "salience", 0.3))
        urgency = float(getattr(event, "urgency", 0.0))

        score = (
            0.45 * urgency
            + 0.30 * salience
            + 0.15 * novelty
            + 0.10 * drives.curiosity
        )
        # Urgency is allowed to dominate: a critical battery should get
        # through a mildly raised bar without needing anything else to agree.
        score = max(score, urgency * 0.95)

        # ---- Threshold --------------------------------------------------
        threshold = self.base_threshold
        refr = self._refractory(now, urgency)
        if refr > 0.01:
            threshold += refr
            reasons.append(f"refractory +{refr:.2f}")

        busy_cost = self.interrupt_cost_weight * user_busy
        if busy_cost > 0.01:
            threshold += busy_cost
            reasons.append(f"user busy +{busy_cost:.2f}")

        fatigue_cost = 0.15 * drives.fatigue
        if fatigue_cost > 0.01:
            threshold += fatigue_cost
            reasons.append(f"fatigue +{fatigue_cost:.2f}")

        # A high urgency drive lowers the bar -- that is the limbic system
        # deciding this is a moment to speak up.
        threshold -= 0.12 * drives.urgency
        # Caution raises it. An ASH that is unsure of itself stays quiet.
        threshold += 0.10 * drives.caution

        # ---- Hard stops -------------------------------------------------
        if score < self.ignore_below:
            self.counts[Disposition.IGNORE.value] += 1
            return AttentionDecision(Disposition.IGNORE, score, threshold,
                                     reasons + ["below noise floor"])

        if self._in_quiet_hours(now) and urgency < 0.85:
            self.counts[Disposition.OBSERVE.value] += 1
            return AttentionDecision(Disposition.OBSERVE, score, threshold,
                                     reasons + ["quiet hours"])

        spent = self._budget_spent(now)
        if spent >= self.max_unsolicited_per_hour and urgency < 0.9:
            self.counts[Disposition.OBSERVE.value] += 1
            return AttentionDecision(
                Disposition.OBSERVE, score, threshold,
                reasons + [f"hourly budget spent ({spent}/{self.max_unsolicited_per_hour})"])

        # ---- Verdict ------------------------------------------------------
        if score >= threshold:
            self._last_response = now
            self._unsolicited.append(now)
            self.counts[Disposition.RESPOND.value] += 1
            return AttentionDecision(Disposition.RESPOND, score, threshold,
                                     reasons + ["volunteered"])

        self.counts[Disposition.OBSERVE.value] += 1
        return AttentionDecision(Disposition.OBSERVE, score, threshold,
                                 reasons + [f"score {score:.2f} < threshold {threshold:.2f}"])

    # ------------------------------------------------------------------
    def note_user_turn(self):
        """A direct exchange resets the refractory clock: mid-conversation,
        ASH should answer freely rather than rationing itself."""
        self._last_response = 0.0

    def mute(self, on: bool = True):
        self.muted = on
        logger.info("Attention gate %s", "MUTED" if on else "unmuted")

    def status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "muted": self.muted,
            "counts": dict(self.counts),
            "unsolicited_last_hour": self._budget_spent(now),
            "budget": self.max_unsolicited_per_hour,
            "seconds_since_response": (round(now - self._last_response, 1)
                                       if self._last_response else None),
            "quiet_hours": self.quiet_hours,
        }
