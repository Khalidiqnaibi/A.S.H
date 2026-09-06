"""
src/brain/vla.py

VLA channel -- Vision-Language-Action.

This is the constituent that owns *embodiment*. Anything with a physical or
irreversible effect on the world falls in its jurisdiction: actuators, file
deletion, sending messages, spending money, hardware state changes.

Its powers are deliberately asymmetric, and that asymmetry is the whole
design:

    Inside its jurisdiction (PHYSICAL / IRREVERSIBLE / MUTATION):
        - votes with full weight
        - holds a veto (bounded -- overridable by supermajority; see
          executive.py)
        - can PROPOSE grounded actions when it has fresh perceptual evidence

    Outside its jurisdiction (CONVERSATION / RETRIEVAL / COMPUTE):
        - abstains entirely. It gets no say in how ASH phrases a sentence.

So the VLA cannot start a conversation and the LLM cannot move a motor. That
is the answer to "nothing should have full control": control is partitioned
by *kind of consequence*, not by seniority.

Grounding rule
--------------
The VLA refuses to authorize a physical action it cannot see the
preconditions for. If an intent is PHYSICAL and there is no vision percept in
this cycle -- or the last one is stale -- it vetoes with reason
"ungrounded_action". A brain that reaches for something with its eyes closed
is a brain that knocks things over. This is the single most useful thing the
channel does even before any real robot is attached.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .signals import (
    ActionClass, ActionProposal, Drives, PerceptBundle, Prediction, Vote,
)

logger = logging.getLogger("ash.brain.vla")

# Action classes this channel has standing on.
VLA_JURISDICTION = {
    ActionClass.PHYSICAL,
    ActionClass.IRREVERSIBLE,
    ActionClass.MUTATION,
}

# How old a vision percept may be and still count as "grounding", in seconds.
GROUNDING_TTL = 5.0


class VLAChannel:
    """Embodied action channel. Votes only where consequences are physical."""

    def __init__(self, registry=None, require_grounding: bool = True,
                 caution_veto_threshold: float = 0.75):
        self.registry = registry
        self.require_grounding = require_grounding
        self.caution_veto_threshold = caution_veto_threshold
        self._last_vision_ts: Optional[float] = None
        self._last_scene: Optional[Any] = None
        self.actuators: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Actuator surface -- register real hardware here later.
    # ------------------------------------------------------------------
    def register_actuator(self, name: str, fn, reversible: bool = False):
        self.actuators[name] = {"fn": fn, "reversible": reversible}
        logger.info("VLA: registered actuator %r (reversible=%s)", name, reversible)

    # ------------------------------------------------------------------
    def note_perception(self, bundle: PerceptBundle):
        vis = bundle.by_modality("vision")
        if vis is not None:
            self._last_vision_ts = bundle.timestamp
            self._last_scene = vis.embedding

    def is_grounded(self, bundle: PerceptBundle) -> bool:
        if bundle.has("vision"):
            return True
        if self._last_vision_ts is None:
            return False
        return (time.time() - self._last_vision_ts) <= GROUNDING_TTL

    def has_standing(self, proposal: ActionProposal) -> bool:
        return proposal.action_class in VLA_JURISDICTION

    # ------------------------------------------------------------------
    # Proposal: the VLA can put forward its own action when it sees
    # something the language channel has no way to notice.
    # ------------------------------------------------------------------
    def propose(self, bundle: PerceptBundle, drives: Drives,
                prediction: Prediction) -> Optional[ActionProposal]:
        if not bundle.has("vision"):
            return None
        vis = bundle.by_modality("vision")
        # Placeholder policy: a high-salience visual change with no verbal
        # instruction is worth surfacing, not acting on. Real VLA policies
        # (RT-2 / OpenVLA-style) drop in here behind the same return type.
        if vis is not None and vis.salience > 0.75 and not bundle.text.strip():
            return ActionProposal(
                intent="describe_scene",
                tool_name=None,
                args=None,
                origin="vla",
                action_class=ActionClass.CONVERSATION,
                confidence=float(vis.salience),
                expected_success=0.7,
                reversible=True,
                rationale="high-salience visual change with no verbal instruction",
            )
        return None

    # ------------------------------------------------------------------
    # Vote
    # ------------------------------------------------------------------
    def vote(self, proposal: ActionProposal, bundle: PerceptBundle,
             drives: Drives, prediction: Prediction) -> Optional[Vote]:
        """Return None to abstain -- which is what happens on every ordinary
        conversational turn."""
        if not self.has_standing(proposal):
            return None

        reasons: List[str] = []
        support = 0.2
        veto = False

        # 1. Grounding.
        if proposal.action_class == ActionClass.PHYSICAL and self.require_grounding:
            if not self.is_grounded(bundle):
                return Vote(
                    member="vla", support=-1.0, confidence=0.95, veto=True,
                    veto_scope=ActionClass.PHYSICAL,
                    reason="ungrounded_action: physical intent with no fresh visual evidence",
                )
            reasons.append("grounded")
            support += 0.3

        # 2. Caution gate on irreversibility. High limbic caution + an
        # unrecoverable action = block. This is the drive vector reaching
        # into the decision layer in the one place it is allowed to.
        if not proposal.reversible and drives.caution >= self.caution_veto_threshold:
            return Vote(
                member="vla", support=-1.0, confidence=0.9, veto=True,
                veto_scope=proposal.action_class,
                reason=f"irreversible action under high caution ({drives.caution:.2f})",
            )

        # 3. Surprise gate. If the world model is currently wrong, do not
        # let the robot arm move on it.
        if prediction.surprise > 0.7 and proposal.action_class in (
            ActionClass.PHYSICAL, ActionClass.IRREVERSIBLE
        ):
            return Vote(
                member="vla", support=-0.8, confidence=0.85, veto=True,
                veto_scope=proposal.action_class,
                reason=f"world model unreliable (surprise={prediction.surprise:.2f}); "
                       "refusing consequential action",
            )

        # 4. Track record.
        if proposal.expected_success < 0.35:
            support -= 0.5
            reasons.append(f"poor track record ({proposal.expected_success:.2f})")
        elif proposal.expected_success > 0.75:
            support += 0.3
            reasons.append("reliable historically")

        if not proposal.reversible:
            support -= 0.3 * drives.caution
            reasons.append(f"irreversible, caution={drives.caution:.2f}")

        return Vote(
            member="vla",
            support=max(-1.0, min(1.0, support)),
            confidence=0.8,
            veto=veto,
            veto_scope=proposal.action_class,
            reason="; ".join(reasons) or "in jurisdiction, no objection",
        )

    # ------------------------------------------------------------------
    def execute(self, proposal: ActionProposal) -> Dict[str, Any]:
        """Only reached for proposals the board cleared and the VLA owns."""
        act = self.actuators.get(proposal.intent or "")
        if act is None:
            return {"ok": False, "error": f"no actuator registered for {proposal.intent!r}"}
        try:
            out = act["fn"](proposal.args)
            return {"ok": True, "result": out}
        except Exception as e:
            logger.exception("Actuator %r failed", proposal.intent)
            return {"ok": False, "error": str(e)}
