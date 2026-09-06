"""
src/brain/system1.py

System 1 Policy  (the "Fast Reflexes / Actor-Critic" box).

The actor is the machinery ASH already had: the embedding intent classifier
plus the tool registry. Given a query it names an action, fast, with no
reasoning. That part was always a reflex arc -- it just was never labeled as
one, and it was never allowed to *skip* the LLM, so every reflex still paid
full deliberation cost.

The critic is new and is the reason this is a policy rather than a lookup
table. It's a linear value head over an 11-dim feature vector that estimates
the return of firing this reflex *right now*, in this context, for this user
state. It trains online by Widrow-Hoff on a reward assembled from what
actually happened afterward:

    tool succeeded                     +1.0
    tool failed                        -1.0
    had to escalate to the slow path   -0.4   (the reflex was overconfident)
    user immediately rephrased/repeated -0.6  (we answered the wrong question)
    next-turn sentiment positive       +0.3
    next-turn sentiment negative       -0.3

That reward is necessarily deferred -- you cannot know a reflex was wrong
until the next turn -- so the critic keeps a pending slot and settles up when
the following cycle reports back. This is the only component in ASH that
learns from live interaction, and it learns a bounded, auditable thing: not
*what to say*, only *when it is safe to not think*.

Critically, the critic never selects an action. It scores the action the
classifier already chose. The Executive Core is what decides whether that
score is good enough to act on.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .signals import ActionClass, ActionProposal, Drives, PerceptBundle, Prediction

logger = logging.getLogger("ash.brain.system1")

STATE_DIR = os.environ.get("ASH_BRAIN_STATE", os.path.join(os.getcwd(), "state", "brain"))

FEATURE_NAMES = [
    "intent_score", "novelty", "salience", "surprise", "expected_success",
    "urgency", "caution", "curiosity", "effort_budget", "reversible", "bias",
]

# Intent-name heuristics for classifying an action's blast radius when the
# tool's shape file doesn't declare it. Conservative by construction: unknown
# verbs that smell destructive get treated as irreversible.
_IRREVERSIBLE_HINTS = {
    "delete", "remove", "purge", "drop", "send", "email", "post", "publish",
    "buy", "purchase", "pay", "transfer", "format", "wipe", "kill", "shutdown",
    "reboot", "overwrite", "deploy", "merge", "push",
}
_PHYSICAL_HINTS = {
    "motor", "servo", "actuate", "move", "drive", "arm", "gripper", "led",
    "relay", "gpio", "camera", "pan", "tilt", "door", "lock", "valve", "pump",
}
_MUTATION_HINTS = {"write", "save", "create", "update", "set", "rename", "append", "edit"}
_RETRIEVAL_HINTS = {"read", "list", "search", "find", "get", "show", "info", "stats", "lookup"}
_COMPUTE_HINTS = {"calc", "calculate", "math", "convert", "count", "sum"}


def classify_action(intent: Optional[str], shape: Optional[Dict[str, Any]] = None) -> Tuple[ActionClass, bool]:
    """Return (action_class, reversible).

    A shape file may declare these explicitly:
        {"action_class": "physical", "reversible": false}
    Otherwise infer from the intent tag.
    """
    if shape:
        declared = shape.get("action_class")
        if declared:
            try:
                cls = ActionClass(declared)
                return cls, bool(shape.get("reversible", cls not in (ActionClass.IRREVERSIBLE, ActionClass.PHYSICAL)))
            except ValueError:
                logger.warning("Unknown action_class %r in shape; inferring instead", declared)

    tag = (intent or "").lower()
    words = set(re.findall(r"[a-z]+", tag))

    if words & _IRREVERSIBLE_HINTS:
        return ActionClass.IRREVERSIBLE, False
    if words & _PHYSICAL_HINTS:
        return ActionClass.PHYSICAL, False
    if words & _MUTATION_HINTS:
        return ActionClass.MUTATION, True
    if words & _COMPUTE_HINTS:
        return ActionClass.COMPUTE, True
    if words & _RETRIEVAL_HINTS:
        return ActionClass.RETRIEVAL, True
    if not tag or tag in ("conversation", "chat", "unknown", "none"):
        return ActionClass.CONVERSATION, True
    return ActionClass.RETRIEVAL, True


class Critic:
    """Linear value head V(s) = w . x, trained by Widrow-Hoff."""

    def __init__(self, dim: int = len(FEATURE_NAMES), lr: float = 0.05):
        self.w = np.zeros(dim, dtype=np.float32)
        # Warm start with the priors a sensible engineer would hand-code, so
        # the reflex path isn't wildly miscalibrated on turn one.
        init = {
            "intent_score": 1.2, "expected_success": 0.9, "reversible": 0.4,
            "surprise": -1.1, "novelty": -0.6, "caution": -0.5,
            "urgency": 0.3, "effort_budget": -0.2, "bias": -0.1,
        }
        for i, name in enumerate(FEATURE_NAMES):
            self.w[i] = init.get(name, 0.0)
        self.lr = lr
        self.updates = 0

    def value(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def learn(self, x: np.ndarray, reward: float) -> float:
        pred = self.value(x)
        err = reward - pred
        self.w += self.lr * err * x
        np.clip(self.w, -5.0, 5.0, out=self.w)
        self.updates += 1
        return err

    def explain(self) -> Dict[str, float]:
        return {n: round(float(self.w[i]), 3) for i, n in enumerate(FEATURE_NAMES)}


class System1Policy:
    """Actor (classifier + registry) + Critic (value head) + reflex cache."""

    def __init__(self, classifier, registry, shapes: Optional[Dict[str, Dict[str, Any]]] = None,
                 reflex_ttl: float = 25.0, persist: bool = True):
        self.classify = classifier          # callable(query) -> {"intent","score"}
        self.registry = registry            # tools.registry.REGISTRY
        self.shapes = shapes or {}
        self.critic = Critic()
        self.persist = persist
        self.path = os.path.join(STATE_DIR, "critic.json")

        # Reflex cache: exact repeats inside the TTL skip everything.
        self.reflex_ttl = reflex_ttl
        self._reflex: Dict[str, Tuple[float, str, Any]] = {}

        self._recent_queries: Deque[str] = deque(maxlen=8)
        self._pending: Optional[Dict[str, Any]] = None   # deferred reward slot
        self._load()

    # ------------------------------------------------------------------
    # Reflex cache -- the REFLEX pathway
    # ------------------------------------------------------------------
    @staticmethod
    def _norm(q: str) -> str:
        return " ".join(re.findall(r"[a-z0-9']+", (q or "").lower()))

    def reflex_lookup(self, query: str) -> Optional[Tuple[str, Any]]:
        hit = self._reflex.get(self._norm(query))
        if not hit:
            return None
        ts, text, tool_out = hit
        if time.time() - ts > self.reflex_ttl:
            self._reflex.pop(self._norm(query), None)
            return None
        return text, tool_out

    def reflex_store(self, query: str, text: str, tool_out: Any):
        self._reflex[self._norm(query)] = (time.time(), text, tool_out)
        if len(self._reflex) > 128:
            oldest = sorted(self._reflex.items(), key=lambda kv: kv[1][0])[:32]
            for k, _ in oldest:
                self._reflex.pop(k, None)

    def invalidate_reflex(self):
        """Any mutating action makes cached answers untrustworthy."""
        self._reflex.clear()

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    def features(self, intent_score: float, bundle: PerceptBundle, drives: Drives,
                 prediction: Prediction, expected_success: float, reversible: bool) -> np.ndarray:
        vals = {
            "intent_score": float(intent_score),
            "novelty": float(bundle.novelty),
            "salience": float(bundle.salience),
            "surprise": float(prediction.surprise),
            "expected_success": float(expected_success),
            "urgency": float(drives.urgency),
            "caution": float(drives.caution),
            "curiosity": float(drives.curiosity),
            "effort_budget": float(drives.effort_budget),
            "reversible": 1.0 if reversible else 0.0,
            "bias": 1.0,
        }
        return np.array([vals[n] for n in FEATURE_NAMES], dtype=np.float32)

    # ------------------------------------------------------------------
    # Actor
    # ------------------------------------------------------------------
    def propose(self, query: str, bundle: PerceptBundle, drives: Drives,
                prediction: Prediction) -> ActionProposal:
        """One forward pass of the reflex arc. No tool is executed here --
        the Executive Core decides whether this proposal is acted on."""
        t0 = time.perf_counter()
        try:
            routed = self.classify(query) or {}
        except Exception:
            logger.exception("System1 classifier failed")
            routed = {}

        intent = routed.get("intent") or routed.get("tag")
        score = float(routed.get("score", routed.get("intent_score", 0.0)) or 0.0)

        entry = self.registry.match(intent) if intent else None
        shape = self.shapes.get(intent or "")
        action_class, reversible = classify_action(intent, shape)

        if entry is None:
            # No tool answers to this intent -> it is a conversational turn,
            # which System 1 is structurally incapable of handling alone.
            action_class = ActionClass.CONVERSATION
            reversible = True

        exp_succ = prediction.action_outcomes.get(intent, 0.5) if intent else 0.5
        x = self.features(score, bundle, drives, prediction, exp_succ, reversible)
        value = self.critic.value(x)

        proposal = ActionProposal(
            intent=intent,
            tool_name=entry.name if entry else None,
            args=query,
            origin="system1",
            action_class=action_class,
            confidence=score,
            value=value,
            expected_success=exp_succ,
            reversible=reversible,
            est_latency_ms=(time.perf_counter() - t0) * 1000.0,
            rationale=(f"classifier matched '{intent}' @ {score:.2f}" if intent
                       else "no intent matched"),
        )
        proposal.__dict__["_features"] = x   # stashed for the deferred update
        return proposal

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, proposal: ActionProposal) -> Dict[str, Any]:
        entry = self.registry.match(proposal.intent) if proposal.intent else None
        if entry is None:
            return {"ok": False, "error": f"no tool for intent {proposal.intent!r}"}
        out = entry.run(proposal.args if proposal.args is not None else "")
        if proposal.action_class in (ActionClass.MUTATION, ActionClass.IRREVERSIBLE, ActionClass.PHYSICAL):
            self.invalidate_reflex()
        return out

    # ------------------------------------------------------------------
    # Deferred reward
    # ------------------------------------------------------------------
    def stage_reward(self, proposal: ActionProposal, tool_success: Optional[bool], escalated: bool,
                     bundle: Optional[PerceptBundle] = None, drives: Optional[Drives] = None,
                     prediction: Optional[Prediction] = None):
        """Open a pending reward slot for the action just taken. Settled on
        the next call to `settle_reward` once we can see how it landed.

        Proposals that came from System 2 or the VLA carry no feature vector,
        so one is computed here from the same cycle context. That matters:
        the critic needs to learn from deliberated outcomes too, otherwise it
        only ever sees the cases it already felt confident about and its
        estimate of "when is it safe to skip thinking" never improves on the
        hard cases.
        """
        x = proposal.__dict__.get("_features")
        if x is None and bundle is not None and drives is not None and prediction is not None:
            x = self.features(
                proposal.confidence, bundle, drives, prediction,
                proposal.expected_success, proposal.reversible,
            )
        if x is None:
            return
        self._pending = {
            "x": x,
            "query": self._norm(str(proposal.args or "")),
            "tool_success": tool_success,
            "escalated": escalated,
            "intent": proposal.intent,
            "tool_used": proposal.tool_name,
        }

    def settle_reward(self, next_query: str, next_sentiment: Optional[str]) -> Optional[float]:
        """Close out the previous turn's pending reward using this turn's
        evidence, and train the critic on it."""
        if not self._pending:
            self._recent_queries.append(self._norm(next_query))
            return None

        p = self._pending
        self._pending = None

        r = 0.0
        if p["tool_success"] is True:
            r += 1.0
        elif p["tool_success"] is False:
            r -= 1.0
        if p["escalated"]:
            r -= 0.4

        nq = self._norm(next_query)
        # Repeat detection: heavy token overlap on consecutive turns usually
        # means the previous answer missed. But this only counts as failure
        # if the previous turn actually claimed to answer -- if it asked a
        # clarifying question, a closely-related follow-up is the *correct*
        # outcome, not evidence of a miss. Penalizing that was teaching the
        # critic to distrust exactly the turns it handled well.
        if p["query"] and nq and p.get("tool_used"):
            a, b = set(p["query"].split()), set(nq.split())
            if a and b and len(a & b) / max(1, len(a | b)) > 0.6:
                r -= 0.6

        if next_sentiment == "POSITIVE":
            r += 0.3
        elif next_sentiment == "NEGATIVE":
            r -= 0.3

        r = float(np.clip(r, -2.0, 2.0))
        err = self.critic.learn(p["x"], r)
        self._recent_queries.append(nq)
        logger.debug("System1 critic update: reward=%.2f td_err=%.3f", r, err)
        return r

    # ------------------------------------------------------------------
    def save(self):
        if not self.persist:
            return
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"w": self.critic.w.tolist(), "updates": self.critic.updates}, fh, indent=2)
        except Exception:
            logger.exception("Failed to persist critic")

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                w = np.array(d.get("w", []), dtype=np.float32)
                if w.shape == self.critic.w.shape:
                    self.critic.w = w
                    self.critic.updates = int(d.get("updates", 0))
                    logger.info("Critic restored (%d updates)", self.critic.updates)
        except Exception:
            logger.exception("Failed to restore critic -- using warm-start weights")
