"""
src/brain/signals.py

Shared value types passed between brain subsystems.

Everything in this file is a plain dataclass with no behavior beyond
serialization. The point is that Sensory -> Executive -> System1/System2 ->
Predictive -> Memory all speak one vocabulary, so a subsystem can be swapped
out (a real ViT for the stub extractor, a real JEPA for the linear predictor)
without touching anything else.

Naming maps 1:1 onto the architecture diagram:

    Percept          <- Sensory Extractors output
    Drives           <- Homeostatic Modulator output ("latent drives")
    Prediction       <- Predictive Model output
    ActionProposal   <- what System 1 / System 2 / VLA each put forward
    Vote             <- what each constituent casts at the Executive Core
    Verdict          <- the Executive Core's arbitrated decision
    Pathway          <- which execution speed was chosen (the fast/slow split)
    BrainTrace       <- full introspection record of one cognitive cycle
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ----------------------------------------------------------------------
# Execution pathways -- the fast/slow split, made explicit.
# ----------------------------------------------------------------------
class Pathway(str, Enum):
    """How much cognition a turn is allowed to spend.

    OBSERVE -- perceive and remember, say nothing. The default for ambient
               events in daemon mode; costs one embedding and a journal line.
    REFLEX  -- pure lookup. No tool, no memory retrieval, no LLM.
               Sub-millisecond. Repeat of a very recent identical query.
    FAST    -- System 1. Classifier fires, tool runs, response comes from a
               template. No LLM call, no full memory retrieval. ~10-200ms.
    SLOW    -- System 2. Full memory retrieval, predictive rollout over
               candidate actions, LLM deliberation and narration. Seconds.
    ESCALATED -- started FAST, hit a surprise/conflict mid-flight, and got
               re-run through the slow path. The fast answer is discarded.
    """

    OBSERVE = "observe"
    REFLEX = "reflex"
    FAST = "fast"
    SLOW = "slow"
    ESCALATED = "escalated"


class ActionClass(str, Enum):
    """Jurisdiction categories. Which board members may vote or veto on a
    given proposal depends on which of these it falls into -- that is the
    mechanism that stops any one subsystem owning every decision."""

    CONVERSATION = "conversation"   # pure talk; no side effects
    RETRIEVAL = "retrieval"         # reads memory / files; no mutation
    COMPUTE = "compute"             # deterministic pure function (calculator)
    MUTATION = "mutation"           # writes state the user can observe
    PHYSICAL = "physical"           # actuators, hardware, real-world effects
    IRREVERSIBLE = "irreversible"   # cannot be undone (delete, send, purchase)


@dataclass
class Percept:
    """Output of the Sensory Extractors. One per modality, fused into a bundle."""

    modality: str                       # "text" | "vision" | "audio" | ...
    raw: Any = None                     # original input (str, ndarray, bytes)
    embedding: Optional[Any] = None     # np.ndarray, L2-normalized
    tokens: List[str] = field(default_factory=list)
    salience: float = 0.0               # 0..1 how much this demands attention
    novelty: float = 0.0                # 0..1 vs. recent percept history
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        d.pop("raw", None)
        d["has_embedding"] = self.embedding is not None
        return d


@dataclass
class PerceptBundle:
    """All modalities for one cycle, fused."""

    percepts: List[Percept] = field(default_factory=list)
    fused_embedding: Optional[Any] = None
    text: str = ""
    salience: float = 0.0
    novelty: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def by_modality(self, modality: str) -> Optional[Percept]:
        for p in self.percepts:
            if p.modality == modality:
                return p
        return None

    def has(self, modality: str) -> bool:
        return self.by_modality(modality) is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modalities": [p.modality for p in self.percepts],
            "text": self.text[:200],
            "salience": round(self.salience, 3),
            "novelty": round(self.novelty, 3),
            "percepts": [p.as_dict() for p in self.percepts],
        }


@dataclass
class Drives:
    """Latent drives broadcast by the Homeostatic Modulator.

    These are NOT emotions. They are the small set of scalars that emotion
    gets compressed into before it is allowed to touch cognition. Emotion
    still only shapes tone (per ASH's original constraint); drives shape
    *how much thinking to buy*, which is a separate lever.

    All values 0..1.
    """

    urgency: float = 0.3        # answer now vs. answer well
    caution: float = 0.3        # tolerance for irreversible / risky actions
    curiosity: float = 0.4      # willingness to explore instead of exploit
    effort_budget: float = 0.5  # how much compute this turn is worth
    social: float = 0.5         # need to attend to the human's emotional state
    fatigue: float = 0.0        # accumulated load; suppresses effort_budget

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in asdict(self).items()}

    def as_prompt_block(self) -> str:
        return (
            "[LATENT DRIVES]\n"
            f"urgency: {self.urgency:.2f} | caution: {self.caution:.2f} | "
            f"curiosity: {self.curiosity:.2f} | effort: {self.effort_budget:.2f} | "
            f"social: {self.social:.2f}"
        )


@dataclass
class Prediction:
    """Predictive Model output for one cycle."""

    predicted_latent: Optional[Any] = None   # next-state guess, np.ndarray
    surprise: float = 0.0                    # 0..1 normalized state-prediction error
    outcome_surprise: float = 0.0            # 0..1 how wrong outcome estimates have been
    raw_error: float = 0.0                   # unnormalized cosine distance
    confidence: float = 0.0                  # 0..1 model's faith in itself
    # Per-candidate expected outcome: intent -> P(success)
    action_outcomes: Dict[str, float] = field(default_factory=dict)
    horizon: int = 1

    def effective_surprise(self) -> float:
        """What the Executive's gate should actually react to.

        Raw state-prediction error is a poor gating signal on its own: humans
        change topic for reasons no forward model can anticipate, so a high
        residual usually means "the user moved on", not "my model is broken".
        Weighting it by the model's own confidence fixes that -- error only
        counts as *surprise* when the model normally does well and just
        failed. Outcome error (tools behaving differently than predicted) is
        always meaningful and is taken at full strength.
        """
        return max(self.outcome_surprise, self.surprise * self.confidence)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "surprise": round(self.surprise, 3),
            "outcome_surprise": round(self.outcome_surprise, 3),
            "raw_error": round(self.raw_error, 3),
            "confidence": round(self.confidence, 3),
            "action_outcomes": {k: round(v, 3) for k, v in self.action_outcomes.items()},
        }


@dataclass
class ActionProposal:
    """A candidate action put forward by some subsystem.

    `origin` matters: the Executive Core weights proposals partly by who
    made them and refuses to let any single origin dominate.
    """

    intent: Optional[str]
    tool_name: Optional[str] = None
    args: Any = None
    origin: str = "system1"                       # system1 | system2 | vla | fallback
    action_class: ActionClass = ActionClass.CONVERSATION
    confidence: float = 0.0                       # actor's own confidence 0..1
    value: float = 0.0                            # critic's estimated return
    expected_success: float = 0.5                 # from the predictive model
    reversible: bool = True
    est_latency_ms: float = 0.0
    rationale: str = ""

    def key(self) -> str:
        return f"{self.origin}:{self.intent or 'none'}"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_class"] = self.action_class.value
        d["args"] = str(self.args)[:200] if self.args is not None else None
        return d


@dataclass
class Vote:
    """One board member's position on one proposal."""

    member: str
    support: float                # -1 (oppose) .. +1 (endorse)
    confidence: float = 0.5       # 0..1, scales the member's effective weight
    veto: bool = False
    veto_scope: Optional[ActionClass] = None   # veto only valid in-jurisdiction
    absolute: bool = False        # only the Constitution may set this
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "member": self.member,
            "support": round(self.support, 3),
            "confidence": round(self.confidence, 3),
            "veto": self.veto,
            "absolute": self.absolute,
            "reason": self.reason,
        }


@dataclass
class Verdict:
    """Result of Executive Core arbitration."""

    chosen: Optional[ActionProposal]
    pathway: Pathway
    score: float = 0.0
    margin: float = 0.0                 # gap to runner-up; small = contested
    deadlocked: bool = False
    vetoed_by: List[str] = field(default_factory=list)
    escalation_reasons: List[str] = field(default_factory=list)
    votes: List[Vote] = field(default_factory=list)
    tally: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chosen": self.chosen.as_dict() if self.chosen else None,
            "pathway": self.pathway.value,
            "score": round(self.score, 3),
            "margin": round(self.margin, 3),
            "deadlocked": self.deadlocked,
            "vetoed_by": self.vetoed_by,
            "escalation_reasons": self.escalation_reasons,
            "votes": [v.as_dict() for v in self.votes],
            "tally": {k: round(v, 3) for k, v in self.tally.items()},
        }


@dataclass
class BrainTrace:
    """Full introspection record of one cognitive cycle. Logged, and
    exposed via Brain.last_trace for debugging / UI."""

    query: str = ""
    pathway: Pathway = Pathway.SLOW
    percepts: Dict[str, Any] = field(default_factory=dict)
    drives: Dict[str, float] = field(default_factory=dict)
    prediction: Dict[str, Any] = field(default_factory=dict)
    concepts: Dict[str, Any] = field(default_factory=dict)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    verdict: Dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    memory_hits: int = 0
    timings_ms: Dict[str, float] = field(default_factory=dict)
    escalated_from: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pathway"] = self.pathway.value
        return d


@dataclass
class BrainResponse:
    """What Brain.think() hands back to ASH."""

    text: str
    trace: BrainTrace
    tool_output: Any = None
    intent: Optional[str] = None
    tool_used: Optional[str] = None
    tool_success: Optional[bool] = None
    pathway: Pathway = Pathway.SLOW
