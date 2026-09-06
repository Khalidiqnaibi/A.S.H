"""
src/concepts/model.py

Data types for the associative layer.

A concept is a unit with a prototype vector in the same space MiniLM already
produces, plus the bookkeeping that plasticity needs. A synapse is a directed
weighted edge between two of them.

The one structural decision worth flagging up front is that every synapse
carries TWO weights:

    w_fast   potentiated within a single session, decays with a time constant
             of hours, and is lost if nothing consolidates it
    w_slow   persistent, only ever changed during the sleep phase

Effective strength is their sum. This is early-LTP versus late-LTP, and it is
not decoration: it is what lets ASH form an association immediately -- so a
thing you told it ten minutes ago actually connects -- without that
association being permanent before anything has confirmed it was useful. A
single-weight design forces a choice between "learns nothing during the day"
and "permanently rewires itself on one coincidence". Both are worse.

Named vs unnamed
----------------
Seed concepts come from the intent catalog, CoreMemory and EntityMemory, so
they arrive with labels. Concepts grown by resonance have no label, because
nothing named them -- they are identified by their top activators, the same
way a cortical unit is identified by what makes it fire. `describe()` builds
that description on demand.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SynType(str, Enum):
    """What kind of relationship an edge encodes.

    The brain does not have one undifferentiated associative substrate --
    anatomy constrains what can wire to what, and different projections obey
    different rules. Typing the edges is how that constraint is expressed
    here, and it matters for three reasons:

      * different types learn with different rules (ASSOC is symmetric and
        Hebbian; CAUSAL is directional and learned by spike timing)
      * different types decay at different rates (a causal sequence should
        outlive a coincidence)
      * spreading activation can weight them differently, so "these co-occur"
        does not get confused with "this leads to that"

    Without typing, a graph learns that battery-low and charger go together
    but has no representation of which one comes first -- which is exactly
    the information you need to be useful rather than merely associative.
    """

    ASSOC = "assoc"       # symmetric co-occurrence
    CAUSAL = "causal"     # directional, learned by pre-before-post timing
    ROUTE = "route"       # concept -> intent; how perception reaches action
    INHIBIT = "inhibit"   # suppression, including frozen constraint edges
    PART_OF = "part_of"   # hierarchical containment, set by consolidation


# Per-type learning parameters. Causal edges are harder to form (they need
# consistent temporal ordering) and slower to forget once formed.
TYPE_RULES: Dict[str, Dict[str, float]] = {
    SynType.ASSOC.value:   {"lr": 1.00, "decay": 1.00, "spread": 1.00},
    SynType.CAUSAL.value:  {"lr": 0.70, "decay": 0.55, "spread": 1.15},
    SynType.ROUTE.value:   {"lr": 0.85, "decay": 0.70, "spread": 1.30},
    SynType.INHIBIT.value: {"lr": 0.50, "decay": 0.40, "spread": 1.00},
    SynType.PART_OF.value: {"lr": 0.60, "decay": 0.30, "spread": 0.85},
}


class Origin(str, Enum):
    """Where a concept came from. Determines whether it can be pruned."""

    INTENT = "intent"         # seeded from the classifier catalog
    CORE = "core"             # seeded from a CoreMemory rule
    ENTITY = "entity"         # seeded from EntityMemory
    TOOL = "tool"             # seeded from the tool registry
    GROWN = "grown"           # created by resonance failure at runtime
    MERGED = "merged"         # produced by consolidating two others


# Concepts from these origins are structural: they can be weakened but never
# pruned, because deleting them would silently remove a capability.
PROTECTED_ORIGINS = {Origin.INTENT, Origin.CORE, Origin.TOOL}


@dataclass
class Concept:
    """One unit in the network."""

    id: int
    prototype: np.ndarray                     # L2-normalized, embedding-space
    label: Optional[str] = None               # None for grown units
    origin: Origin = Origin.GROWN
    intent: Optional[str] = None              # set if this unit routes to a tool

    # Activation bookkeeping -- drives PMI, pruning and merging.
    activations: int = 0                      # times it entered the active set
    activation_sum: float = 0.0               # for mean activation strength
    created: float = field(default_factory=time.time)
    last_active: float = 0.0

    # Prototype drift. A concept slowly moves toward the inputs that fire it,
    # which is competitive learning -- without it, seeds stay wherever the
    # catalog put them and never adapt to how this particular user talks.
    plasticity: float = 0.05                  # 0 = frozen prototype

    # Provenance for grown units: the text that created them, so a graph
    # inspected six months later is still explicable.
    seed_text: str = ""

    frozen: bool = False                      # exempt from all learning

    def touch(self, strength: float, now: Optional[float] = None):
        self.activations += 1
        self.activation_sum += float(strength)
        self.last_active = now or time.time()

    def mean_activation(self) -> float:
        return self.activation_sum / self.activations if self.activations else 0.0

    def adapt(self, vec: np.ndarray, strength: float = 1.0):
        """Move the prototype toward an input that activated this concept."""
        if self.frozen or self.plasticity <= 0:
            return
        eta = self.plasticity * float(strength)
        p = self.prototype + eta * (vec - self.prototype)
        n = float(np.linalg.norm(p))
        if n > 1e-9:
            self.prototype = (p / n).astype(np.float32)

    def name(self) -> str:
        return self.label or f"c{self.id}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "origin": self.origin.value,
            "intent": self.intent,
            "activations": self.activations,
            "mean_activation": round(self.mean_activation(), 4),
            "age_h": round((time.time() - self.created) / 3600, 2),
            "frozen": self.frozen,
            "seed_text": self.seed_text[:80],
        }


@dataclass
class Synapse:
    """A directed edge. See the module docstring for why there are two weights."""

    src: int
    dst: int
    w_slow: float = 0.0
    w_fast: float = 0.0
    kind: SynType = SynType.ASSOC
    created: float = field(default_factory=time.time)
    last_potentiated: float = 0.0
    potentiations: int = 0
    # Running evidence of temporal ordering, from STDP. Positive means src
    # reliably precedes dst. This is what turns a correlation into a
    # direction, and it is tracked separately from the weight so a strong
    # symmetric association and a strong causal one are distinguishable.
    order_evidence: float = 0.0
    # Frozen edges are seeded from hard constraints and are never touched by
    # Hebbian learning. The graph may learn anything it likes; it may not
    # learn its way around a CoreMemory rule.
    frozen: bool = False

    def weight(self) -> float:
        return self.w_slow + self.w_fast

    def rules(self) -> Dict[str, float]:
        return TYPE_RULES.get(self.kind.value, TYPE_RULES[SynType.ASSOC.value])

    def is_causal(self) -> bool:
        return self.kind == SynType.CAUSAL or self.order_evidence > 0.5

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src, "dst": self.dst, "kind": self.kind.value,
            "w": round(self.weight(), 4),
            "slow": round(self.w_slow, 4), "fast": round(self.w_fast, 4),
            "order": round(self.order_evidence, 3),
            "n": self.potentiations, "frozen": self.frozen,
        }


@dataclass
class Activation:
    """The result of presenting one input to the network.

    `initial` is the bottom-up k-WTA response; `spread` is what it becomes
    after activation propagates along synapses. Keeping both is what makes
    the graph debuggable -- the interesting question is almost always which
    concepts lit up *because of* other concepts rather than because of the
    input.
    """

    initial: Dict[int, float] = field(default_factory=dict)
    spread: Dict[int, float] = field(default_factory=dict)
    novelty: float = 0.0            # 1 - best bottom-up match
    grown: Optional[int] = None     # id of a concept created by this input
    steps: int = 0

    def top(self, n: int = 8) -> List[Tuple[int, float]]:
        return sorted(self.spread.items(), key=lambda kv: -kv[1])[:n]

    def active_ids(self) -> List[int]:
        return list(self.spread.keys())

    def get(self, cid: int) -> float:
        return self.spread.get(cid, 0.0)

    def inferred(self, n: int = 5) -> List[Tuple[int, float]]:
        """Concepts that the spread lit up but the input did not.

        This is the associative payoff, isolated: everything here is
        something ASH connected rather than something it was told.
        """
        out = [(cid, v) for cid, v in self.spread.items()
               if cid not in self.initial]
        return sorted(out, key=lambda kv: -kv[1])[:n]
