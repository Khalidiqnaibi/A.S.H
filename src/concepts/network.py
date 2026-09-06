"""
src/concepts/network.py

The network itself: units, edges, activation, and growth.

Four mechanisms, in the order they run on each input.

1. k-WINNERS-TAKE-ALL
   Cosine the input against every prototype, keep the top k above a floor,
   discard the rest. This is lateral inhibition, and it is what makes the
   representation sparse. Sparsity is not an optimization here -- it is what
   makes Hebbian learning tractable (k^2 updates instead of N^2) and what
   stops every concept from weakly associating with every other one.

2. SPREADING ACTIVATION
   Push activation along synapses for a couple of steps, resparsifying after
   each. Two steps is deliberate: one step gives you direct associates, two
   gives you "the thing those have in common", three gives you noise. Every
   spreading-activation system that has ever been built converges on 2-3.

3. RESONANCE GROWTH
   If nothing matches well enough (best cosine below `vigilance`), the input
   is unfamiliar and a new unit is allocated on the spot, centered on it.
   This is adaptive resonance, and the vigilance parameter is the single knob
   controlling how coarse the concept vocabulary ends up. Low vigilance gives
   a few broad concepts; high gives thousands of narrow ones.

4. PRIMING
   Activation does not reset between inputs. A decaying trace carries into
   the next cycle as a baseline, so recent context biases what lights up.
   This is why "and what about tomorrow?" resolves -- `tomorrow` arrives into
   a network where `schedule` is still warm.

Prototype drift
---------------
Concepts move toward the inputs that fire them. Seeds from the intent catalog
therefore stop being wherever the catalog put them and migrate toward how this
particular user actually phrases things. That is most of the practical value:
the flat classifier is stuck with its catalog forever, this isn't.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .model import (
    Activation, Concept, Origin, PROTECTED_ORIGINS, Synapse,
)

logger = logging.getLogger("ash.concepts.network")

STATE_DIR = os.environ.get("ASH_CONCEPT_STATE",
                           os.path.join(os.getcwd(), "state", "concepts"))


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


class ConceptNetwork:
    def __init__(
        self,
        dim: int = 384,
        k: int = 12,                   # winners per input
        activation_floor: float = 0.22,
        vigilance: float = 0.62,       # below this similarity -> grow a unit
        spread_steps: int = 2,
        spread_gain: float = 0.55,     # how much of the signal propagates
        spread_keep: int = 24,         # resparsify to this many after spreading
        inferred_cap: float = 0.85,    # inference can never be as strong as perception
        prime_decay: float = 0.55,     # per-cycle carryover of activation
        max_concepts: int = 4096,
        persist: bool = True,
    ):
        self.dim = dim
        self.k = k
        self.activation_floor = activation_floor
        self.vigilance = vigilance
        self.spread_steps = spread_steps
        self.spread_gain = spread_gain
        self.spread_keep = spread_keep
        self.inferred_cap = inferred_cap
        self.prime_decay = prime_decay
        self.max_concepts = max_concepts
        self.persist = persist

        self.concepts: Dict[int, Concept] = {}
        self.synapses: Dict[Tuple[int, int], Synapse] = {}
        self._out: Dict[int, Dict[int, Synapse]] = {}   # adjacency, src -> dst
        self._next_id = 0

        # Prototype matrix, kept in sync with `concepts` so activation is one
        # matmul rather than a Python loop. Rebuilt lazily on structural change.
        self._matrix: Optional[np.ndarray] = None
        self._ids: List[int] = []
        self._dirty = True

        # Priming trace.
        self._prime: Dict[int, float] = {}

        # Co-occurrence statistics for the PMI correction in plasticity.py.
        self.cycles = 0
        self.co_counts: Dict[Tuple[int, int], int] = {}

        self.last: Optional[Activation] = None
        self._by_label: Dict[str, int] = {}
        self._by_intent: Dict[str, int] = {}

    # ==================================================================
    # Structure
    # ==================================================================
    def add_concept(self, prototype: np.ndarray, label: Optional[str] = None,
                    origin: Origin = Origin.GROWN, intent: Optional[str] = None,
                    seed_text: str = "", frozen: bool = False) -> Concept:
        if len(self.concepts) >= self.max_concepts:
            victim = self._weakest_prunable()
            if victim is None:
                raise RuntimeError("concept capacity reached and nothing is prunable")
            self.remove_concept(victim)

        c = Concept(
            id=self._next_id, prototype=_l2(np.asarray(prototype, dtype=np.float32)),
            label=label, origin=origin, intent=intent, seed_text=seed_text,
            frozen=frozen,
            # Seeded concepts adapt more slowly than grown ones: they encode
            # something we asserted, and letting them drift fast would erase
            # the catalog within a week of heavy use.
            plasticity=0.02 if origin in PROTECTED_ORIGINS else 0.08,
        )
        self._next_id += 1
        self.concepts[c.id] = c
        if label:
            self._by_label[label] = c.id
        if intent:
            self._by_intent[intent] = c.id
        self._dirty = True
        return c

    def remove_concept(self, cid: int):
        c = self.concepts.pop(cid, None)
        if c is None:
            return
        if c.label:
            self._by_label.pop(c.label, None)
        if c.intent:
            self._by_intent.pop(c.intent, None)
        for (s, d) in [k for k in self.synapses if k[0] == cid or k[1] == cid]:
            self.synapses.pop((s, d), None)
            if s in self._out:
                self._out[s].pop(d, None)
        self._out.pop(cid, None)
        self._prime.pop(cid, None)
        self._dirty = True

    def _weakest_prunable(self) -> Optional[int]:
        cands = [c for c in self.concepts.values()
                 if c.origin not in PROTECTED_ORIGINS and not c.frozen]
        if not cands:
            return None
        # Least used, oldest-inactive first.
        return min(cands, key=lambda c: (c.activations, c.last_active)).id

    def synapse(self, src: int, dst: int, create: bool = True) -> Optional[Synapse]:
        s = self.synapses.get((src, dst))
        if s is None and create:
            s = Synapse(src=src, dst=dst)
            self.synapses[(src, dst)] = s
            self._out.setdefault(src, {})[dst] = s
        return s

    def link(self, a: int, b: int, weight: float, bidirectional: bool = True,
             frozen: bool = False, kind=None):
        """Explicit wiring, used for seeding. Frozen links survive every
        plasticity pass untouched."""
        for (s, d) in ((a, b), (b, a)) if bidirectional else ((a, b),):
            syn = self.synapse(s, d)
            syn.w_slow = float(weight)
            syn.frozen = frozen
            if kind is not None:
                syn.kind = kind

    def by_label(self, label: str) -> Optional[Concept]:
        cid = self._by_label.get(label)
        return self.concepts.get(cid) if cid is not None else None

    def by_intent(self, intent: str) -> Optional[Concept]:
        cid = self._by_intent.get(intent)
        return self.concepts.get(cid) if cid is not None else None

    # ==================================================================
    # Activation
    # ==================================================================
    def _rebuild(self):
        if not self._dirty:
            return
        self._ids = sorted(self.concepts)
        if self._ids:
            self._matrix = np.stack([self.concepts[i].prototype for i in self._ids])
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._dirty = False

    def _kwta(self, vec: np.ndarray) -> Tuple[Dict[int, float], float]:
        """Bottom-up response. Returns (active, best_similarity)."""
        self._rebuild()
        if self._matrix.shape[0] == 0:
            return {}, 0.0

        sims = self._matrix @ vec                     # cosine; both normalized
        best = float(sims.max()) if sims.size else 0.0

        k = min(self.k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        out: Dict[int, float] = {}
        for i in idx:
            s = float(sims[i])
            if s >= self.activation_floor:
                out[self._ids[int(i)]] = s
        return out, best

    def _spread(self, seed: Dict[int, float]) -> Dict[int, float]:
        """Propagate along synapses, resparsifying each step.

        A concept reached only by association is capped below the strongest
        bottom-up match. Without that ceiling, two mutually-connected units
        reverberate to saturation in two steps and an inferred concept ends up
        indistinguishable from a perceived one -- which quietly breaks the
        proposer, since it discounts inferred routes by checking exactly that
        distinction. Inference must stay weaker than perception.
        """
        act = dict(seed)
        ceiling = self.inferred_cap * (max(seed.values()) if seed else 1.0)
        for _ in range(self.spread_steps):
            delta: Dict[int, float] = {}
            for cid, a in act.items():
                edges = self._out.get(cid)
                if not edges:
                    continue
                for dst, syn in edges.items():
                    w = syn.weight()
                    if w == 0.0:
                        continue
                    # Inhibitory edges propagate their negative sign; a frozen
                    # constraint edge must actively suppress the route it
                    # forbids, not merely fail to support it.
                    gain = self.spread_gain * syn.rules()["spread"]
                    delta[dst] = delta.get(dst, 0.0) + gain * a * w

            if not delta:
                break
            for cid, d in delta.items():
                limit = 1.0 if cid in seed else ceiling
                act[cid] = max(0.0, min(limit, act.get(cid, 0.0) + d))

            if len(act) > self.spread_keep:
                act = dict(sorted(act.items(), key=lambda kv: -kv[1])[:self.spread_keep])
        return act

    def perceive(self, vec: np.ndarray, text: str = "",
                 allow_growth: bool = True, learn_prototypes: bool = True) -> Activation:
        """Present one input. This is the hot path -- one matmul plus a
        handful of dict operations."""
        vec = _l2(np.asarray(vec, dtype=np.float32))
        self.cycles += 1

        initial, best = self._kwta(vec)
        novelty = float(max(0.0, 1.0 - best))

        grown = None
        if allow_growth and best < self.vigilance:
            # Resonance failure: nothing in the vocabulary explains this.
            c = self.add_concept(vec, origin=Origin.GROWN, seed_text=text[:200])
            grown = c.id
            initial[c.id] = 1.0
            logger.debug("Grew concept c%d (best match %.2f < vigilance %.2f): %r",
                         c.id, best, self.vigilance, text[:60])

        # Priming: carry a decayed trace of the previous cycle in as baseline.
        for cid, a in self._prime.items():
            if cid in self.concepts:
                initial[cid] = max(initial.get(cid, 0.0), a)

        spread = self._spread(initial)

        now = time.time()
        for cid, a in spread.items():
            c = self.concepts.get(cid)
            if c is None:
                continue
            c.touch(a, now)
            if learn_prototypes and cid in initial:
                # Only bottom-up winners drift. A concept reached purely by
                # association should not move toward an input it never matched.
                c.adapt(vec, strength=a)
        if any(cid in initial for cid in spread):
            self._dirty = True

        self._prime = {cid: a * self.prime_decay for cid, a in spread.items()
                       if a * self.prime_decay > 0.05}

        act = Activation(initial=initial, spread=spread, novelty=novelty,
                         grown=grown, steps=self.spread_steps)
        self.last = act
        self._record_cooccurrence(act)
        return act

    def _record_cooccurrence(self, act: Activation):
        """Counts for the PMI correction. Bounded by capping the table -- an
        always-on process would otherwise accumulate pairs forever."""
        ids = sorted(act.spread)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                key = (a, b)
                self.co_counts[key] = self.co_counts.get(key, 0) + 1
        if len(self.co_counts) > 200_000:
            keep = sorted(self.co_counts.items(), key=lambda kv: -kv[1])[:100_000]
            self.co_counts = dict(keep)

    # ==================================================================
    # Readouts
    # ==================================================================
    def latent(self, act: Optional[Activation] = None, dim: int = 256) -> np.ndarray:
        """Project the sparse activation into a fixed-size dense vector.

        Given to the predictive model in place of the raw sentence embedding.
        Predicting the next *conceptual* state is both a better-posed problem
        and closer to what a JEPA is supposed to be doing than predicting the
        next raw embedding -- the concept space has already thrown away the
        surface form that makes consecutive utterances look unrelated.

        The projection is a fixed seeded random matrix per concept id, so it
        is stable across restarts and new concepts do not disturb old codes.
        """
        act = act or self.last
        out = np.zeros(dim, dtype=np.float32)
        if act is None:
            return out
        for cid, a in act.spread.items():
            rng = np.random.default_rng(0x5EED + cid)
            out += a * rng.standard_normal(dim).astype(np.float32)
        return _l2(out)

    def intent_activation(self, act: Optional[Activation] = None) -> Dict[str, float]:
        """Activation restricted to units that route to a tool."""
        act = act or self.last
        if act is None:
            return {}
        out: Dict[str, float] = {}
        for cid, a in act.spread.items():
            c = self.concepts.get(cid)
            if c is not None and c.intent:
                out[c.intent] = max(out.get(c.intent, 0.0), a)
        return out

    def neighbours(self, cid: int, n: int = 8) -> List[Tuple[int, float]]:
        edges = self._out.get(cid, {})
        return sorted(((d, s.weight()) for d, s in edges.items()),
                      key=lambda kv: -kv[1])[:n]

    def describe(self, cid: int) -> str:
        """Human-readable identity of a unit.

        A grown concept has no name, so it is described the way a cortical
        unit is described: by what fires it and what it connects to.
        """
        c = self.concepts.get(cid)
        if c is None:
            return f"<missing c{cid}>"
        if c.label:
            return c.label
        nb = [self.concepts[d].name() for d, _ in self.neighbours(cid, 3)
              if d in self.concepts]
        if nb:
            return f"c{cid}<{'/'.join(nb)}>"
        return f"c{cid}<{c.seed_text[:32]}>" if c.seed_text else f"c{cid}"

    def explain(self, act: Optional[Activation] = None, n: int = 6) -> Dict[str, Any]:
        act = act or self.last
        if act is None:
            return {}
        return {
            "matched": [f"{self.describe(c)}={a:.2f}"
                        for c, a in sorted(act.initial.items(), key=lambda kv: -kv[1])[:n]],
            "inferred": [f"{self.describe(c)}={a:.2f}" for c, a in act.inferred(n)],
            "novelty": round(act.novelty, 3),
            "grew": self.describe(act.grown) if act.grown is not None else None,
        }

    # ==================================================================
    # Persistence
    # ==================================================================
    def save(self, directory: str = STATE_DIR):
        if not self.persist:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            self._rebuild()
            np.savez_compressed(
                os.path.join(directory, "prototypes.npz"),
                ids=np.array(self._ids, dtype=np.int64),
                protos=self._matrix if self._matrix is not None
                else np.zeros((0, self.dim), dtype=np.float32),
            )
            meta = {
                "dim": self.dim, "next_id": self._next_id, "cycles": self.cycles,
                "concepts": [
                    {**c.as_dict(), "plasticity": c.plasticity}
                    for c in self.concepts.values()
                ],
                "synapses": [s.as_dict() for s in self.synapses.values()
                             if abs(s.w_slow) > 1e-4 or s.frozen],
            }
            with open(os.path.join(directory, "graph.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)
        except Exception:
            logger.exception("Failed to persist concept network")

    def load(self, directory: str = STATE_DIR) -> bool:
        proto_path = os.path.join(directory, "prototypes.npz")
        meta_path = os.path.join(directory, "graph.json")
        if not (os.path.exists(proto_path) and os.path.exists(meta_path)):
            return False
        try:
            d = np.load(proto_path)
            ids = [int(i) for i in d["ids"]]
            protos = d["protos"]
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if int(meta.get("dim", self.dim)) != self.dim:
                logger.warning("Saved concept graph has dim %s, expected %s -- ignoring",
                               meta.get("dim"), self.dim)
                return False

            self.concepts.clear()
            self.synapses.clear()
            self._out.clear()
            self._by_label.clear()
            self._by_intent.clear()

            by_id = {int(c["id"]): c for c in meta.get("concepts", [])}
            for i, cid in enumerate(ids):
                m = by_id.get(cid, {})
                c = Concept(
                    id=cid, prototype=protos[i].astype(np.float32),
                    label=m.get("label"), origin=Origin(m.get("origin", "grown")),
                    intent=m.get("intent"), activations=int(m.get("activations", 0)),
                    seed_text=m.get("seed_text", ""), frozen=bool(m.get("frozen", False)),
                    plasticity=float(m.get("plasticity", 0.05)),
                )
                self.concepts[cid] = c
                if c.label:
                    self._by_label[c.label] = cid
                if c.intent:
                    self._by_intent[c.intent] = cid

            from .model import SynType
            for s in meta.get("synapses", []):
                syn = self.synapse(int(s["src"]), int(s["dst"]))
                syn.w_slow = float(s.get("slow", s.get("w", 0.0)))
                syn.frozen = bool(s.get("frozen", False))
                syn.potentiations = int(s.get("n", 0))
                syn.order_evidence = float(s.get("order", 0.0))
                try:
                    syn.kind = SynType(s.get("kind", "assoc"))
                except ValueError:
                    pass

            self._next_id = int(meta.get("next_id", max(ids, default=-1) + 1))
            self.cycles = int(meta.get("cycles", 0))
            self._dirty = True
            logger.info("Concept network restored: %d units, %d synapses",
                        len(self.concepts), len(self.synapses))
            return True
        except Exception:
            logger.exception("Failed to restore concept network -- starting fresh")
            return False

    # ==================================================================
    def status(self) -> Dict[str, Any]:
        origins: Dict[str, int] = {}
        for c in self.concepts.values():
            origins[c.origin.value] = origins.get(c.origin.value, 0) + 1
        weights = [s.weight() for s in self.synapses.values()]
        kinds: Dict[str, int] = {}
        for s in self.synapses.values():
            kinds[s.kind.value] = kinds.get(s.kind.value, 0) + 1
        return {
            "concepts": len(self.concepts),
            "by_synapse_type": kinds,
            "by_origin": origins,
            "synapses": len(self.synapses),
            "frozen_synapses": sum(1 for s in self.synapses.values() if s.frozen),
            "mean_weight": round(float(np.mean(weights)), 4) if weights else 0.0,
            "max_weight": round(float(np.max(weights)), 4) if weights else 0.0,
            "cycles": self.cycles,
            "primed": len(self._prime),
            "density": round(len(self.synapses) / max(1, len(self.concepts) ** 2), 5),
        }
