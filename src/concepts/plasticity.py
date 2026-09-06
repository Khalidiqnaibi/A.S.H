"""
src/concepts/plasticity.py

Rewiring. "Neurons that fire together wire together" -- with the four
corrections that stop that sentence from producing a useless graph.

Naive Hebb, w += eta * a_i * a_j, fails in four ways that are all worth
naming because each one has a specific fix here:

  1. RUNAWAY. Weights grow without bound and the strongest edge eventually
     dominates every spread. Fixed by Oja's rule -- the update subtracts a
     term proportional to the post-synaptic activity times the current
     weight, which is mathematically a normalization and keeps ||w|| bounded
     without ever needing a hard clip.

  2. LEARNING BASE RATES INSTEAD OF ASSOCIATIONS. If some concept is active
     in 80% of cycles, everything correlates with it and everything wires to
     it. Fixed by gating on pointwise mutual information: potentiate on
     co-activation *above chance*, not on co-activation. This is the
     correction people most often skip, and it is the one that decides
     whether the graph encodes anything or just encodes frequency.

  3. HUB DOMINATION. Even with PMI, a few nodes accumulate most of the total
     weight. Fixed by homeostatic synaptic scaling: after each pass, every
     node's outgoing weights are multiplicatively rescaled to a fixed budget.
     Multiplicative, not subtractive, because that preserves the *relative*
     strengths the node has learned while capping its total influence -- which
     is exactly what synaptic scaling does in cortex.

  4. NO FORGETTING MEANS NO REWIRING. A graph that only adds edges is not
     rewireable, it is accumulating. Fixed by decay plus pruning: unreinforced
     edges weaken and are deleted below a threshold, which frees the unit to
     wire somewhere else.

Three factors, not two
----------------------
Pure Hebb learns correlation, which is not the same as usefulness. Every
update here is gated by a third factor -- the System 1 critic's TD error,
which the brain package already computes for a different purpose. Positive
error consolidates the associations that were active when things went well;
negative error weakens them. That is the dopamine term, and it is what makes
this learning rather than statistics.

Two timescales
--------------
`potentiate()` runs on every cycle and only touches `w_fast`, which decays
with a time constant of hours. `consolidate()` runs during sleep and is the
only thing that ever writes `w_slow`. So an association forms immediately,
and becomes permanent only if it survives until sleep and the reward signal
agreed with it.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import Activation, Origin, PROTECTED_ORIGINS, SynType
from .network import ConceptNetwork

logger = logging.getLogger("ash.concepts.plasticity")


@dataclass
class PlasticityConfig:
    eta_fast: float = 0.06          # per-cycle potentiation into w_fast
    eta_slow: float = 0.35          # fraction of trace committed at sleep
    fast_tau_s: float = 5400.0      # w_fast half-life, ~90 min
    oja_gain: float = 0.9           # strength of the normalizing term
    pmi_floor: float = 0.15         # below this PMI, no potentiation at all
    weight_budget: float = 2.0      # per-node outgoing weight after scaling
    decay_per_sleep: float = 0.06   # multiplicative decay of untouched edges
    prune_below: float = 0.02       # delete edges weaker than this
    merge_similarity: float = 0.93  # merge concepts whose prototypes converge
    min_potentiations: int = 2      # an edge needs this many before it persists
    # Saturation scale for the consolidation trace. Traces accumulate all day,
    # so committing the raw sum means a long day and a short one both peg the
    # weight at its clip and the reward signal is erased. Compressing through
    # tanh keeps one sleep's update bounded while preserving ordering.
    trace_scale: float = 6.0

    # --- STDP -------------------------------------------------------------
    # Spike-timing-dependent plasticity window, in cycles rather than
    # milliseconds. Pre-before-post potentiates and makes the edge causal;
    # post-before-pre depresses it. This is what turns "these co-occur" into
    # "this leads to that", and without it the graph can only ever represent
    # correlation.
    stdp_window: int = 4
    stdp_potentiation: float = 0.05
    stdp_depression: float = 0.032     # asymmetric: depression is weaker,
                                       # or every edge decays to zero
    causal_threshold: float = 0.55     # order_evidence above this -> CAUSAL

    # --- Development ------------------------------------------------------
    # Plasticity starts high and anneals with experience. A newborn cortex
    # over-produces synapses and prunes hard; an adult one changes slowly.
    # Without this, ASH either learns too slowly to be useful on day one or
    # stays permanently unstable.
    infant_multiplier: float = 3.0
    maturity_cycles: int = 4000

    # --- Adaptation --------------------------------------------------------
    # Neurons habituate: a stimulus that is constantly present stops driving
    # spikes, and therefore stops driving plasticity. This is the mechanism
    # that protects the graph from ubiquitous concepts, and it belongs here
    # rather than in the PMI gate because it also has to cover STDP -- spike
    # timing has no notion of mutual information, so without adaptation an
    # always-on concept forms strong timing edges to literally everything.
    adaptation_strength: float = 0.92

    # --- Schema fast-track -------------------------------------------------
    # An association consistent with existing structure consolidates in far
    # fewer exposures than one that stands alone (Tse et al. 2007).
    schema_bonus: float = 2.0
    max_synapses: int = 120_000
    reward_floor: float = -1.0
    reward_ceiling: float = 1.0


class Plasticity:
    """Owns all modification of the graph's weights."""

    def __init__(self, net: ConceptNetwork, cfg: Optional[PlasticityConfig] = None):
        self.net = net
        self.cfg = cfg or PlasticityConfig()

        # Eligibility traces: co-activation accumulated since the last sleep,
        # awaiting a reward signal and consolidation.
        self.traces: Dict[Tuple[int, int], float] = {}
        # Reward-weighted traces, updated when a TD error arrives.
        self.credited: Dict[Tuple[int, int], float] = {}

        self._pending: List[Tuple[Tuple[int, int], float]] = []
        # Recent activation history for STDP. Holds the last `stdp_window`
        # activation sets, so ordering across cycles is visible.
        self._recent: List[Dict[int, float]] = []
        self.neuromod = None          # set by ConceptSystem
        # Replay mode swaps the gating rule. PMI and adaptation exist to
        # protect against a continuous stream where some concepts are always
        # present; replay is discrete episodes, where "this pair appeared
        # together" is already a filtered fact -- the hippocampus decided it
        # was worth encoding. So during replay the gate becomes the episode's
        # own strength rather than the streaming statistics, which otherwise
        # read every replayed concept as ubiquitous and block everything.
        self.replay_mode = False
        self.replay_gate = 1.0
        # STDP can be switched off. It must be, during any replay whose order
        # is not the order things were actually experienced in -- otherwise
        # the network learns causality from the replay schedule, which is
        # noise, and every edge in the graph ends up marked causal.
        self.stdp_enabled = True
        self.potentiations = 0
        self.consolidations = 0
        self.last_report: Dict[str, Any] = {}

    # ==================================================================
    # Developmental annealing
    # ==================================================================
    def maturity(self) -> float:
        """0 = newborn, 1 = mature. Anneals with lifetime experience."""
        return float(min(1.0, self.net.cycles / max(1, self.cfg.maturity_cycles)))

    def developmental_gain(self) -> float:
        """Learning rate multiplier from developmental stage.

        Starts at `infant_multiplier` and decays to 1.0. This is what lets
        ASH be useful on its first day without being permanently volatile:
        the first few hundred experiences move the graph a great deal, and
        by the time it has seen a few thousand it takes real evidence to
        change anything.
        """
        m = self.maturity()
        return 1.0 + (self.cfg.infant_multiplier - 1.0) * (1.0 - m) ** 2

    def _gain(self) -> float:
        """Total learning-rate multiplier: development x neuromodulation."""
        g = self.developmental_gain()
        if self.neuromod is not None:
            g *= self.neuromod.plasticity_gain()
        return g

    def adapted(self, cid: int, a: float) -> float:  # noqa: D401
        """Activation as plasticity sees it, after habituation.

        A concept that fires in nearly every cycle is almost invisible to
        learning; one that fires rarely has its full effect. This is why you
        do not form associations with the hum of your own refrigerator.
        """
        if self.replay_mode:
            return a
        c = self.net.concepts.get(cid)
        if c is None or self.net.cycles < 5:
            return a
        rate = min(1.0, c.activations / max(1, self.net.cycles))
        return a * (1.0 - self.cfg.adaptation_strength * rate)

    # ==================================================================
    # PMI
    # ==================================================================
    def pmi(self, a: int, b: int) -> float:
        """Pointwise mutual information of two concepts co-activating.

        Positive means they co-occur more than independence predicts. The
        gate below rejects everything else, which is what keeps ubiquitous
        concepts from wiring to the entire graph.

        One consequence is worth knowing before it surprises you in a debug
        session: two concepts that fire in EVERY cycle have PMI of exactly
        zero and will never wire together, no matter how many times you
        present them. That is correct -- if both are always on, knowing one
        tells you nothing about the other -- and it is the same property that
        stops a concept like "the user typed something" from associating with
        the entire vocabulary. But it does mean a unit with a base rate near
        1.0 is effectively inert as far as plasticity is concerned. If that
        happens to a unit you care about, the fix is to split it into
        something that discriminates, not to lower `pmi_floor`.
        """
        net = self.net
        n = max(1, net.cycles)
        ca = net.concepts.get(a)
        cb = net.concepts.get(b)
        if ca is None or cb is None:
            return 0.0
        key = (a, b) if a < b else (b, a)
        n_ab = net.co_counts.get(key, 0)
        if n_ab < 2:
            return 0.0
        p_ab = n_ab / n
        p_a = max(1e-9, ca.activations / n)
        p_b = max(1e-9, cb.activations / n)
        return float(math.log(p_ab / (p_a * p_b) + 1e-12))

    # ==================================================================
    # Runtime: fast potentiation
    # ==================================================================
    def potentiate(self, act: Activation, now: Optional[float] = None) -> int:
        """One cycle of Hebbian learning over the active set.

        Cost is O(k^2) in the number of active concepts -- with k around 12
        that is ~140 dict operations, which is why sparsity was worth
        enforcing upstream.
        """
        now = now or time.time()

        # STDP first, and unconditionally. It is inherently cross-cycle: a
        # cycle with one active concept still carries ordering information
        # about the cycle before it. Running it after the "fewer than two
        # active" guard below meant the single-concept case -- precisely the
        # clean sequential signal STDP is for -- was silently skipped.
        if self.stdp_enabled:
            self._stdp(act)

        ids = [c for c in act.spread if c in self.net.concepts]
        if len(ids) < 2:
            return 0

        self._decay_fast(now)
        gain = self._gain()
        touched = 0
        for i, a in enumerate(ids):
            act_a = self.adapted(a, act.spread[a])
            if act_a < 0.05:
                continue
            for b in ids[i + 1:]:
                act_b = self.adapted(b, act.spread[b])
                if act_b < 0.05:
                    continue

                if self.replay_mode:
                    gate = self.replay_gate
                else:
                    gate = self.pmi(a, b)
                    if gate < self.cfg.pmi_floor:
                        continue
                    # Bounded so a single very high PMI (two concepts that
                    # have only ever appeared together) cannot dominate.
                    gate = min(2.5, gate)
                coact = act_a * act_b * gate

                for (s, d, pre, post) in ((a, b, act_a, act_b), (b, a, act_b, act_a)):
                    syn = self.net.synapse(s, d)
                    if syn.frozen:
                        continue
                    w = syn.weight()
                    # Oja: Hebbian term minus post * w. The subtraction is
                    # what bounds the weight -- as w grows, the update shrinks
                    # and reaches equilibrium instead of diverging.
                    lr = self.cfg.eta_fast * gain * syn.rules()["lr"]
                    delta = lr * post * (pre - self.cfg.oja_gain * post * w)
                    syn.w_fast += delta
                    syn.last_potentiated = now
                    syn.potentiations += 1
                    touched += 1

                key = (a, b) if a < b else (b, a)
                self.traces[key] = self.traces.get(key, 0.0) + coact
                self._pending.append((key, coact))

        self.potentiations += 1
        if len(self.net.synapses) > self.cfg.max_synapses:
            self._emergency_prune()
        return touched

    def _stdp(self, act: Activation):
        """Spike-timing-dependent plasticity across recent cycles.

        For each concept active now, look back over the window: anything that
        was active *before* it gets a potentiated forward edge and its
        order_evidence increases; anything active only *after* gets depressed.
        Accumulated order evidence promotes the edge to CAUSAL, at which point
        spreading activation weights it more heavily.

        The asymmetry (potentiation stronger than depression) is not a
        detail: with symmetric windows every edge random-walks to zero, and
        the whole mechanism does nothing.
        """
        current = act.spread
        for lag, past in enumerate(reversed(self._recent), start=1):
            # Nearer in time = stronger effect, as in the biological window.
            scale = math.exp(-(lag - 1) / max(1.0, self.cfg.stdp_window / 2))
            for pre_id, raw_pre in past.items():
                if pre_id not in self.net.concepts:
                    continue
                pre_a = self.adapted(pre_id, raw_pre)
                if pre_a < 0.1:
                    continue
                for post_id, raw_post in current.items():
                    if post_id == pre_id or post_id not in self.net.concepts:
                        continue
                    post_a = self.adapted(post_id, raw_post)
                    if post_a < 0.1:
                        continue

                    fwd = self.net.synapse(pre_id, post_id)
                    if not fwd.frozen:
                        d = self.cfg.stdp_potentiation * scale * pre_a * post_a
                        fwd.w_fast += d
                        fwd.order_evidence = min(2.0, fwd.order_evidence + 0.8 * d)
                        if (fwd.order_evidence >= self.cfg.causal_threshold
                                and fwd.kind == SynType.ASSOC):
                            fwd.kind = SynType.CAUSAL
                            logger.debug("Edge %s->%s promoted to CAUSAL",
                                         self.net.describe(pre_id),
                                         self.net.describe(post_id))

                    # The reverse edge is depressed: post-before-pre is
                    # evidence against that direction.
                    rev = self.net.synapses.get((post_id, pre_id))
                    if rev is not None and not rev.frozen:
                        dep = self.cfg.stdp_depression * scale * pre_a * post_a
                        rev.w_fast -= dep
                        # The depression of order evidence must scale with lag
                        # and activation exactly as the potentiation does. A
                        # flat constant here almost exactly cancelled the
                        # potentiation in any cyclic sequence (A->B->C->A,
                        # where every pair is both before and after the
                        # other), so ordering never accumulated and no edge
                        # was ever promoted to causal.
                        rev.order_evidence = max(-2.0, rev.order_evidence - 0.3 * dep)

        self._recent.append(dict(current))
        if len(self._recent) > self.cfg.stdp_window:
            self._recent.pop(0)

    def _decay_fast(self, now: float):
        """Exponential decay of w_fast. Applied lazily per synapse using its
        own last-potentiated stamp, so this costs nothing on the hot path for
        the edges that are not active."""
        tau = self.cfg.fast_tau_s
        for syn in self.net.synapses.values():
            if syn.w_fast == 0.0 or syn.frozen:
                continue
            dt = now - (syn.last_potentiated or now)
            if dt > 60:
                syn.w_fast *= math.exp(-dt / tau)
                syn.last_potentiated = now
                if abs(syn.w_fast) < 1e-4:
                    syn.w_fast = 0.0

    # ==================================================================
    # The third factor
    # ==================================================================
    def reward(self, td_error: float, decay: float = 0.7):
        """Apply a reward signal to everything recently active.

        `td_error` is the System 1 critic's prediction error. Traces are
        credited with exponentially decreasing weight going backwards in
        time, so the association active at the moment of the outcome gets
        most of the credit -- standard eligibility-trace credit assignment.
        """
        r = float(np.clip(td_error, self.cfg.reward_floor, self.cfg.reward_ceiling))
        if abs(r) < 1e-3 or not self._pending:
            self._pending.clear()
            return 0

        weight = 1.0
        n = 0
        for key, coact in reversed(self._pending):
            self.credited[key] = self.credited.get(key, 0.0) + r * coact * weight
            weight *= decay
            n += 1
            if weight < 0.02:
                break
        self._pending.clear()
        return n

    # ==================================================================
    # Sleep: consolidation
    # ==================================================================
    def consolidate(self, evaluator: Optional[Callable[[], float]] = None,
                    regression_margin: float = 0.05) -> Dict[str, Any]:
        """Commit traces to w_slow, decay, prune, merge, rescale.

        If `evaluator` is supplied it is called before and after; a drop
        larger than `regression_margin` reverts the whole pass. A system that
        rewires itself every night without this will eventually degrade, and
        you will have no way to find out which night it started.
        """
        t0 = time.perf_counter()
        before_score = None
        snapshot = None
        if evaluator is not None:
            try:
                before_score = float(evaluator())
                snapshot = self._snapshot()
            except Exception:
                logger.exception("Pre-consolidation evaluation failed; proceeding unguarded")

        report: Dict[str, Any] = {"traces": len(self.traces),
                                  "credited": len(self.credited)}

        committed = self._commit_traces(schema_lookup=getattr(self, "_schema_lookup", None))
        report["committed"] = committed
        report["decayed"] = self._decay_slow()
        report["scaled"] = self._homeostatic_scaling()
        report["pruned"] = self._prune()
        report["merged"] = self._merge_converged()

        self.traces.clear()
        self.credited.clear()
        self._pending.clear()
        self.consolidations += 1

        if evaluator is not None and snapshot is not None:
            try:
                after_score = float(evaluator())
                report["score_before"] = round(before_score, 4)
                report["score_after"] = round(after_score, 4)
                if after_score < before_score - regression_margin:
                    self._restore(snapshot)
                    report["ROLLED_BACK"] = True
                    logger.error(
                        "Consolidation regressed routing accuracy %.3f -> %.3f; reverted",
                        before_score, after_score)
                else:
                    logger.info("Consolidation kept: accuracy %.3f -> %.3f",
                                before_score, after_score)
            except Exception:
                logger.exception("Post-consolidation evaluation failed; keeping changes")

        report["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.last_report = report
        logger.info("Concept consolidation: %s", report)
        return report

    def _commit_traces(self, schema_lookup=None) -> int:
        """Move accumulated co-activation into the persistent weight.

        Reward-credited traces are committed at full strength; uncredited
        ones at a fraction. So associations that merely co-occurred still
        consolidate slowly (you do learn things nobody rewarded you for), but
        associations that preceded a good outcome consolidate much faster.
        """
        n = 0
        for key, trace in self.traces.items():
            credit = self.credited.get(key, 0.0)
            # Unrewarded learning is real but weak; rewarded learning
            # dominates. A purely reward-gated rule would fail to form
            # obvious factual associations that never led to a tool call.
            # tanh-compressed, so a single sleep can move a weight by a
            # bounded amount however busy the day was. Without this, w_slow
            # hits its clip on day one and stops encoding anything.
            strength = (0.25 * math.tanh(trace / self.cfg.trace_scale)
                        + 1.0 * math.tanh(credit / self.cfg.trace_scale))

            # Schema fast-track: an association that fits existing structure
            # consolidates far faster than one standing alone. This is the
            # mechanism behind "the tenth example in a domain is easier than
            # the first", and it is what stops slow cortical learning from
            # being uniformly slow forever.
            if schema_lookup is not None:
                sch = schema_lookup(key)
                if sch > 0.4:
                    strength *= 1.0 + (self.cfg.schema_bonus - 1.0) * sch
            if abs(strength) < 1e-4:
                continue
            a, b = key
            for (s, d) in ((a, b), (b, a)):
                syn = self.net.synapses.get((s, d))
                if syn is None or syn.frozen:
                    continue
                if syn.potentiations < self.cfg.min_potentiations and credit <= 0:
                    continue    # one coincidence is not a memory
                cortical = (self.neuromod.cortical_gain()
                            if self.neuromod is not None else 1.0)
                syn.w_slow += self.cfg.eta_slow * strength * cortical * syn.rules()["lr"]
                syn.w_slow = float(np.clip(syn.w_slow, -1.5, 1.5))
                # Consolidated: the fast component has done its job.
                syn.w_fast *= 0.3
                n += 1
        return n

    def _decay_slow(self) -> int:
        """Forgetting. Every non-frozen edge weakens each sleep; only
        reinforcement keeps it alive."""
        n = 0
        base = self.cfg.decay_per_sleep
        for syn in self.net.synapses.values():
            if syn.frozen:
                continue
            # Causal and route edges forget more slowly than coincidences --
            # a learned sequence is worth more than a co-occurrence and
            # should not evaporate at the same rate.
            syn.w_slow *= 1.0 - base * syn.rules()["decay"]
            n += 1
        return n

    def _homeostatic_scaling(self) -> int:
        """Cap each node's total outgoing influence, multiplicatively."""
        budget = self.cfg.weight_budget
        scaled = 0
        for src, edges in self.net._out.items():
            movable = [s for s in edges.values() if not s.frozen]
            if not movable:
                continue
            total = sum(abs(s.weight()) for s in edges.values())
            if total <= budget:
                continue
            factor = budget / total
            for s in movable:
                s.w_slow *= factor
                s.w_fast *= factor
            scaled += 1
        return scaled

    def _prune(self) -> int:
        """Delete edges that have fallen below usefulness. This is the half
        of 'rewireable' that additive learning cannot provide."""
        dead = [k for k, s in self.net.synapses.items()
                if not s.frozen and abs(s.weight()) < self.cfg.prune_below]
        for k in dead:
            self.net.synapses.pop(k, None)
            out = self.net._out.get(k[0])
            if out:
                out.pop(k[1], None)
        return len(dead)

    def _emergency_prune(self):
        """Hard cap. Keeps the strongest edges and drops the rest."""
        keep = self.cfg.max_synapses // 2
        ranked = sorted(self.net.synapses.items(),
                        key=lambda kv: (kv[1].frozen, abs(kv[1].weight())), reverse=True)
        survivors = dict(ranked[:keep])
        self.net.synapses = survivors
        self.net._out = {}
        for (s, d), syn in survivors.items():
            self.net._out.setdefault(s, {})[d] = syn
        logger.warning("Synapse cap hit; pruned to %d strongest", len(survivors))

    def _merge_converged(self) -> int:
        """Merge concepts whose prototypes have drifted together.

        Growth by resonance is generous by design -- it is better to allocate
        a spurious unit than to blur two real distinctions. Merging is what
        makes that safe: duplicates created on different days by slightly
        different phrasings collapse back into one during sleep, so the
        vocabulary tracks the diversity of your life rather than the number
        of things you have said.
        """
        net = self.net
        net._rebuild()
        if net._matrix is None or net._matrix.shape[0] < 2:
            return 0

        ids = list(net._ids)
        M = net._matrix
        sims = M @ M.T
        np.fill_diagonal(sims, -1.0)

        merged = 0
        gone: set = set()
        pairs = np.argwhere(sims >= self.cfg.merge_similarity)
        for i, j in pairs:
            if i >= j:
                continue
            a, b = ids[int(i)], ids[int(j)]
            if a in gone or b in gone:
                continue
            ca, cb = net.concepts.get(a), net.concepts.get(b)
            if ca is None or cb is None:
                continue
            # Never merge two protected units -- that would silently collapse
            # two distinct tool routes into one.
            if ca.origin in PROTECTED_ORIGINS and cb.origin in PROTECTED_ORIGINS:
                continue
            # The protected one absorbs the other.
            keep, drop = (ca, cb) if cb.origin not in PROTECTED_ORIGINS else (cb, ca)
            self._absorb(keep.id, drop.id)
            gone.add(drop.id)
            merged += 1

        return merged

    def _absorb(self, keep: int, drop: int):
        """Fold one concept's edges and statistics into another."""
        net = self.net
        ck, cd = net.concepts[keep], net.concepts[drop]

        # Prototype becomes the activation-weighted mean.
        wk = max(1, ck.activations)
        wd = max(1, cd.activations)
        p = (ck.prototype * wk + cd.prototype * wd) / (wk + wd)
        n = float(np.linalg.norm(p))
        if n > 1e-9:
            ck.prototype = (p / n).astype(np.float32)
        ck.activations += cd.activations
        ck.activation_sum += cd.activation_sum
        if ck.origin == Origin.GROWN and cd.origin != Origin.GROWN:
            ck.origin = Origin.MERGED

        for (s, d), syn in list(net.synapses.items()):
            if s == drop or d == drop:
                ns, nd = (keep if s == drop else s), (keep if d == drop else d)
                if ns == nd:
                    continue
                tgt = net.synapse(ns, nd)
                if not tgt.frozen:
                    tgt.w_slow = max(tgt.w_slow, syn.w_slow)
                    tgt.w_fast = max(tgt.w_fast, syn.w_fast)
                    tgt.potentiations += syn.potentiations
        net.remove_concept(drop)

    # ==================================================================
    # Rollback support
    # ==================================================================
    def _snapshot(self) -> Dict[str, Any]:
        return {
            "weights": {k: (s.w_slow, s.w_fast) for k, s in self.net.synapses.items()},
            "protos": {c.id: c.prototype.copy() for c in self.net.concepts.values()},
        }

    def _restore(self, snap: Dict[str, Any]):
        for k, (slow, fast) in snap["weights"].items():
            syn = self.net.synapses.get(k)
            if syn is None:
                a, b = k
                if a in self.net.concepts and b in self.net.concepts:
                    syn = self.net.synapse(a, b)
                else:
                    continue
            syn.w_slow, syn.w_fast = slow, fast
        for cid, proto in snap["protos"].items():
            c = self.net.concepts.get(cid)
            if c is not None:
                c.prototype = proto
        self.net._dirty = True

    # ==================================================================
    def status(self) -> Dict[str, Any]:
        return {
            "potentiation_cycles": self.potentiations,
            "consolidations": self.consolidations,
            "maturity": round(self.maturity(), 3),
            "developmental_gain": round(self.developmental_gain(), 3),
            "causal_edges": sum(1 for s in self.net.synapses.values()
                                if s.kind == SynType.CAUSAL),
            "open_traces": len(self.traces),
            "credited_traces": len(self.credited),
            "last_consolidation": self.last_report,
        }
