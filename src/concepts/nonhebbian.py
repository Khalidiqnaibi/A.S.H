"""
src/concepts/nonhebbian.py

The forms of plasticity that are not "fire together, wire together".

Hebbian learning (plasticity.py) is input-specific and positive-feedback: the
synapses that were active get stronger, which makes them more likely to be
active. Left alone it runs away and it barely makes synapses compete. The
literature (Chistiakova et al. 2014/2015, Zenke & Gerstner 2017, Turrigiano,
Abraham & Bear 1996) is consistent that the brain does not leave it alone, and
that the correction has to act on the same timescale as the learning itself.
Three mechanisms live here.

  1. HETEROSYNAPTIC DEPRESSION  (fast, same timescale as Hebbian)
     After a strong burst of postsynaptic activity, the inputs onto that unit
     which were NOT active get weaker, in proportion to how strong they are.
     This is what makes inputs compete for a unit: when one input is learned,
     the ones that did not take part in the firing lose ground, and it is
     triggered by the same event as the potentiation instead of by an hours-
     long average the way synaptic scaling is. Zenke & Gerstner's point is that
     slow homeostasis alone cannot stabilise Hebbian learning; this is the
     rapid compensatory process it needs. Scaling (plasticity.py, at sleep)
     stays on as the slow fine-tuning layer.

  2. METAPLASTICITY  (a sliding threshold, as in BCM)
     How easily a unit's inputs potentiate depends on how active the unit has
     been lately. A unit that has been firing a lot is harder to potentiate
     further; a quiet one is easier. It is the plasticity of plasticity, and it
     is the second brake on runaway: the more a unit has been used, the harder
     it is to push further.

  3. NEIGHBOURHOOD PLASTICITY  (spatial / cluster spread)
     When a synapse is strengthened, units whose prototypes lie close to its
     endpoints are strengthened with it, in proportion to how close they are.
     Related ground is learned together instead of one phrasing at a time,
     which is how a skill generalises and how a topic forms a region that
     specialises as a whole. Two parts:
       * synaptic spread: the strengthened edge is copied onto the neighbours
         of both its ends, always strictly weaker than the edge that triggered
         it, and only provisionally -- neighbour edges consolidate at sleep
         only if reward credit arrives, because nothing about them was
         actually observed;
       * a prototype pull, as in a self-organising map: the neighbours of a
         winning unit drift a little toward the input, and stop at a ceiling
         similarity so a neighbourhood never collapses into one unit.
     Seeded (protected) units are never pulled, so the classifier's catalog
     cannot be dragged around by it.

All of this runs only on the live path. Replay during sleep is a different
regime (see plasticity.replay_mode) and would read its own schedule as
activity, so it is excluded.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .model import PROTECTED_ORIGINS, SynType
from .network import ConceptNetwork

logger = logging.getLogger("ash.concepts.nonhebbian")


@dataclass
class NonHebbianConfig:
    # --- heterosynaptic depression ----------------------------------------
    hetero_enabled: bool = True
    hetero_trigger: float = 0.60     # postsynaptic activity needed to trigger it
    hetero_rate: float = 0.05        # fraction of an inactive input's weight, per full burst
    hetero_slow_share: float = 0.25  # part of that, also taken from the persistent weight
    hetero_max_posts: int = 4        # strongest units considered per cycle
    hetero_min_weight: float = 0.02  # inputs weaker than this are left to decay
    hetero_active_floor: float = 0.05  # an input at least this active counts as active

    # --- metaplasticity (sliding threshold) ---------------------------------
    meta_enabled: bool = True
    meta_tau: float = 200.0          # cycles over which "recent activity" is averaged
    meta_target: float = 0.15        # activity at which learning runs at its normal rate
    meta_gain: float = 0.5           # how steeply the threshold slides
    meta_min: float = 0.25           # learning-rate multiplier bounds
    meta_max: float = 2.0

    # --- neighbourhood plasticity -------------------------------------------
    neigh_enabled: bool = True
    neigh_floor: float = 0.55        # prototype cosine at which a unit stops being a neighbour
    neigh_max: int = 4               # neighbours considered per triggering unit
    neigh_gain: float = 0.35         # share of the trigger's change a neighbour gets, at most
    neigh_max_triggers: int = 3      # strongest strengthened edges that spread, per cycle
    neigh_new_edges: bool = True     # allow a neighbour edge that did not exist
    neigh_pull: float = 0.02         # prototype pull strength for neighbours of a winner
    neigh_pull_ceiling: float = 0.88 # neighbours are never pulled closer than this
    neigh_winner_floor: float = 0.30 # a bottom-up match must be at least this strong to spread


class NonHebbian:
    """Owned by `Plasticity`, which calls it at three points in a cycle."""

    def __init__(self, net: ConceptNetwork, cfg: Optional[NonHebbianConfig] = None):
        self.net = net
        self.cfg = cfg or NonHebbianConfig()
        self.stats: Dict[str, int] = {
            "hetero_events": 0, "hetero_inputs": 0,
            "neigh_events": 0, "neigh_edges": 0, "neigh_new_edges": 0,
            "prototype_pulls": 0,
        }

    # ==================================================================
    # Metaplasticity
    # ==================================================================
    def _alpha(self) -> float:
        return 1.0 / max(1.0, self.cfg.meta_tau)

    def current_rate(self, cid: int) -> float:
        """Recent activity of a unit, decayed up to now without touching it."""
        c = self.net.concepts.get(cid)
        if c is None:
            return 0.0
        dt = max(0, self.net.cycles - c.rate_cycle)
        return float(c.rate * (1.0 - self._alpha()) ** dt)

    def meta_scale(self, cid: int) -> float:
        """Learning-rate multiplier for inputs onto `cid`.

        1.0 at the target activity, below it for a unit that has been busy and
        above it for one that has been quiet, within [meta_min, meta_max].
        """
        if not self.cfg.meta_enabled:
            return 1.0
        cfg = self.cfg
        r = self.current_rate(cid)
        s = math.exp(-cfg.meta_gain * (r - cfg.meta_target) / max(1e-6, cfg.meta_target))
        return float(min(cfg.meta_max, max(cfg.meta_min, s)))

    def observe(self, adapted: Dict[int, float]):
        """Fold this cycle's activity into every active unit's running rate."""
        if not self.cfg.meta_enabled:
            return
        alpha = self._alpha()
        now_cycle = self.net.cycles
        for cid, a in adapted.items():
            c = self.net.concepts.get(cid)
            if c is None:
                continue
            dt = max(1, now_cycle - c.rate_cycle)
            c.rate = float(c.rate * (1.0 - alpha) ** dt + alpha * min(1.0, max(0.0, a)))
            c.rate_cycle = now_cycle

    # ==================================================================
    # Heterosynaptic depression
    # ==================================================================
    def heterosynaptic(self, adapted: Dict[int, float], active: Dict[int, float],
                       now: float) -> int:
        """Depress the inactive inputs onto each strongly active unit.

        Returns how many synapses were depressed. Never touches a frozen
        synapse, and only learned kinds (associative and causal): a route into
        an intent or an inhibitory edge is structure, not something to erode
        because another route happened to fire.
        """
        cfg = self.cfg
        if not cfg.hetero_enabled:
            return 0
        posts = sorted(((a, cid) for cid, a in adapted.items() if a >= cfg.hetero_trigger),
                       reverse=True)[:cfg.hetero_max_posts]
        if not posts:
            return 0

        kinds = (SynType.ASSOC, SynType.CAUSAL)
        hit = 0
        for a, post in posts:
            excess = min(1.0, (a - cfg.hetero_trigger) / max(1e-6, 1.0 - cfg.hetero_trigger))
            # A burst exactly at the trigger depresses nothing; a full one
            # depresses at the full rate. Between, it scales.
            if excess <= 0.0:
                continue
            cost = cfg.hetero_rate * excess
            for src, syn in self.net.inputs(post).items():
                if syn.frozen or syn.kind not in kinds:
                    continue
                if active.get(src, 0.0) >= cfg.hetero_active_floor:
                    continue
                w = syn.weight()
                if w <= cfg.hetero_min_weight:
                    continue
                d = cost * w
                syn.w_fast -= d
                if syn.w_slow > 0.0:
                    syn.w_slow = max(0.0, syn.w_slow - cfg.hetero_slow_share * d)
                if not syn.last_potentiated:
                    syn.last_potentiated = now      # so the fast part decays back
                hit += 1
        if hit:
            self.stats["hetero_events"] += 1
            self.stats["hetero_inputs"] += hit
        return hit

    # ==================================================================
    # Neighbourhood plasticity
    # ==================================================================
    def _share(self, sim: float) -> float:
        """How much of a trigger's change a neighbour at cosine `sim` receives:
        zero at the floor, the full gain at identity."""
        cfg = self.cfg
        if sim <= cfg.neigh_floor:
            return 0.0
        return float(cfg.neigh_gain * (sim - cfg.neigh_floor) / max(1e-6, 1.0 - cfg.neigh_floor))

    def neighbours_of(self, cid: int, exclude: Optional[Set[int]] = None
                      ) -> List[Tuple[int, float]]:
        """Units whose prototypes lie close to `cid`'s, nearest first.

        Frozen units are not neighbours: they are exempt from learning.
        """
        cfg = self.cfg
        net = self.net
        net._rebuild()
        if net._matrix is None or net._matrix.shape[0] < 2:
            return []
        pos = bisect.bisect_left(net._ids, cid)
        if pos >= len(net._ids) or net._ids[pos] != cid:
            return []
        sims = net._matrix @ net._matrix[pos]
        order = np.argsort(-sims)
        out: List[Tuple[int, float]] = []
        for i in order:
            s = float(sims[int(i)])
            if s <= cfg.neigh_floor:
                break
            nid = net._ids[int(i)]
            if nid == cid or (exclude and nid in exclude):
                continue
            c = net.concepts.get(nid)
            if c is None or c.frozen:
                continue
            out.append((nid, s))
            if len(out) >= cfg.neigh_max:
                break
        return out

    def spread_synapses(self, updates: List[Tuple[int, int, float, float]],
                        active: Set[int], now: float
                        ) -> List[Tuple[Tuple[int, int], float]]:
        """Copy the strongest Hebbian changes of this cycle onto neighbours.

        `updates` is (src, dst, delta, coactivation) for each synapse that was
        potentiated. Returns (pair key, credit) entries for the caller to add
        to its eligibility traces, so neighbour edges can be rewarded and
        consolidated -- but, since no synapse of theirs was actually
        potentiated, only if reward credit arrives.
        """
        cfg = self.cfg
        if not cfg.neigh_enabled or not updates:
            return []
        triggers = sorted((u for u in updates if u[2] > 0.0), key=lambda u: -u[2])
        triggers = triggers[:cfg.neigh_max_triggers]
        credits: List[Tuple[Tuple[int, int], float]] = []
        touched_any = False
        cache: Dict[int, List[Tuple[int, float]]] = {}

        def near(x: int) -> List[Tuple[int, float]]:
            if x not in cache:
                cache[x] = self.neighbours_of(x, exclude=active)
            return cache[x]

        for (s, d, delta, coact) in triggers:
            src_syn = self.net.synapses.get((s, d))
            if src_syn is None:
                continue
            ref = src_syn.weight()
            if ref <= 1e-6:
                continue
            targets: List[Tuple[int, int, float]] = []
            for n, sim in near(s):                       # neighbours of the source end
                targets.append((n, d, sim))
            for n, sim in near(d):                       # neighbours of the target end
                targets.append((s, n, sim))
            for (ts, td, sim) in targets:
                if ts == td:
                    continue
                share = self._share(sim)
                if share <= 0.0:
                    continue
                syn = self.net.synapses.get((ts, td))
                if syn is None:
                    if not cfg.neigh_new_edges:
                        continue
                    syn = self.net.synapse(ts, td)
                    self.stats["neigh_new_edges"] += 1
                if syn.frozen or syn.kind not in (SynType.ASSOC, SynType.CAUSAL):
                    continue
                # A neighbour edge never reaches the strength of the edge that
                # triggered it: the step is bounded by the gap still to close.
                w = syn.weight()
                gap = ref - w
                inc = min(share * delta, share * gap)
                if inc <= 1e-9:
                    continue
                syn.w_fast += inc
                if not syn.last_potentiated:
                    syn.last_potentiated = now
                self.stats["neigh_edges"] += 1
                touched_any = True
                key = (ts, td) if ts < td else (td, ts)
                credits.append((key, share * coact))
        if touched_any:
            self.stats["neigh_events"] += 1
        return credits

    def pull_prototypes(self, vec: np.ndarray, initial: Dict[int, float]) -> int:
        """Self-organising-map step: neighbours of the winning units move a
        little toward the input, up to a ceiling similarity.

        Only units that are not protected and not frozen are moved, so seeded
        intents (the classifier's catalog) are never dragged by this.
        """
        cfg = self.cfg
        if not cfg.neigh_enabled or cfg.neigh_pull <= 0.0 or not initial:
            return 0
        vec = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n <= 1e-9:
            return 0
        vec = vec / n
        winners = sorted(((a, cid) for cid, a in initial.items()
                          if a >= cfg.neigh_winner_floor), reverse=True)[:cfg.neigh_max_triggers]
        moved = 0
        done: Set[int] = set()
        members = set(initial)
        for a, w in winners:
            for nid, sim in self.neighbours_of(w, exclude=members):
                if nid in done:
                    continue
                c = self.net.concepts.get(nid)
                if c is None or c.frozen or c.origin in PROTECTED_ORIGINS or c.plasticity <= 0:
                    continue
                if sim >= cfg.neigh_pull_ceiling:
                    continue
                eta = cfg.neigh_pull * a * (sim - cfg.neigh_floor) / max(1e-6, 1.0 - cfg.neigh_floor)
                p = c.prototype + eta * (vec - c.prototype)
                norm = float(np.linalg.norm(p))
                if norm > 1e-9:
                    c.prototype = (p / norm).astype(np.float32)
                    done.add(nid)
                    moved += 1
        if moved:
            self.net._dirty = True
            self.stats["prototype_pulls"] += moved
        return moved

    # ==================================================================
    def status(self) -> Dict[str, object]:
        rates = [self.current_rate(cid) for cid in self.net.concepts]
        return {
            **self.stats,
            "mean_rate": round(float(np.mean(rates)), 4) if rates else 0.0,
            "max_rate": round(float(np.max(rates)), 4) if rates else 0.0,
        }
