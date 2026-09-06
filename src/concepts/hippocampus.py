"""
src/concepts/hippocampus.py

The fast learner. One-shot, sparse, temporary.

This is the half of complementary learning systems that ASH was missing, and
it is the direct answer to "adapt without a lot of data, like a baby".

The problem it solves
---------------------
A single network cannot both learn from one example and accumulate robust
general knowledge. Fast learning overwrites (catastrophic forgetting); slow
learning needs thousands of samples. Every continual-learning system that
tries to do both with one learning rate ends up bad at both.

The brain's answer is two systems with a transfer channel:

    HIPPOCAMPUS   high learning rate, sparse, pattern-separated, capacity
                  limited, decays in days. Stores an episode after ONE
                  exposure, verbatim.
    NEOCORTEX     low learning rate, overlapping, statistical, permanent.
                  Cannot learn from one example without damage.
    REPLAY        during sleep, the hippocampus re-presents its episodes to
                  cortex thousands of times, interleaved with old material,
                  so cortex extracts the statistics safely.

So ASH can know something the moment you tell it (hippocampal trace), and
still have that knowledge become part of its general structure only after it
has been rehearsed against everything else it knows.

Three mechanisms make this work
-------------------------------
1. PATTERN SEPARATION (dentate gyrus). Before storage, the input is expanded
   into a much sparser, higher-dimensional code where similar inputs become
   *less* similar. This is why two similar episodes on consecutive days stay
   distinguishable instead of blurring. Implemented as a fixed random
   expansion followed by aggressive k-WTA -- which is, to within a constant,
   what the DG actually does.

2. PATTERN COMPLETION (CA3 recurrent). Storage is autoassociative: a partial
   cue retrieves the whole pattern. This is what makes one-shot recall from a
   fragment work.

3. INTERLEAVED REPLAY. Replaying only new episodes to cortex causes exactly
   the forgetting the split was supposed to prevent. Replay must mix new
   episodes with old ones, which is why `replay_batch` samples from both.

Decay is a feature
------------------
Hippocampal traces fade. An episode that is never replayed, never retrieved,
and never rewarded is gone in days. That is correct: the store is a buffer
for what might turn out to matter, not an archive. The archive is cortex, and
things get there by being rehearsed.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("ash.concepts.hippocampus")


@dataclass
class Trace:
    """One episodic memory. Stored once, verbatim."""

    id: int
    sparse: np.ndarray                 # pattern-separated code (indices)
    values: np.ndarray                 # activation at those indices
    dense: np.ndarray                  # original embedding, for cortical replay
    concepts: Dict[int, float] = field(default_factory=dict)
    text: str = ""
    intent: Optional[str] = None
    reward: float = 0.0
    created: float = field(default_factory=time.time)
    last_recalled: float = 0.0
    recalls: int = 0
    replays: int = 0
    strength: float = 1.0              # decays; boosted by recall and reward

    def age_h(self) -> float:
        return (time.time() - self.created) / 3600.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "text": self.text[:80], "intent": self.intent,
            "reward": round(self.reward, 3), "strength": round(self.strength, 3),
            "recalls": self.recalls, "replays": self.replays,
            "age_h": round(self.age_h(), 2),
        }


class Hippocampus:
    def __init__(self, dim: int = 384, expansion: int = 2048, sparsity: int = 24,
                 capacity: int = 600, decay_tau_h: float = 72.0,
                 completion_threshold: float = 0.22):
        self.dim = dim
        self.expansion = expansion
        self.sparsity = sparsity          # active units in the separated code
        self.capacity = capacity
        self.decay_tau_h = decay_tau_h
        self.completion_threshold = completion_threshold

        # Fixed random expansion. Seeded so the code for a given input is
        # stable across restarts -- a separation matrix that changed would
        # orphan every stored trace.
        rng = np.random.default_rng(0xD6)
        self._proj = rng.standard_normal((dim, expansion)).astype(np.float32)
        self._proj /= math.sqrt(dim)

        self.traces: Dict[int, Trace] = {}
        self._next_id = 0
        self._dense: Optional[np.ndarray] = None    # cache for recall
        self._dense_ids: List[int] = []
        self._dense_dirty = True
        # Inverted index: separated-unit -> trace ids. Makes completion a
        # sparse lookup instead of a scan over the whole store.
        self._index: Dict[int, List[int]] = {}

        self.stored = 0
        self.evicted = 0
        self.consolidated = 0

    # ==================================================================
    # Pattern separation
    # ==================================================================
    def separate(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Expand and sparsify. Similar inputs come out *less* similar.

        The expansion alone would preserve similarity; it is the aggressive
        k-WTA afterwards that does the separating, because two inputs that
        differ slightly select partially different winner sets and the
        overlap falls off much faster than the cosine does.
        """
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            v = v / n
        expanded = v @ self._proj
        k = min(self.sparsity, expanded.shape[0])
        idx = np.argpartition(-expanded, k - 1)[:k]
        idx = idx[np.argsort(-expanded[idx])]
        vals = expanded[idx]
        vals = vals / (float(np.linalg.norm(vals)) + 1e-9)
        return idx.astype(np.int32), vals.astype(np.float32)

    @staticmethod
    def overlap(a_idx: np.ndarray, a_val: np.ndarray,
                b_idx: np.ndarray, b_val: np.ndarray) -> float:
        """Similarity of two sparse codes: weighted index overlap."""
        common = np.intersect1d(a_idx, b_idx, assume_unique=False)
        if common.size == 0:
            return 0.0
        am = {int(i): float(v) for i, v in zip(a_idx, a_val)}
        bm = {int(i): float(v) for i, v in zip(b_idx, b_val)}
        return float(sum(am[int(c)] * bm[int(c)] for c in common))

    # ==================================================================
    # Storage -- one shot
    # ==================================================================
    def store(self, vec: np.ndarray, text: str = "", concepts: Optional[Dict[int, float]] = None,
              intent: Optional[str] = None, reward: float = 0.0,
              gain: float = 1.0) -> Trace:
        """Write one episode. No repetition required, no gradient steps.

        `gain` is the acetylcholine encoding gain: in a familiar situation the
        encoder is turned down and the trace starts weak, so it will fade
        unless something makes it matter. That is why the hundredth day
        somewhere produces almost no memories.
        """
        idx, vals = self.separate(vec)
        t = Trace(
            id=self._next_id, sparse=idx, values=vals,
            dense=np.asarray(vec, dtype=np.float32).copy(),
            concepts=dict(concepts or {}), text=text[:400], intent=intent,
            reward=float(reward),
            # A freshly encoded trace starts at full strength under normal
            # encoding gain. The earlier form halved it, which silently made
            # every recall score half what it should be and broke completion
            # from anything but an exact cue.
            strength=float(min(2.0, gain * (1.0 + 0.5 * abs(reward)))),
        )
        self._next_id += 1
        self.traces[t.id] = t
        for i in idx:
            self._index.setdefault(int(i), []).append(t.id)
        self.stored += 1
        self._dense_dirty = True

        if len(self.traces) > self.capacity:
            self._evict()
        return t

    def _evict(self):
        """Drop the weakest trace. Rewarded and recently-recalled ones
        survive; the rest are supposed to be transient."""
        if not self.traces:
            return
        now = time.time()

        def keep_score(t: Trace) -> float:
            recency = math.exp(-(now - max(t.created, t.last_recalled)) / (3600 * 12))
            return t.strength + 0.6 * abs(t.reward) + 0.4 * recency + 0.15 * t.recalls

        victim = min(self.traces.values(), key=keep_score)
        self._remove(victim.id)
        self.evicted += 1

    def _remove(self, tid: int):
        t = self.traces.pop(tid, None)
        if t is None:
            return
        self._dense_dirty = True
        for i in t.sparse:
            lst = self._index.get(int(i))
            if lst:
                try:
                    lst.remove(tid)
                except ValueError:
                    pass
                if not lst:
                    self._index.pop(int(i), None)

    # ==================================================================
    # Pattern completion -- recall from a fragment
    # ==================================================================
    def _rebuild_dense(self):
        if not self._dense_dirty:
            return
        self._dense_ids = list(self.traces)
        if self._dense_ids:
            self._dense = np.stack([self.traces[i].dense for i in self._dense_ids])
            norms = np.linalg.norm(self._dense, axis=1, keepdims=True)
            self._dense = self._dense / np.maximum(norms, 1e-9)
        else:
            self._dense = None
        self._dense_dirty = False

    def recall(self, vec: np.ndarray, n: int = 5,
               min_overlap: Optional[float] = None) -> List[Tuple[Trace, float]]:
        """Autoassociative retrieval. A partial cue returns whole episodes.

        Two stages, which is the practical approximation of CA3 settling into
        an attractor:

          1. DENSE candidate generation. Sparse index alone cannot complete
             from a badly degraded cue -- heavy noise changes the k-WTA winner
             set entirely and the overlap goes to zero, so nothing is even
             considered. The dense stage is what makes completion work.
          2. SPARSE verification. The separated code then discriminates among
             the candidates, which is what keeps two similar-but-distinct
             episodes from being confused. Dropping this stage would give
             completion at the cost of separation.

        Using only one of the two gives you a store that either cannot recall
        from fragments or cannot tell similar memories apart.
        """
        if not self.traces:
            return []
        self._rebuild_dense()
        idx, vals = self.separate(vec)
        thresh = self.completion_threshold if min_overlap is None else min_overlap

        q = np.asarray(vec, dtype=np.float32)
        q = q / max(1e-9, float(np.linalg.norm(q)))
        dense_scores = self._dense @ q if self._dense is not None else np.zeros(0)

        # Stage 1: widest plausible candidate set, by dense similarity.
        take = min(len(self._dense_ids), max(4 * n, 20))
        order = np.argpartition(-dense_scores, take - 1)[:take] if take else []

        scored: List[Tuple[Trace, float]] = []
        now = time.time()
        for i in np.atleast_1d(order):
            tid = self._dense_ids[int(i)]
            t = self.traces.get(tid)
            if t is None:
                continue
            dense_sim = float(dense_scores[int(i)])
            if dense_sim <= 0.05:
                continue
            # Stage 2: sparse overlap sharpens the ranking. The sum, rather
            # than the product, means a strong dense match with a scrambled
            # sparse code can still be retrieved -- that is completion -- while
            # the sparse term still decides between near-neighbours.
            sparse_sim = self.overlap(idx, vals, t.sparse, t.values)
            o = (0.55 * dense_sim + 0.45 * sparse_sim) * t.strength
            if o >= thresh:
                scored.append((t, o))

        scored.sort(key=lambda kv: -kv[1])
        for t, _ in scored[:n]:
            t.recalls += 1
            t.last_recalled = now
            # Retrieval strengthens the trace. Testing-effect: recalling a
            # memory is a better predictor of its survival than re-studying.
            t.strength = min(2.0, t.strength + 0.12)
        return scored[:n]

    # ==================================================================
    # Decay
    # ==================================================================
    def decay(self, hours: float = 1.0) -> int:
        """Fade unrehearsed traces. Returns the number removed."""
        factor = math.exp(-hours / self.decay_tau_h)
        dead = []
        for t in self.traces.values():
            # Rewarded and frequently-consolidated traces decay slower.
            protection = 1.0 + 0.5 * abs(t.reward) + 0.1 * t.replays
            t.strength *= factor ** (1.0 / protection)
            if t.strength < 0.05:
                dead.append(t.id)
        for tid in dead:
            self._remove(tid)
        return len(dead)

    # ==================================================================
    # Replay -- the transfer channel to cortex
    # ==================================================================
    def replay_batch(self, size: int = 32, new_fraction: float = 0.4,
                     new_window_h: float = 24.0) -> List[Trace]:
        """Sample a replay batch, INTERLEAVED.

        The interleaving is the whole point and the easiest thing to get
        wrong. Replaying only recent episodes teaches cortex the last day and
        overwrites the rest -- precisely the catastrophic forgetting the two
        systems exist to prevent. A batch is therefore part new material and
        part old, which is what lets cortex integrate today without losing
        last month.

        Sampling is weighted by strength and |reward|, so surprising and
        consequential episodes are rehearsed more -- which is what sharp-wave
        ripple replay is observed to do.
        """
        if not self.traces:
            return []
        now = time.time()
        new, old = [], []
        for t in self.traces.values():
            (new if (now - t.created) / 3600.0 <= new_window_h else old).append(t)

        def sample(pool: List[Trace], k: int) -> List[Trace]:
            if not pool or k <= 0:
                return []
            weights = [max(1e-3, t.strength * (1.0 + abs(t.reward))) for t in pool]
            total = sum(weights)
            probs = [w / total for w in weights]
            k = min(k, len(pool))
            picked = np.random.default_rng().choice(
                len(pool), size=k, replace=len(pool) < k, p=probs)
            return [pool[int(i)] for i in np.atleast_1d(picked)]

        n_new = int(size * new_fraction)
        batch = sample(new, n_new) + sample(old, size - n_new)
        # If one pool is empty, top up from the other rather than short the batch.
        if len(batch) < size:
            batch += sample(new + old, size - len(batch))
        for t in batch:
            t.replays += 1
        return batch

    # ==================================================================
    # Schema detection -- why the tenth example is faster than the first
    # ==================================================================
    def schema_strength(self, vec: np.ndarray, radius: float = 0.30) -> float:
        """How much existing structure supports this input.

        Once a schema exists, a consistent new instance consolidates in one
        trial instead of many -- this is a real and well-replicated effect
        (Tse et al. 2007), and it is the mechanism behind "learning gets
        easier once you know the area". Here it is measured as the density of
        related traces, and it is used to fast-track consolidation.
        """
        hits = self.recall(vec, n=12, min_overlap=radius)
        if not hits:
            return 0.0
        support = sum(o for _, o in hits)
        return float(min(1.0, support / 3.0))

    # ==================================================================
    def status(self) -> Dict[str, Any]:
        if not self.traces:
            return {"traces": 0, "stored": self.stored, "evicted": self.evicted}
        strengths = [t.strength for t in self.traces.values()]
        ages = [t.age_h() for t in self.traces.values()]
        return {
            "traces": len(self.traces),
            "capacity": self.capacity,
            "stored": self.stored,
            "evicted": self.evicted,
            "consolidated": self.consolidated,
            "mean_strength": round(float(np.mean(strengths)), 3),
            "oldest_h": round(max(ages), 1),
            "mean_age_h": round(float(np.mean(ages)), 1),
            "rewarded": sum(1 for t in self.traces.values() if abs(t.reward) > 0.2),
        }
