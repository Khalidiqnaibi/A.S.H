"""
src/concepts/sleep.py

The sleep cycle: when and how the graph is actually saved and updated.

Sleep is not one operation. It has stages that do different jobs, and running
them in the wrong order or skipping one produces a specific failure. ASH now
runs a cycle of them:

    NREM  -- REPLAY. The hippocampus re-presents episodes to cortex,
             interleaved old with new, and cortex learns slowly from that
             stream. This is systems consolidation: the only way a one-shot
             episode becomes general knowledge without overwriting anything.

          -- DOWNSCALE. Every synapse is multiplicatively weakened by a small
             factor proportional to the day's total potentiation. This is the
             synaptic homeostasis hypothesis (Tononi & Cirelli): waking is net
             potentiating, so without a global renormalization the network
             saturates and signal-to-noise collapses. Multiplicative
             downscaling preserves relative strengths while restoring dynamic
             range -- weak edges fall below threshold and are pruned, strong
             ones survive. It is why you wake up with the gist and not the
             noise.

    REM   -- RECOMBINATION. Replay with the constraints loosened: samples are
             drawn more widely, activation spreads further, and associations
             formed between things that never actually co-occurred but sit in
             overlapping structure. This is where a schema gets abstracted out
             of instances. It is also the stage most likely to introduce
             nonsense, which is why it runs last and inside the rollback guard.

Ordering matters
----------------
Replay must come before downscaling. Downscale first and you weaken the very
traces you were about to consolidate. Prune before replay and you delete edges
that were about to be reinforced. The sequence here is deliberate and the
tests check it.

Why cycles rather than one pass
-------------------------------
Real sleep alternates NREM and REM several times a night, with NREM dominant
early and REM later. Consolidation benefits from that alternation: replay
builds the structure, recombination generalizes over it, and repeating the
pair lets the generalization feed back into the next round of replay.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .model import Activation, SynType

logger = logging.getLogger("ash.concepts.sleep")


@dataclass
class SleepConfig:
    cycles: int = 3                    # NREM/REM alternations
    nrem_batches: int = 6              # replay batches per NREM stage
    batch_size: int = 24
    new_fraction: float = 0.4          # new vs old in each interleaved batch
    downscale_base: float = 0.03       # minimum global weakening per night
    downscale_max: float = 0.22
    rem_batches: int = 2
    rem_spread_gain: float = 1.6       # activation travels further in REM
    rem_new_fraction: float = 0.25
    trace_decay_hours: float = 8.0


class SleepCycle:
    """Orchestrates hippocampal replay into the cortical graph."""

    def __init__(self, net, plasticity, hippocampus, neuromod=None,
                 cfg: Optional[SleepConfig] = None):
        self.net = net
        self.plasticity = plasticity
        self.hpc = hippocampus
        self.neuromod = neuromod
        self.cfg = cfg or SleepConfig()
        self.nights = 0
        self.last_report: Dict[str, Any] = {}

    # ==================================================================
    def run(self, evaluator: Optional[Callable[[], float]] = None,
            hours_since_last: float = 8.0) -> Dict[str, Any]:
        """One night. Returns a report of what changed."""
        t0 = time.perf_counter()
        self.nights += 1
        report: Dict[str, Any] = {"night": self.nights, "stages": []}

        # Drop acetylcholine. This is the switch from encoding mode to
        # consolidation mode -- with ACh high, hippocampal output cannot drive
        # cortex and replay accomplishes nothing.
        if self.neuromod is not None:
            for _ in range(6):
                self.neuromod.update(novelty=0.0, surprise=0.0, asleep=True)
            report["ach"] = round(self.neuromod.state.ach, 3)

        before = self._snapshot_stats()
        pre_score = None
        snapshot = None
        if evaluator is not None:
            try:
                pre_score = float(evaluator())
                snapshot = self.plasticity._snapshot()
            except Exception:
                logger.exception("Pre-sleep evaluation failed; running unguarded")

        for c in range(self.cfg.cycles):
            nrem = self._nrem()
            report["stages"].append({"cycle": c, "stage": "NREM", **nrem})
            rem = self._rem()
            report["stages"].append({"cycle": c, "stage": "REM", **rem})

        # Downscaling happens AFTER all replay, once per night.
        report["downscale"] = self._downscale()
        report["pruned"] = self.plasticity._prune()
        report["scaled"] = self.plasticity._homeostatic_scaling()
        report["merged"] = self.plasticity._merge_converged()
        report["faded_traces"] = self.hpc.decay(hours=hours_since_last)

        if evaluator is not None and snapshot is not None:
            try:
                post = float(evaluator())
                report["score_before"] = round(pre_score, 4)
                report["score_after"] = round(post, 4)
                if post < pre_score - 0.05:
                    self.plasticity._restore(snapshot)
                    report["ROLLED_BACK"] = True
                    logger.error("Sleep regressed routing %.3f -> %.3f; reverted",
                                 pre_score, post)
            except Exception:
                logger.exception("Post-sleep evaluation failed; keeping changes")

        report["delta"] = self._delta(before)
        report["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.last_report = report
        logger.info("Sleep night %d: %s", self.nights,
                    {k: v for k, v in report.items() if k != "stages"})
        return report

    # ==================================================================
    # NREM: interleaved replay
    # ==================================================================
    def _nrem(self) -> Dict[str, Any]:
        replayed = 0
        associated = 0
        self.plasticity.replay_mode = True

        for _ in range(self.cfg.nrem_batches):
            batch = self.hpc.replay_batch(
                size=self.cfg.batch_size, new_fraction=self.cfg.new_fraction)
            if not batch:
                break

            # Replay in experienced order. Hippocampal replay is sequential
            # and time-compressed, not shuffled, which is what lets sleep
            # consolidate ORDER as well as association.
            batch.sort(key=lambda t: t.created)

            for trace in batch:
                # Re-present the episode to cortex. Both growth and prototype
                # adaptation are off. Growth off because sleep consolidates
                # what exists rather than inventing categories from replayed
                # material. Prototype adaptation off because replaying one
                # episode repeatedly drags its concepts toward each other
                # until the merge step collapses them -- which deletes the
                # edge replay just spent the whole stage building.
                # Consolidation strengthens CONNECTIONS; category formation is
                # a waking job.
                # Clear the STDP history at each episode boundary. Two
                # consecutively replayed episodes are separate events, not a
                # sequence, and letting spike-timing run across the boundary
                # teaches ordering from the replay schedule -- which marked
                # every edge in the graph causal the first time this ran.
                # Ordering is only learned WITHIN an episode; with today's
                # single-snapshot episodes that means STDP contributes
                # nothing here, which is correct, and it starts contributing
                # automatically once episodes carry multiple frames.
                self.plasticity._recent.clear()

                act = self.net.perceive(trace.dense, allow_growth=False,
                                        learn_prototypes=False)
                if len(act.spread) < 2:
                    continue
                self.plasticity.replay_gate = max(0.3, trace.strength *
                                                  (1.0 + abs(trace.reward)))
                self.plasticity.potentiate(act)
                # The episode's own reward is replayed with it, so
                # consolidation is still reward-weighted rather than becoming
                # pure correlation the moment we leave the waking loop.
                if abs(trace.reward) > 0.05:
                    self.plasticity.reward(trace.reward)
                replayed += 1
                associated += len(act.spread)
                self.hpc.consolidated += 1

        self.plasticity.replay_mode = False
        committed = self.plasticity._commit_traces(
            schema_lookup=getattr(self.plasticity, "_schema_lookup", None))
        self.plasticity.traces.clear()
        self.plasticity.credited.clear()
        return {"replayed": replayed, "committed": committed,
                "mean_active": round(associated / max(1, replayed), 1)}

    # ==================================================================
    # Synaptic homeostasis
    # ==================================================================
    def _downscale(self) -> Dict[str, Any]:
        """Global multiplicative weakening proportional to the day's gain.

        The size of the downscale tracks how much potentiation happened while
        awake, so a busy day is renormalized harder than a quiet one. Without
        this the graph ratchets upward -- every day adds weight and nothing
        removes it -- until every edge is near its ceiling and the network can
        no longer discriminate anything.
        """
        total = sum(abs(s.weight()) for s in self.net.synapses.values())
        n = max(1, len(self.net.synapses))
        mean_w = total / n

        # Scale the downscale to how saturated the graph has become.
        pressure = float(min(1.0, mean_w / 0.8))
        factor = self.cfg.downscale_base + (
            self.cfg.downscale_max - self.cfg.downscale_base) * pressure

        touched = 0
        for syn in self.net.synapses.values():
            if syn.frozen:
                continue
            syn.w_slow *= (1.0 - factor)
            syn.w_fast *= 0.2      # fast weights do not survive the night
            touched += 1

        return {"factor": round(factor, 4), "mean_weight_before": round(mean_w, 4),
                "synapses": touched}

    # ==================================================================
    # REM: recombination
    # ==================================================================
    def _rem(self) -> Dict[str, Any]:
        """Replay with the constraints loosened.

        Spreading gain is raised and sampling is biased toward older
        material, so activation reaches concepts that never co-occurred with
        the replayed episode but sit in overlapping structure. Associations
        formed here are between things ASH inferred rather than witnessed --
        which is where generalization comes from, and also where nonsense
        comes from, hence the reduced learning rate and the rollback guard
        wrapped around the whole night.
        """
        original_gain = self.net.spread_gain
        original_steps = self.net.spread_steps
        original_eta = self.plasticity.cfg.eta_fast

        self.net.spread_gain = original_gain * self.cfg.rem_spread_gain
        self.net.spread_steps = original_steps + 1
        self.plasticity.cfg.eta_fast = original_eta * 0.5

        novel_links = 0
        recombined = 0
        self.plasticity.replay_mode = True
        self.plasticity.replay_gate = 0.6
        # No STDP in REM. Recombination deliberately samples out of order and
        # across contexts; any ordering it produced would be an artefact.
        self.plasticity.stdp_enabled = False
        try:
            for _ in range(self.cfg.rem_batches):
                batch = self.hpc.replay_batch(
                    size=self.cfg.batch_size // 2,
                    new_fraction=self.cfg.rem_new_fraction)
                for trace in batch:
                    act = self.net.perceive(trace.dense, allow_growth=False,
                                            learn_prototypes=False)
                    inferred = act.inferred(6)
                    if len(inferred) >= 2:
                        novel_links += self.plasticity.potentiate(act)
                    recombined += 1
            self.plasticity._commit_traces()
            self.plasticity.traces.clear()
            self.plasticity.credited.clear()
        finally:
            self.plasticity.stdp_enabled = True
            self.plasticity.replay_mode = False
            self.net.spread_gain = original_gain
            self.net.spread_steps = original_steps
            self.plasticity.cfg.eta_fast = original_eta

        return {"recombined": recombined, "novel_links": novel_links}

    # ==================================================================
    def _snapshot_stats(self) -> Dict[str, float]:
        weights = [s.weight() for s in self.net.synapses.values()]
        return {
            "synapses": len(self.net.synapses),
            "concepts": len(self.net.concepts),
            "mean_w": float(np.mean(weights)) if weights else 0.0,
            "causal": sum(1 for s in self.net.synapses.values()
                          if s.kind == SynType.CAUSAL),
        }

    def _delta(self, before: Dict[str, float]) -> Dict[str, Any]:
        after = self._snapshot_stats()
        return {k: (round(after[k] - before[k], 4)
                    if isinstance(after[k], float) else after[k] - before[k])
                for k in before}

    def status(self) -> Dict[str, Any]:
        return {"nights": self.nights, "last": self.last_report}
