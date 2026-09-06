"""
src/brain/predictive.py

Predictive Model  (the "JEPA / Diffusion" box).

Two jobs, both in latent space, never in token space:

  1. STATE PREDICTION -- given the current fused percept latent and the action
     about to be taken, predict the *next* percept latent. Compare against
     what actually arrives next turn. The residual is `surprise`.

  2. OUTCOME PREDICTION -- given a candidate action, estimate P(it succeeds).
     Maintained as per-tool Beta posteriors, conditioned coarsely on context.

Surprise is the load-bearing signal of the whole architecture. It is what
turns a fast reflex into a slow deliberation:

    low surprise  -> the world is behaving as modeled -> trust System 1
    high surprise -> the model is wrong about something -> escalate to System 2

This is a JEPA in the structural sense -- it predicts in representation space
rather than reconstructing the input, which is the entire point of the
architecture -- implemented as an online-trained linear forward model rather
than a deep encoder-predictor pair. That choice is deliberate: it trains from
the first turn with no pretraining corpus, costs microseconds, and is
inspectable. `LatentForwardModel` is a clean seam -- swap in a torch MLP or a
real JEPA checkpoint behind the same `predict()` / `learn()` interface and
nothing else in the brain changes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .signals import ActionProposal, Prediction

logger = logging.getLogger("ash.brain.predictive")

STATE_DIR = os.environ.get("ASH_BRAIN_STATE", os.path.join(os.getcwd(), "state", "brain"))


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class LatentForwardModel:
    """s_{t+1} ~= W_s @ s_t + W_a @ a_t + b, trained online by SGD.

    `a_t` is a learned-by-accumulation embedding per intent (a lookup table),
    so the model learns "asking for the time moves the conversation *here*"
    separately from "asking to delete a file moves it *there*".
    """

    def __init__(self, dim: int = 384, action_dim: int = 32, lr: float = 0.02):
        self.dim = dim
        self.action_dim = action_dim
        self.lr = lr
        rng = np.random.default_rng(7)
        # Initialize near-identity: the best zero-knowledge prior for
        # "what happens next" is "roughly what is happening now".
        self.W_s = np.eye(dim, dtype=np.float32) + 0.01 * rng.standard_normal((dim, dim)).astype(np.float32)
        self.W_a = 0.01 * rng.standard_normal((action_dim, dim)).astype(np.float32)
        self.b = np.zeros(dim, dtype=np.float32)
        self.action_table: Dict[str, np.ndarray] = {}
        self.steps = 0

    def action_vec(self, intent: Optional[str]) -> np.ndarray:
        key = intent or "__none__"
        if key not in self.action_table:
            rng = np.random.default_rng(abs(hash(key)) % (2**32))
            self.action_table[key] = _l2(
                rng.standard_normal(self.action_dim).astype(np.float32)
            )
        return self.action_table[key]

    def accepts(self, s: Optional[np.ndarray]) -> bool:
        """Encoders can change under us (a different ASH_EMBED_MODEL, or the
        hash fallback kicking in). Rather than crash on a dimension mismatch,
        the model reports that it can't use this latent and the brain runs
        the cycle without a prediction."""
        return s is not None and getattr(s, "shape", (None,))[0] == self.dim

    def predict(self, s: np.ndarray, intent: Optional[str]) -> Optional[np.ndarray]:
        if not self.accepts(s):
            return None
        a = self.action_vec(intent)
        return _l2(s @ self.W_s + a @ self.W_a + self.b)

    def learn(self, s: np.ndarray, intent: Optional[str], s_next: np.ndarray) -> float:
        """One SGD step. Returns the pre-update cosine error (0..2)."""
        if not (self.accepts(s) and self.accepts(s_next)):
            return 0.0
        a = self.action_vec(intent)
        pred = s @ self.W_s + a @ self.W_a + self.b
        pred_n = _l2(pred)
        err_vec = pred_n - s_next
        cos_err = float(1.0 - np.dot(pred_n, s_next))

        g = err_vec * self.lr
        self.W_s -= np.outer(s, g)
        self.W_a -= np.outer(a, g)
        self.b -= g
        self.steps += 1
        return cos_err

    def state_dict(self) -> Dict[str, Any]:
        return {
            "W_s": self.W_s, "W_a": self.W_a, "b": self.b,
            "actions": np.array(list(self.action_table.keys()), dtype=object),
            "action_vecs": (np.stack(list(self.action_table.values()))
                            if self.action_table else np.zeros((0, self.action_dim), dtype=np.float32)),
            "steps": self.steps,
        }

    def load_state(self, d: Dict[str, Any]):
        self.W_s = d["W_s"].astype(np.float32)
        self.W_a = d["W_a"].astype(np.float32)
        self.b = d["b"].astype(np.float32)
        keys = list(d["actions"])
        vecs = d["action_vecs"]
        self.action_table = {str(k): vecs[i].astype(np.float32) for i, k in enumerate(keys)}
        self.steps = int(d.get("steps", 0))


@dataclass
class _BetaCounts:
    alpha: float = 1.0
    beta: float = 1.0

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def n(self) -> float:
        return self.alpha + self.beta - 2.0

    def observe(self, success: bool, weight: float = 1.0):
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        # Bounded memory: let old evidence decay so a tool that got fixed
        # is not condemned forever by its history.
        total = self.alpha + self.beta
        if total > 200.0:
            scale = 200.0 / total
            self.alpha = 1.0 + (self.alpha - 1.0) * scale
            self.beta = 1.0 + (self.beta - 1.0) * scale


class PredictiveModel:
    """Forward model + outcome estimator + surprise normalizer."""

    def __init__(self, dim: int = 384, persist: bool = True, replay_size: int = 512):
        self.dim = dim
        self.forward = LatentForwardModel(dim=dim)
        self.outcomes: Dict[str, _BetaCounts] = {}
        self.persist = persist
        self.path = os.path.join(STATE_DIR, "predictive.npz")
        self.outcome_path = os.path.join(STATE_DIR, "outcomes.json")

        # Running stats for normalizing raw cosine error into 0..1 surprise.
        self._err_mean = 0.35
        self._err_var = 0.05
        self._n_err = 0

        # Rolling Brier-style error on outcome predictions. Unlike state
        # error this is always diagnostic: if the model said a tool would
        # work and it didn't, something really is wrong.
        self._outcome_err = 0.0

        # Replay buffer for the sleep/consolidation phase.
        self.replay: Deque[Tuple[np.ndarray, Optional[str], np.ndarray]] = deque(maxlen=replay_size)

        self._pending: Optional[Tuple[np.ndarray, Optional[str]]] = None
        self._load()

    # ------------------------------------------------------------------
    def _normalize_surprise(self, raw: float) -> float:
        """z-score the cosine error against its own running distribution,
        then squash. Absolute error magnitude is meaningless; error
        *relative to how well this model usually does* is the real signal."""
        if self._n_err < 5:
            return float(np.clip(raw / 1.2, 0.0, 1.0))
        sd = max(0.05, float(np.sqrt(self._err_var)))
        z = (raw - self._err_mean) / sd
        return float(np.clip(0.5 + 0.25 * z, 0.0, 1.0))

    def _track_err(self, raw: float):
        self._n_err += 1
        d = raw - self._err_mean
        self._err_mean += d / min(self._n_err, 200)
        self._err_var = 0.98 * self._err_var + 0.02 * (d * d)

    # ------------------------------------------------------------------
    # Job 1: state prediction / surprise
    # ------------------------------------------------------------------
    def observe(self, latent: Optional[np.ndarray], candidate_intents: Optional[List[str]] = None) -> Prediction:
        """Called at the START of a cycle, before acting.

        Scores how surprising the *arriving* percept was relative to the
        prediction made last cycle, then emits a fresh prediction plus
        per-candidate outcome estimates.
        """
        surprise, raw = 0.0, 0.0

        if self.forward.accepts(latent) and self._pending is not None:
            prev_s, prev_intent = self._pending
            pred_prev = self.forward.predict(prev_s, prev_intent)
            if pred_prev is not None and prev_s.shape == latent.shape:
                raw = float(1.0 - np.dot(pred_prev, latent))
                surprise = self._normalize_surprise(raw)
                self._track_err(raw)
                # Learn from the transition we just closed out.
                self.forward.learn(prev_s, prev_intent, latent)
                self.replay.append((prev_s.copy(), prev_intent, latent.copy()))

        outcomes = {}
        for intent in (candidate_intents or []):
            outcomes[intent] = self.expected_success(intent)

        conf = float(np.clip(1.0 - self._err_mean, 0.0, 1.0)) * float(
            np.clip(self.forward.steps / 25.0, 0.0, 1.0)
        )

        pred_next = self.forward.predict(latent, None)
        return Prediction(
            predicted_latent=pred_next,
            surprise=surprise,
            outcome_surprise=float(np.clip(self._outcome_err, 0.0, 1.0)),
            raw_error=raw,
            confidence=conf,
            action_outcomes=outcomes,
        )

    def commit(self, latent: Optional[np.ndarray], intent: Optional[str]):
        """Called at the END of a cycle: remember (state, action) so the next
        `observe()` can score its own prediction."""
        if self.forward.accepts(latent):
            self._pending = (latent.copy(), intent)
        else:
            self._pending = None

    # ------------------------------------------------------------------
    # Job 2: outcome prediction
    # ------------------------------------------------------------------
    def expected_success(self, intent: Optional[str]) -> float:
        if not intent:
            return 0.5
        c = self.outcomes.get(intent)
        if c is None:
            return 0.5
        # Shrink toward 0.5 when evidence is thin -- an untested tool is not
        # "100% reliable" after one lucky call.
        n = c.n()
        w = n / (n + 4.0)
        return float(0.5 + w * (c.mean() - 0.5))

    def record_outcome(self, intent: Optional[str], success: bool):
        if not intent:
            return
        # Score the estimate we made BEFORE folding in the new evidence.
        predicted = self.expected_success(intent)
        self._outcome_err = 0.8 * self._outcome_err + 0.2 * abs(predicted - (1.0 if success else 0.0))
        self.outcomes.setdefault(intent, _BetaCounts()).observe(bool(success))

    # ------------------------------------------------------------------
    # Rollout: used only by the slow path
    # ------------------------------------------------------------------
    def rollout(self, latent: Optional[np.ndarray], proposal: ActionProposal, horizon: int = 3) -> Dict[str, float]:
        """Imagine taking this action, `horizon` steps ahead in latent space.

        Returns a drift score (how far this action moves the conversation from
        where it is now) and a stability score (whether the trajectory settles
        or runs away). A proposal whose imagined future diverges is one the
        Executive should be nervous about.
        """
        if not self.forward.accepts(latent):
            return {"drift": 0.0, "stability": 1.0,
                    "expected_success": self.expected_success(proposal.intent)}

        s = latent.copy()
        drifts = []
        for step in range(horizon):
            nxt = self.forward.predict(s, proposal.intent if step == 0 else None)
            if nxt is None:
                break
            drifts.append(float(1.0 - np.dot(_l2(s), nxt)))
            s = nxt

        if not drifts:
            return {"drift": 0.0, "stability": 1.0,
                    "expected_success": self.expected_success(proposal.intent)}

        drift = float(np.clip(np.mean(drifts), 0.0, 2.0) / 2.0)
        # Stable = successive steps stop changing much.
        stability = float(np.clip(1.0 - abs(drifts[-1] - drifts[0]), 0.0, 1.0)) if len(drifts) > 1 else 1.0
        return {
            "drift": drift,
            "stability": stability,
            "expected_success": self.expected_success(proposal.intent),
        }

    # ------------------------------------------------------------------
    # Consolidation (sleep): replay the buffer to sharpen the forward model
    # ------------------------------------------------------------------
    def consolidate(self, epochs: int = 6) -> Dict[str, Any]:
        if not self.replay:
            return {"trained": 0, "epochs": 0}
        rng = np.random.default_rng(int(time.time()) % (2**32))
        buf = list(self.replay)
        before = self._err_mean
        n = 0
        for _ in range(epochs):
            rng.shuffle(buf)
            for s, a, s_next in buf:
                self.forward.learn(s, a, s_next)
                n += 1
        self.save()
        logger.info("Predictive consolidation: %d replay steps over %d epochs", n, epochs)
        return {"trained": n, "epochs": epochs, "err_mean_before": before}

    # ------------------------------------------------------------------
    def save(self):
        if not self.persist:
            return
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            np.savez_compressed(self.path, **self.forward.state_dict())
            with open(self.outcome_path, "w", encoding="utf-8") as fh:
                json.dump({k: [v.alpha, v.beta] for k, v in self.outcomes.items()}, fh, indent=2)
        except Exception:
            logger.exception("Failed to persist predictive model")

    def _load(self):
        try:
            if os.path.exists(self.path):
                d = np.load(self.path, allow_pickle=True)
                if d["W_s"].shape[0] == self.dim:
                    self.forward.load_state({k: d[k] for k in d.files})
                    logger.info("Predictive model restored (%d steps)", self.forward.steps)
            if os.path.exists(self.outcome_path):
                with open(self.outcome_path, "r", encoding="utf-8") as fh:
                    for k, (a, b) in json.load(fh).items():
                        self.outcomes[k] = _BetaCounts(alpha=float(a), beta=float(b))
        except Exception:
            logger.exception("Failed to restore predictive model -- starting fresh")
