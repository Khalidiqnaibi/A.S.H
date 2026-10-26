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
import zlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .signals import ActionProposal, Prediction

logger = logging.getLogger("ash.brain.predictive")

STATE_DIR = os.environ.get("ASH_BRAIN_STATE", os.path.join(os.getcwd(), "state", "brain"))

WARM_STEPS = 25              # transitions before the forward model's rollouts say anything
RELIABLE_CONFIDENCE = 0.2    # model_confidence() needed to let planning steer decisions
PERSISTENCE_MARGIN = 0.85    # to be reliable, error must be under this fraction of "nothing changes" error


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
            # crc32, not hash(): str hashing is salted per process, which would
            # hand the same intent a different vector after every restart.
            rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
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
        self.goal_path = os.path.join(STATE_DIR, "goals.npz")

        # Outcome prototypes: per intent, a running mean of the latent that
        # FOLLOWED a successful use -- "what the world looks like once this
        # works". Learned from experience, no language model involved. It is
        # the cheapest honest answer to "what state am I trying to reach?".
        self.goal_protos: Dict[str, np.ndarray] = {}
        self.goal_counts: Dict[str, int] = {}

        # Running stats for normalizing raw cosine error into 0..1 surprise.
        self._err_mean = 0.35
        self._err_var = 0.05
        self._n_err = 0
        # Error of the laziest possible predictor ("the next state is the
        # current one"). A forward model that cannot beat it knows nothing.
        self._base_mean = 0.35

        # Rolling Brier-style error on outcome predictions. Unlike state
        # error this is always diagnostic: if the model said a tool would
        # work and it didn't, something really is wrong.
        self._outcome_err = 0.0

        # Replay buffer for the sleep/consolidation phase.
        self.replay: Deque[Tuple[np.ndarray, Optional[str], np.ndarray]] = deque(maxlen=replay_size)

        # (state, action, outcome, stream): the guess is about the next event
        # *of that stream*; see observe().
        self._pending: Optional[Tuple[np.ndarray, Optional[str], Optional[bool], str]] = None
        # Set by record_outcome(), consumed by the next commit() so the
        # transition it closes out knows whether the action it followed worked.
        self._last_outcome: Optional[Tuple[Optional[str], bool]] = None
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

    def _track_err(self, raw: float, baseline: Optional[float] = None):
        self._n_err += 1
        d = raw - self._err_mean
        self._err_mean += d / min(self._n_err, 200)
        self._err_var = 0.98 * self._err_var + 0.02 * (d * d)
        if baseline is not None:
            self._base_mean += (baseline - self._base_mean) / min(self._n_err, 200)

    # ------------------------------------------------------------------
    # Job 1: state prediction / surprise
    # ------------------------------------------------------------------
    def observe(self, latent: Optional[np.ndarray], candidate_intents: Optional[List[str]] = None,
                stream: str = "request") -> Prediction:
        """Called at the START of a cycle, before acting.

        Scores how surprising the *arriving* percept was relative to the
        prediction made last cycle, then emits a fresh prediction plus
        per-candidate outcome estimates.

        A prediction is only about the next event of the stream it was made
        in. The one chain used to run across everything, so the guess made
        after a window switch ("the next window switch") was scored against
        the user's next question, which is of course different: every request
        after sensor activity read as a surprise, was trained on as if the
        model had been wrong, and dragged the error statistics the ambient
        stream is judged by. A transition that crosses streams is therefore
        neither scored nor learned from.
        """
        surprise, raw = 0.0, 0.0

        if self._pending is not None and self._pending[3] != stream:
            self._pending = None
        if self.forward.accepts(latent) and self._pending is not None:
            prev_s, prev_intent, prev_ok, _ = self._pending
            pred_prev = self.forward.predict(prev_s, prev_intent)
            if pred_prev is not None and prev_s.shape == latent.shape:
                raw = float(1.0 - np.dot(pred_prev, latent))
                surprise = self._normalize_surprise(raw)
                self._track_err(raw, float(1.0 - np.dot(_l2(prev_s), _l2(latent))))
                # Learn from the transition we just closed out.
                self.forward.learn(prev_s, prev_intent, latent)
                self.replay.append((prev_s.copy(), prev_intent, latent.copy()))
                if prev_ok is True and prev_intent:
                    self._update_proto(prev_intent, latent)

        outcomes = {}
        for intent in (candidate_intents or []):
            outcomes[intent] = self.expected_success(intent)

        conf = self.model_confidence()

        pred_next = self.forward.predict(latent, None)
        return Prediction(
            predicted_latent=pred_next,
            surprise=surprise,
            outcome_surprise=float(np.clip(self._outcome_err, 0.0, 1.0)),
            raw_error=raw,
            confidence=conf,
            action_outcomes=outcomes,
        )

    def model_confidence(self) -> float:
        """0..1: how far the forward model's own predictions can be trusted."""
        return float(np.clip(1.0 - self._err_mean, 0.0, 1.0)) * float(
            np.clip(self.forward.steps / 25.0, 0.0, 1.0)
        )

    @property
    def reliable(self) -> bool:
        """Whether imagined futures mean anything yet. A cold linear model
        predicts nearly the same next state whatever the action, so the small
        differences between candidates are noise -- and goal gain multiplies
        them by six. Found running real queries: with planning on, "hello"
        went to read_file."""
        return (self.model_confidence() >= RELIABLE_CONFIDENCE
                and self._err_mean < PERSISTENCE_MARGIN * self._base_mean)

    def commit(self, latent: Optional[np.ndarray], intent: Optional[str], stream: str = "request"):
        """Called at the END of a cycle: remember (state, action) so the next
        `observe()` of the same stream can score its own prediction."""
        ok: Optional[bool] = None
        if self._last_outcome is not None and self._last_outcome[0] == intent:
            ok = self._last_outcome[1]
        self._last_outcome = None
        if self.forward.accepts(latent):
            self._pending = (latent.copy(), intent, ok, stream)
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
        self._last_outcome = (intent, bool(success))

    # ------------------------------------------------------------------
    # Goals: outcome prototypes
    # ------------------------------------------------------------------
    def _update_proto(self, intent: str, latent: np.ndarray, rate: float = 0.2):
        if not self.forward.accepts(latent):
            return
        v = _l2(latent.astype(np.float32))
        n = self.goal_counts.get(intent, 0)
        old = self.goal_protos.get(intent)
        if old is None or old.shape != v.shape:
            self.goal_protos[intent] = v
            self.goal_counts[intent] = 1
            return
        # Running mean while evidence is thin, then a fixed-rate EMA so the
        # prototype can follow a world that drifts instead of freezing.
        r = max(rate, 1.0 / (n + 1))
        self.goal_protos[intent] = _l2((1.0 - r) * old + r * v)
        self.goal_counts[intent] = n + 1

    def seed_goal(self, intent: str, vec: Optional[np.ndarray]):
        """Cold-start a prototype (e.g. from an embedded tool description).
        Counts as one observation, so real experience quickly overrides it."""
        if vec is None or not self.forward.accepts(vec):
            return
        if intent not in self.goal_protos:
            self._update_proto(intent, vec)

    def goal_for(self, intent: Optional[str], min_obs: int = 2) -> Optional[np.ndarray]:
        """The state that tends to follow `intent` working, or None if there
        is not yet enough evidence to call it a goal."""
        if not intent:
            return None
        if self.goal_counts.get(intent, 0) < min_obs:
            return None
        return self.goal_protos.get(intent)

    def trajectory(self, latent: Optional[np.ndarray],
                   path: List[Optional[str]]) -> List[np.ndarray]:
        """The imagined state after each step of `path` (one intent per step).
        Empty when the latent is unusable."""
        if not self.forward.accepts(latent):
            return []
        s = _l2(latent.astype(np.float32))
        out: List[np.ndarray] = []
        for intent in path:
            s = self.forward.predict(s, intent)
            if s is None:
                break
            out.append(s)
        return out

    # ------------------------------------------------------------------
    # Rollout: used only by the slow path
    # ------------------------------------------------------------------
    def rollout(self, latent: Optional[np.ndarray], proposal: ActionProposal, horizon: int = 3,
                goal: Optional[np.ndarray] = None,
                path: Optional[List[Optional[str]]] = None) -> Dict[str, Any]:
        """Imagine taking this action, `horizon` steps ahead in latent space.

        Returns a drift score (how far this action moves the conversation from
        where it is now) and a stability score (whether the trajectory settles
        or runs away). A proposal whose imagined future diverges is one the
        Executive should be nervous about.

        With a `goal` latent the rollout also answers the better question --
        did the imagined future get CLOSER to where we want to be:

            goal_gain   cos(s_end, goal) - cos(s_0, goal), in [-1, 1]
            has_goal    whether goal_gain means anything (False => 0.0)

        `path` supplies one intent per step (a multi-step plan); by default
        only step 0 carries the proposal's intent, as before.

        `success_fit` is separate: how much the imagined next state resembles
        what followed this intent's past successes (None when unknown).
        """
        fallback = {"drift": 0.0, "stability": 1.0,
                    "expected_success": self.expected_success(proposal.intent),
                    "goal_gain": 0.0, "has_goal": False, "success_fit": None}
        if not self.forward.accepts(latent):
            return fallback

        steps = list(path) if path else [proposal.intent] + [None] * (horizon - 1)
        s = _l2(latent.astype(np.float32))
        s0 = s
        drifts: List[float] = []
        first_next: Optional[np.ndarray] = None
        for intent in steps:
            nxt = self.forward.predict(s, intent)
            if nxt is None:
                break
            if first_next is None:
                first_next = nxt
            drifts.append(float(1.0 - np.dot(_l2(s), nxt)))
            s = nxt

        if not drifts:
            return fallback

        drift = float(np.clip(np.mean(drifts), 0.0, 2.0) / 2.0)
        # Stable = successive steps stop changing much.
        stability = float(np.clip(1.0 - abs(drifts[-1] - drifts[0]), 0.0, 1.0)) if len(drifts) > 1 else 1.0

        out = dict(fallback)
        out.update({"drift": drift, "stability": stability})

        # Until the model has seen enough transitions the differences between
        # candidates' imagined futures are noise: no goal gain, no success fit.
        # (A model trained directly, with no observed transitions to judge it
        # by, is trusted on its step count alone.)
        warm = self.forward.steps >= WARM_STEPS and (self._n_err < WARM_STEPS or self.reliable)
        if warm and goal is not None and self.forward.accepts(goal):
            g = _l2(goal.astype(np.float32))
            out["goal_gain"] = float(np.clip(np.dot(s, g) - np.dot(s0, g), -1.0, 1.0))
            out["has_goal"] = True

        own = self.goal_for(proposal.intent) if warm else None
        if own is not None and first_next is not None and own.shape == first_next.shape:
            out["success_fit"] = float(np.dot(first_next, own))
        return out

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
            if self.goal_protos:
                keys = list(self.goal_protos.keys())
                np.savez_compressed(
                    self.goal_path,
                    intents=np.array(keys, dtype=object),
                    protos=np.stack([self.goal_protos[k] for k in keys]),
                    counts=np.array([self.goal_counts.get(k, 0) for k in keys], dtype=np.int64))
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
            if os.path.exists(self.goal_path):
                g = np.load(self.goal_path, allow_pickle=True)
                if g["protos"].ndim == 2 and g["protos"].shape[1] == self.dim:
                    for i, k in enumerate(g["intents"]):
                        self.goal_protos[str(k)] = g["protos"][i].astype(np.float32)
                        self.goal_counts[str(k)] = int(g["counts"][i])
        except Exception:
            logger.exception("Failed to restore predictive model -- starting fresh")
