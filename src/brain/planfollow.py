"""
src/brain/planfollow.py

PlanFollower -- holds the active plan and decides, after every step, whether
reality is still behaving the way the plan imagined.

A plan (signals.Imagery) is a path plus the state the brain expected to be in
after each step. Following it is a loop:

    take step  ->  observe what actually happened  ->  compare with what was
    imagined  ->  CONTINUE, REPLAN from where we really are, finish, or give up

This module is the "compare" part and nothing else. It does no sensing, runs
no tools and calls no model; Brain does the doing and hands the follower the
observation. That split is what makes the decision rules testable with plain
vectors.

Decision rules, in the order they are applied to the step just taken:

    expired            the plan is older than MAX_AGE_S           -> ABORT
    step failed        the tool reported failure                  -> REPLAN
    surprising         the predictive model's normalized surprise
                       is above SURPRISE_LIMIT                    -> REPLAN
    drifting away      we are measurably FURTHER from the goal
                       than where we started                      -> REPLAN
    path complete      every step has been taken                  -> DONE
    arrived early      already as close to the goal as the final
                       imagined state would have been             -> DONE
    otherwise                                                     -> CONTINUE

REPLAN is capped at MAX_REPLANS per plan lineage; past that the answer is
ABORT, because a plan that keeps being wrong is a plan to stop trusting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional

import numpy as np

from .signals import Imagery, Step

logger = logging.getLogger("ash.brain.planfollow")

SURPRISE_LIMIT = 0.75      # normalized surprise above this => the plan's picture is wrong
DRIFT_MARGIN = 0.05        # how much further from the goal than the start counts as "away"
MAX_REPLANS = 2
MAX_AGE_S = 900.0          # a plan is about the present; after 15 minutes it is stale


class Decision(str, Enum):
    CONTINUE = "continue"
    REPLAN = "replan"
    DONE = "done"
    ABORT = "abort"


@dataclass
class Assessment:
    decision: Decision
    reason: str = ""


def _cos(a: Any, b: Any) -> Optional[float]:
    """Cosine of two vectors, or None if either is missing or they do not
    live in the same space (an encoder can change under a running plan)."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 1 or a.shape != b.shape:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(a, b) / (na * nb))


class PlanFollower:
    def __init__(self, surprise_limit: float = SURPRISE_LIMIT,
                 drift_margin: float = DRIFT_MARGIN, max_replans: int = MAX_REPLANS,
                 max_age_s: float = MAX_AGE_S,
                 clock: Callable[[], float] = time.time):
        self.surprise_limit = surprise_limit
        self.drift_margin = drift_margin
        self.max_replans = max_replans
        self.max_age_s = max_age_s
        self.clock = clock
        self._reset()
        self.last_ended: str = ""          # why the previous plan ended, for traces

    def _reset(self):
        self.plan: Optional[Imagery] = None
        self.idx: int = 0                  # steps taken so far
        self.replans: int = 0
        self.born: float = 0.0
        self.start_latent: Optional[Any] = None
        self.state: Optional[Any] = None   # latest observed latent
        self.proposals: List[Any] = []     # what a replan may choose from
        # The outcome of the last step taken but not yet observed/assessed.
        # think() takes step 1 itself and leaves its result here.
        self.pending_output: Any = None
        self.pending_ok: bool = True

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.plan is not None

    def current(self) -> Optional[Step]:
        if self.plan is None or self.idx >= len(self.plan.path):
            return None
        return self.plan.path[self.idx]

    def adopt(self, plan: Imagery, proposals, start_latent=None, completed: int = 0,
              pending_output: Any = None, pending_ok: bool = True, replans: int = 0):
        self.plan = plan
        self.idx = max(0, min(int(completed), len(plan.path)))
        self.replans = int(replans)
        self.born = self.clock()
        self.start_latent = start_latent
        self.state = start_latent
        self.proposals = list(proposals or [])
        self.pending_output = pending_output
        self.pending_ok = pending_ok

    def drop(self, reason: str = ""):
        if self.plan is not None:
            logger.info("Plan ended: %s", reason or "dropped")
        self.last_ended = reason
        self._reset()

    # ------------------------------------------------------------------
    def assess(self, actual_latent: Any, step_ok: bool, surprise: float = 0.0) -> Assessment:
        """Judge the step that was just taken (plan.path[idx - 1]) against the
        state it was imagined to produce. The caller has already advanced idx."""
        plan = self.plan
        if plan is None:
            return Assessment(Decision.ABORT, "no active plan")

        if self.clock() - self.born > self.max_age_s:
            return Assessment(Decision.ABORT, "plan expired")

        if not step_ok:
            return self._replan_or_abort("step failed")

        if surprise > self.surprise_limit:
            return self._replan_or_abort(f"surprising result (surprise {surprise:.2f})")

        goal = plan.goal
        c_now = _cos(actual_latent, goal)
        c_start = _cos(self.start_latent, goal)
        if c_now is not None and c_start is not None and c_now < c_start - self.drift_margin:
            return self._replan_or_abort(
                f"drifting away from the goal ({c_now:.2f} vs {c_start:.2f} at the start)")

        if self.idx >= len(plan.path):
            return Assessment(Decision.DONE, "path complete")

        last_expected = plan.path[-1].expected_latent
        c_final = _cos(last_expected, goal)
        if c_now is not None and c_final is not None and c_now >= c_final:
            return Assessment(Decision.DONE, "arrived at the goal early")

        return Assessment(Decision.CONTINUE, "on track")

    def _replan_or_abort(self, why: str) -> Assessment:
        if self.replans >= self.max_replans:
            return Assessment(Decision.ABORT, f"{why}; replan limit reached")
        return Assessment(Decision.REPLAN, why)
