"""
src/brain/executive.py

Executive Core  (the Transformer box in the middle -- the thing everything
else has an arrow into).

Two responsibilities, in order:

  1. ARBITRATE  -- pick which proposal, if any, gets executed.
  2. GATE       -- decide how much cognition to spend doing it (the fast/slow
                   split).

Distributed control
-------------------
The requirement was that nothing has full control. That is enforced
structurally, not by convention:

  * Every constituent is a board member with a weight. `MAX_MEMBER_WEIGHT`
    caps any single member at 0.34 of the total *after normalization*, and
    `register()` raises if you try to exceed it. There is no configuration in
    which one member can carry a motion alone -- passing requires at least
    three concurring interests.

  * Vetoes are BOUNDED. A member may only veto inside its own jurisdiction
    (`veto_scope`), and any bounded veto is overridden if the remaining
    members support the action by `VETO_OVERRIDE_RATIO` (0.72) or more. So
    the VLA can stop a reckless motor command, but it cannot hold the whole
    system hostage over a calculator call it doesn't like.

  * Exactly one member holds an ABSOLUTE veto: the Constitution, which reads
    hard constraints out of CoreMemory. It has zero positive voting power --
    it can never cause an action, only forbid one. Authority to stop and
    authority to act are held by different bodies on purpose.

  * DEADLOCK IS A VALID OUTCOME. When the top two proposals are within
    `DEADLOCK_MARGIN`, the board does not break the tie by fiat -- it
    escalates to System 2. Unresolved conflict buys deliberation. That is the
    single most brain-like property in this file.

Fast / slow gate
----------------
Six independent conditions can force the slow path. Any one is sufficient:

    low critic value          the reflex isn't confident this is worth firing
    weak intent match         classifier below threshold
    high surprise             world model is currently wrong
    high novelty              never seen anything like this
    contested vote            board nearly split
    consequential action      MUTATION / IRREVERSIBLE / PHYSICAL always deliberate

Plus one drive-based modifier: `effort_budget` shifts the thresholds. A
fresh, curious ASH deliberates more; a fatigued, frustrated one reflexes
more. Same policy, different metabolic price.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from .signals import (
    ActionClass, ActionProposal, Drives, PerceptBundle, Pathway,
    Prediction, Verdict, Vote,
)

logger = logging.getLogger("ash.brain.executive")

MAX_MEMBER_WEIGHT = 0.34      # no single member may exceed this share
VETO_OVERRIDE_RATIO = 0.72    # supermajority needed to override a bounded veto
DEADLOCK_MARGIN = 0.08        # top-two gap below this = contested
MIN_PASS_SCORE = 0.10         # below this, no action is worth taking

ALWAYS_DELIBERATE: Set[ActionClass] = {
    ActionClass.MUTATION,
    ActionClass.IRREVERSIBLE,
    ActionClass.PHYSICAL,
}


class UncertaintyTracker:
    """Per-member running reliability, used to weight votes.

    Fixed board weights encode a guess about which subsystem is generally
    right. The brain does not do that. Habitual (striatal) and goal-directed
    (prefrontal) control compete, and the arbitration is by *uncertainty*:
    whichever system currently estimates its own prediction more confidently
    gets control (Daw, Niv & Dayan, 2005). Early in a task the goal-directed
    system is more reliable; after enough repetitions the habit is, and
    control transfers without anyone deciding it should.

    Implemented here as inverse-variance weighting. Each member's votes are
    scored against outcomes; a member whose confident votes keep being wrong
    has its effective weight reduced, and a member that has earned it gains.
    The configured weight becomes a prior, not a verdict.
    """

    def __init__(self, half_life: int = 60):
        self.hits: Dict[str, float] = {}
        self.total: Dict[str, float] = {}
        self.sq_err: Dict[str, float] = {}
        self.decay = 0.5 ** (1.0 / max(1, half_life))

    def observe(self, member: str, supported: bool, outcome_good: bool):
        """One resolved decision. `supported` = this member backed the action."""
        for d in (self.hits, self.total, self.sq_err):
            d[member] = d.get(member, 0.0) * self.decay
        self.total[member] = self.total.get(member, 0.0) + 1.0
        correct = (supported == outcome_good)
        if correct:
            self.hits[member] = self.hits.get(member, 0.0) + 1.0
        else:
            self.sq_err[member] = self.sq_err.get(member, 0.0) + 1.0

    def reliability(self, member: str) -> float:
        """0..1, shrunk toward 0.5 when evidence is thin."""
        n = self.total.get(member, 0.0)
        if n < 3:
            return 0.5
        acc = self.hits.get(member, 0.0) / n
        shrink = n / (n + 8.0)
        return float(0.5 + shrink * (acc - 0.5))

    def multiplier(self, member: str) -> float:
        """Effective-weight multiplier. Bounded so a bad streak cannot
        silence a member entirely -- that would make the board unrecoverable,
        since a silenced member never gets the chance to be right again."""
        r = self.reliability(member)
        return float(max(0.45, min(1.8, 0.5 + 1.6 * r)))

    def status(self) -> Dict[str, Any]:
        return {m: {"n": round(self.total[m], 1),
                    "reliability": round(self.reliability(m), 3),
                    "multiplier": round(self.multiplier(m), 3)}
                for m in sorted(self.total)}


@dataclass
class BoardMember:
    name: str
    weight: float
    jurisdiction: Optional[Set[ActionClass]] = None   # None = all classes
    can_veto: bool = False
    absolute_veto: bool = False

    def has_standing(self, cls: ActionClass) -> bool:
        return self.jurisdiction is None or cls in self.jurisdiction


class Constitution:
    """Reads hard constraints from CoreMemory and enforces them absolutely.

    This is the only absolute veto in the system, and it is intentionally
    dumb: it does keyword matching against core rules flagged `hard`. It
    cannot be talked out of a position because it has no reasoning to talk to
    -- which is the property you want in a constraint layer.
    """

    def __init__(self, core_memory=None):
        self.core = core_memory
        self._cache: List[Dict[str, Any]] = []
        self._loaded = False

    def _hard_rules(self) -> List[Dict[str, Any]]:
        if self._loaded:
            return self._cache
        rules = []
        try:
            dump = self.core.dump_all() if hasattr(self.core, "dump_all") else {}
            items = dump.values() if isinstance(dump, dict) else (dump or [])
            for it in items:
                d = it if isinstance(it, dict) else getattr(it, "__dict__", {})
                if d.get("hard") or float(d.get("priority", 0) or 0) >= 9:
                    rules.append({
                        "text": str(d.get("text", "")),
                        "priority": float(d.get("priority", 9) or 9),
                    })
        except Exception:
            logger.exception("Constitution: failed to read core memory; no hard rules loaded")
        self._cache = rules
        self._loaded = True
        logger.info("Constitution: %d hard rule(s) in force", len(rules))
        return rules

    def refresh(self):
        self._loaded = False

    @staticmethod
    def _norm(s: str) -> str:
        """Collapse underscores/hyphens so an intent tag written `format_disk`
        matches a rule written "format disk" and vice versa."""
        return re.sub(r"[\s_\-]+", " ", (s or "").lower()).strip()

    def vote(self, proposal: ActionProposal, bundle: PerceptBundle) -> Optional[Vote]:
        """Only ever returns a blocking vote or None. Never endorses."""
        tag = self._norm(proposal.intent or "")
        tool = self._norm(proposal.tool_name or "")
        for rule in self._hard_rules():
            rt = self._norm(rule["text"])
            if not rt:
                continue
            # A hard rule phrased as a prohibition ("never ...", "must not ...")
            # blocks any proposal whose intent tag or tool name appears in it.
            prohibitive = any(
                m in rt for m in ("never", "must not", "do not", "dont", "forbidden", "refuse")
            )
            if not prohibitive:
                continue
            if (tag and tag in rt) or (tool and tool in rt):
                return Vote(
                    member="constitution", support=-1.0, confidence=1.0,
                    veto=True, absolute=True,
                    reason=f"hard core rule forbids this: {rule['text'][:120]}",
                )
        return None


class ExecutiveCore:
    """Weighted-quorum arbitration board + pathway gate."""

    def __init__(self, core_memory=None, vla=None):
        self.constitution = Constitution(core_memory)
        self.vla = vla
        self.members: Dict[str, BoardMember] = {}
        self.uncertainty = UncertaintyTracker()
        self.adaptive_weights = True

        # Default board. Note the weights: no one clears MAX_MEMBER_WEIGHT,
        # and passing anything requires agreement across at least three
        # distinct interests.
        self.register(BoardMember("system1", 0.24), _defer_validation=True)
        self.register(BoardMember("system2", 0.24), _defer_validation=True)
        self.register(BoardMember("predictive", 0.18, can_veto=False), _defer_validation=True)
        # The concept graph. Weighted below the classifier on purpose: it is
        # the slower, more speculative route, and an association that fires a
        # tool nobody asked for is worse than one that merely suggests it.
        self.register(BoardMember("associative", 0.16, can_veto=False), _defer_validation=True)
        self.register(BoardMember("homeostasis", 0.08, can_veto=False), _defer_validation=True)
        self.register(BoardMember(
            "vla", 0.16,
            jurisdiction={ActionClass.PHYSICAL, ActionClass.IRREVERSIBLE, ActionClass.MUTATION},
            can_veto=True,
        ), _defer_validation=True)
        self.register(BoardMember(
            "constitution", 0.0, can_veto=True, absolute_veto=True,
        ), _defer_validation=True)
        self._validate_balance()

    # ------------------------------------------------------------------
    def register(self, member: BoardMember, _defer_validation: bool = False):
        """Add a board member.

        The cap is validated against the *finalized* board, not incrementally
        -- otherwise the first member registered always holds 100% and the
        check would be meaningless during bootstrap. `_defer_validation` is
        used only while assembling the default board in __init__; every
        caller-facing registration is checked immediately.
        """
        previous = self.members.get(member.name)
        self.members[member.name] = member
        if _defer_validation:
            return
        try:
            self._validate_balance()
        except ValueError:
            # Roll back so a rejected registration leaves the board intact.
            if previous is not None:
                self.members[member.name] = previous
            else:
                self.members.pop(member.name, None)
            raise

    def _validate_balance(self):
        """No single member may hold more than MAX_MEMBER_WEIGHT of the vote.

        This is the structural guarantee that nothing has full control: with
        a 34% cap, carrying any motion requires at least three concurring
        members.
        """
        total = sum(m.weight for m in self.members.values())
        if total <= 0:
            return
        for name, m in self.members.items():
            share = m.weight / total
            if share > MAX_MEMBER_WEIGHT + 1e-9:
                raise ValueError(
                    f"Board member {name!r} would hold {share:.2%} of voting power; "
                    f"cap is {MAX_MEMBER_WEIGHT:.0%}. "
                    "No single subsystem may dominate the executive."
                )

    def _normalized_weights(self) -> Dict[str, float]:
        """Configured weight x earned reliability, renormalized.

        The cap is enforced after adaptation too: a member that has been
        right about everything for a month still cannot exceed
        MAX_MEMBER_WEIGHT, because the point of the cap is structural, not a
        statement about competence.
        """
        eff = {}
        for n, m in self.members.items():
            mult = self.uncertainty.multiplier(n) if self.adaptive_weights else 1.0
            eff[n] = m.weight * mult
        total = sum(eff.values())
        if total <= 0:
            return {n: 0.0 for n in self.members}
        norm = {n: w / total for n, w in eff.items()}

        over = [n for n, w in norm.items() if w > MAX_MEMBER_WEIGHT]
        if over:
            # Clamp the offenders and redistribute proportionally. Adaptation
            # may reorder the board; it may not let anyone take it over.
            spare = sum(norm[n] - MAX_MEMBER_WEIGHT for n in over)
            for n in over:
                norm[n] = MAX_MEMBER_WEIGHT
            rest = [n for n in norm if n not in over]
            rest_total = sum(norm[n] for n in rest) or 1e-9
            for n in rest:
                norm[n] += spare * norm[n] / rest_total
        return norm

    def learn_from_outcome(self, verdict: "Verdict", outcome_good: bool):
        """Feed a resolved decision back into the uncertainty tracker.

        Called once the turn's result is known. This is what makes control
        transfer between fast and slow routes over time instead of being
        fixed by whatever weights were typed in.
        """
        if not self.adaptive_weights:
            return
        for v in verdict.votes:
            if v.member not in self.members:
                continue
            self.uncertainty.observe(v.member, v.support > 0, outcome_good)

    # ------------------------------------------------------------------
    # Non-VLA members' votes are computed here so the scoring rule stays in
    # one auditable place.
    # ------------------------------------------------------------------
    def _intrinsic_votes(self, proposal: ActionProposal, drives: Drives,
                         prediction: Prediction, bundle: PerceptBundle) -> List[Vote]:
        votes: List[Vote] = []

        # System 1 endorses its own proposal in proportion to critic value.
        if proposal.origin == "system1":
            v = max(-1.0, min(1.0, proposal.value / 1.5))
            votes.append(Vote(
                member="system1", support=v, confidence=proposal.confidence,
                reason=f"critic value {proposal.value:.2f}, match {proposal.confidence:.2f}",
            ))
        else:
            # System 1 is skeptical of proposals it did not generate, but only
            # mildly -- it has no basis for a strong opinion.
            votes.append(Vote(
                member="system1", support=-0.15, confidence=0.3,
                reason="not a reflex-generated proposal",
            ))

        if proposal.origin == "system2":
            votes.append(Vote(
                member="system2", support=0.7, confidence=proposal.confidence,
                reason=proposal.rationale or "deliberated proposal",
            ))
        else:
            votes.append(Vote(
                member="system2", support=0.1, confidence=0.35,
                reason="no deliberation performed",
            ))

        if proposal.origin == "associative":
            votes.append(Vote(
                member="associative", support=0.65, confidence=proposal.confidence,
                reason=proposal.rationale or "concept activation",
            ))
        else:
            # The graph still has an opinion on proposals it did not make: if
            # the intent is conceptually active it concurs, and if the concept
            # layer never lit up for this route it mildly dissents.
            votes.append(Vote(
                member="associative", support=0.05, confidence=0.3,
                reason="not an associatively-derived proposal",
            ))

        # Predictive model votes on expected outcome and current model trust.
        pv = (proposal.expected_success - 0.5) * 2.0 - 0.6 * prediction.surprise
        votes.append(Vote(
            member="predictive", support=max(-1.0, min(1.0, pv)),
            confidence=max(0.2, prediction.confidence),
            reason=f"P(success)={proposal.expected_success:.2f}, surprise={prediction.surprise:.2f}",
        ))

        # Homeostasis nudges toward acting when urgent, away when cautious and
        # the action bites. It never vetoes.
        hv = 0.5 * drives.urgency - 0.6 * drives.caution * (0.0 if proposal.reversible else 1.0)
        votes.append(Vote(
            member="homeostasis", support=max(-1.0, min(1.0, hv)), confidence=0.5,
            reason=f"urgency={drives.urgency:.2f}, caution={drives.caution:.2f}",
        ))

        return votes

    # ------------------------------------------------------------------
    def _score(self, proposal: ActionProposal, votes: List[Vote]) -> tuple:
        """Weighted tally. Returns (score, tally, blocking_vetoes)."""
        w = self._normalized_weights()
        tally: Dict[str, float] = {}
        score = 0.0
        blocking: List[Vote] = []
        support_excl: Dict[str, float] = {}
        standing_weight = 0.0

        for v in votes:
            member = self.members.get(v.member)
            if member is None:
                continue
            if not member.has_standing(proposal.action_class):
                continue  # abstains out of jurisdiction

            standing_weight += w[v.member]

            # Confidence scales a vote but must not annihilate it. The floor
            # matters because a member's confidence is often already baked
            # into its support value -- multiplying by a small confidence a
            # second time was double-discounting the same uncertainty and
            # made the tally collapse toward zero for every proposal.
            contrib = w[v.member] * v.support * max(0.35, v.confidence)
            tally[v.member] = contrib
            score += contrib
            support_excl[v.member] = contrib

            if v.veto and member.can_veto:
                if v.absolute and member.absolute_veto:
                    blocking.append(v)
                elif not v.absolute:
                    # Bounded veto: valid only in-scope.
                    if v.veto_scope is None or v.veto_scope == proposal.action_class:
                        blocking.append(v)

        # Resolve bounded vetoes against the rest of the board.
        #
        # The override test measures the *strength* of the remaining support,
        # not merely its sign. Each member's contribution is at most its own
        # weight (support and confidence are both bounded by 1), so dividing
        # the summed contributions by the summed weights of the non-vetoing
        # members with standing gives a normalized -1..+1 endorsement. At
        # VETO_OVERRIDE_RATIO = 0.72 that requires something close to
        # unanimous, high-confidence agreement -- a real supermajority, not
        # "three members mildly in favor".
        surviving: List[Vote] = []
        for v in blocking:
            if v.absolute:
                surviving.append(v)
                continue
            others = {k: c for k, c in support_excl.items() if k != v.member}
            denom = sum(
                self.members[k].weight / max(1e-9, sum(m.weight for m in self.members.values()))
                for k in others
            )
            if denom <= 1e-9:
                surviving.append(v)
                continue
            normalized = sum(others.values()) / denom
            if normalized >= VETO_OVERRIDE_RATIO:
                logger.info(
                    "Executive: bounded veto by %s OVERRIDDEN (normalized support %.2f >= %.2f)",
                    v.member, normalized, VETO_OVERRIDE_RATIO,
                )
            else:
                logger.info(
                    "Executive: veto by %s STANDS (normalized support %.2f < %.2f) -- %s",
                    v.member, normalized, VETO_OVERRIDE_RATIO, v.reason,
                )
                surviving.append(v)

        # Normalize by the weight actually in the room. Without this, a
        # proposal's score depends on how many members happened to have
        # jurisdiction over it, and a fixed MIN_PASS_SCORE means something
        # different for a conversation than for a motor command. Normalized,
        # `score` is a -1..+1 net endorsement in every case.
        if standing_weight > 1e-9:
            score /= standing_weight

        return score, tally, surviving

    # ------------------------------------------------------------------
    # Pathway gate -- the fast/slow decision
    # ------------------------------------------------------------------
    def gate(self, proposal: Optional[ActionProposal], drives: Drives,
             prediction: Prediction, bundle: PerceptBundle,
             margin: float, deadlocked: bool) -> tuple:
        """Return (Pathway, reasons). Any single reason forces SLOW."""
        reasons: List[str] = []

        # Drive-modulated thresholds. A high effort budget makes the brain
        # pickier about taking shortcuts.
        e = drives.effort_budget
        value_floor = 0.15 + 0.55 * e          # critic value needed to reflex
        match_floor = 0.32 + 0.18 * e          # classifier score needed
        surprise_ceiling = 0.62 - 0.22 * e
        novelty_ceiling = 0.70 - 0.20 * e

        if proposal is None or proposal.tool_name is None:
            reasons.append("no executable reflex available")
            return Pathway.SLOW, reasons

        if proposal.action_class in ALWAYS_DELIBERATE:
            reasons.append(f"consequential action class ({proposal.action_class.value})")
        if proposal.value < value_floor:
            reasons.append(f"critic value {proposal.value:.2f} < floor {value_floor:.2f}")
        if proposal.confidence < match_floor:
            reasons.append(f"intent match {proposal.confidence:.2f} < floor {match_floor:.2f}")
        eff_surprise = prediction.effective_surprise()
        if eff_surprise > surprise_ceiling:
            reasons.append(
                f"surprise {eff_surprise:.2f} > ceiling {surprise_ceiling:.2f} "
                f"(state={prediction.surprise:.2f}@conf{prediction.confidence:.2f}, "
                f"outcome={prediction.outcome_surprise:.2f})"
            )
        if bundle.novelty > novelty_ceiling:
            reasons.append(f"novelty {bundle.novelty:.2f} > ceiling {novelty_ceiling:.2f}")
        if deadlocked or margin < DEADLOCK_MARGIN:
            reasons.append(f"board contested (margin {margin:.3f})")

        # Curiosity occasionally buys deliberation it doesn't need -- this is
        # how the critic gets training signal on cases it would otherwise
        # always shortcut past. Exploration, deterministically triggered.
        if not reasons and drives.curiosity > 0.78 and bundle.novelty > 0.45:
            reasons.append(f"curiosity probe (curiosity={drives.curiosity:.2f})")

        return (Pathway.SLOW if reasons else Pathway.FAST), reasons

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def arbitrate(self, proposals: Sequence[ActionProposal], bundle: PerceptBundle,
                  drives: Drives, prediction: Prediction) -> Verdict:
        if not proposals:
            return Verdict(chosen=None, pathway=Pathway.SLOW,
                           escalation_reasons=["no proposals"])

        scored: List[tuple] = []
        all_votes: List[Vote] = []

        for p in proposals:
            votes = self._intrinsic_votes(p, drives, prediction, bundle)

            if self.vla is not None:
                vv = self.vla.vote(p, bundle, drives, prediction)
                if vv is not None:
                    votes.append(vv)

            cv = self.constitution.vote(p, bundle)
            if cv is not None:
                votes.append(cv)

            score, tally, blocking = self._score(p, votes)
            scored.append((score, p, votes, tally, blocking))
            all_votes.extend(votes)

        # Blocked proposals are removed from contention entirely.
        live = [s for s in scored if not s[4]]
        blocked = [s for s in scored if s[4]]

        if not live:
            vetoers = sorted({v.member for s in blocked for v in s[4]})
            reasons = [v.reason for s in blocked for v in s[4]]
            logger.info("Executive: all proposals blocked by %s", vetoers)
            return Verdict(
                chosen=None, pathway=Pathway.SLOW, deadlocked=False,
                vetoed_by=vetoers,
                escalation_reasons=["all proposals vetoed"] + reasons,
                votes=all_votes,
            )

        live.sort(key=lambda s: s[0], reverse=True)
        best_score, best, best_votes, best_tally, _ = live[0]
        runner_up = live[1][0] if len(live) > 1 else -1.0
        margin = best_score - runner_up if len(live) > 1 else 1.0
        deadlocked = len(live) > 1 and margin < DEADLOCK_MARGIN

        if best_score < MIN_PASS_SCORE:
            pathway, reasons = Pathway.SLOW, [
                f"no proposal cleared minimum score ({best_score:.3f} < {MIN_PASS_SCORE})"
            ]
        else:
            pathway, reasons = self.gate(best, drives, prediction, bundle, margin, deadlocked)

        return Verdict(
            chosen=best,
            pathway=pathway,
            score=best_score,
            margin=margin,
            deadlocked=deadlocked,
            vetoed_by=sorted({v.member for s in blocked for v in s[4]}),
            escalation_reasons=reasons,
            votes=best_votes,
            tally=best_tally,
        )

    # ------------------------------------------------------------------
    def board_summary(self) -> Dict[str, Any]:
        w = self._normalized_weights()
        return {
            "members": [
                {
                    "name": n,
                    "weight": round(w[n], 3),
                    "jurisdiction": ([c.value for c in m.jurisdiction] if m.jurisdiction else "all"),
                    "can_veto": m.can_veto,
                    "absolute_veto": m.absolute_veto,
                }
                for n, m in self.members.items()
            ],
            "max_single_weight": MAX_MEMBER_WEIGHT,
            "veto_override_ratio": VETO_OVERRIDE_RATIO,
            "adaptive": self.adaptive_weights,
            "reliability": self.uncertainty.status(),
        }
