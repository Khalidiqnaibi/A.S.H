"""
src/brain/system2.py

System 2 -- the deliberative path.

Reached only when the Executive Core's gate refuses to let a reflex through.
Three phases:

  1. CANDIDATE GENERATION
     Take the classifier's top-k intents (not just the argmax, which is all
     the fast path ever saw) and turn each into a proposal. Optionally ask
     the LLM for one more candidate.

  2. IMAGINATION
     Roll every candidate forward through the Predictive Model. A proposal
     whose imagined trajectory drifts wildly or fails to settle gets marked
     down before it is ever executed. This is the part that makes System 2
     genuinely slower and genuinely better rather than just "same thing but
     with an LLM bolted on".

  3. NARRATION
     Render the final answer through the LLM using memory context, latent
     drives, and the emotional modulation profile.

The LLM's authority
-------------------
The LLM is a *proposer* and a *narrator*. It may put a candidate action on
the table; that candidate then goes to the board and gets weighted like
anyone else's, subject to VLA veto and Constitution veto. It cannot execute
anything itself, and it holds 0.26 of the vote -- below the 0.34 cap, so it
cannot pass a motion alone either.

That's the compromise the original README's "LLM as narrator" and the
diagram's "Transformer as Executive Core" both survive: the LLM gained the
right to *suggest*, and gained nothing else.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from .signals import (
    ActionClass, ActionProposal, Drives, PerceptBundle, Prediction,
)
from .system1 import classify_action

logger = logging.getLogger("ash.brain.system2")

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class System2Deliberator:
    def __init__(self, llm, registry, candidate_fn=None, shapes: Optional[Dict[str, Any]] = None,
                 allow_llm_proposals: bool = True, rollout_horizon: int = 3):
        self.llm = llm
        self.registry = registry
        self.candidate_fn = candidate_fn      # callable(query, top_k) -> [{"intent","score"}]
        self.shapes = shapes or {}
        self.allow_llm_proposals = allow_llm_proposals
        self.rollout_horizon = rollout_horizon

    # ------------------------------------------------------------------
    # Phase 1: candidates
    # ------------------------------------------------------------------
    def generate_candidates(self, query: str, bundle: PerceptBundle, drives: Drives,
                            prediction: Prediction, top_k: int = 4) -> List[ActionProposal]:
        proposals: List[ActionProposal] = []

        raw: List[Dict[str, Any]] = []
        if self.candidate_fn is not None:
            try:
                raw = self.candidate_fn(query, top_k) or []
            except Exception:
                logger.exception("Candidate generation failed")

        for c in raw:
            intent = c.get("intent")
            score = float(c.get("score", 0.0) or 0.0)
            entry = self.registry.match(intent) if intent else None
            if entry is None:
                continue
            cls, reversible = classify_action(intent, self.shapes.get(intent or ""))
            proposals.append(ActionProposal(
                intent=intent,
                tool_name=entry.name,
                args=query,
                origin="system2",
                action_class=cls,
                confidence=score,
                expected_success=prediction.action_outcomes.get(intent, 0.5),
                reversible=reversible,
                rationale=f"deliberative candidate #{len(proposals) + 1} (match {score:.2f})",
            ))

        # A pure-conversation option is always on the table. Without it, the
        # board can only choose between tools, and "just answer the person"
        # would never win.
        proposals.append(ActionProposal(
            intent=None, tool_name=None, args=query, origin="system2",
            action_class=ActionClass.CONVERSATION,
            confidence=0.5, expected_success=0.7, reversible=True,
            rationale="answer conversationally, no tool",
        ))

        if self.allow_llm_proposals and self.llm is not None:
            extra = self._llm_proposal(query, bundle, prediction)
            if extra is not None:
                proposals.append(extra)

        return proposals

    def _llm_proposal(self, query: str, bundle: PerceptBundle,
                      prediction: Prediction) -> Optional[ActionProposal]:
        """Ask the LLM to nominate a tool. Strictly a nomination."""
        tools = self.registry.list_tools()
        if not tools:
            return None
        catalog = "\n".join(
            f"- {t['name']}: intents={t['intents']} :: {(t.get('description') or '')[:120]}"
            for t in tools[:40]
        )
        prompt = (
            "You are a tool-selection module. You do NOT answer the user and you do NOT "
            "execute anything. You nominate at most one tool for a decision board to vote on.\n\n"
            f"Available tools:\n{catalog}\n\n"
            f"User request: {query}\n\n"
            'Reply with ONLY raw JSON, no prose, no markdown: '
            '{"intent": "<intent tag or null>", "confidence": <0..1>, "why": "<one short clause>"}'
        )
        try:
            text = self._call_llm(prompt, system="Output raw JSON only.")
            m = _JSON_FENCE.search(text or "")
            payload = json.loads(m.group(1) if m else (text or "").strip())
        except Exception as e:
            logger.debug("LLM proposal unusable: %s", e)
            return None

        intent = payload.get("intent")
        if not intent or str(intent).lower() in ("null", "none"):
            return None
        entry = self.registry.match(intent)
        if entry is None:
            logger.debug("LLM nominated unknown intent %r -- discarded", intent)
            return None

        cls, reversible = classify_action(intent, self.shapes.get(intent))
        return ActionProposal(
            intent=intent, tool_name=entry.name, args=query, origin="system2",
            action_class=cls,
            confidence=float(min(1.0, max(0.0, float(payload.get("confidence", 0.5) or 0.5)))),
            expected_success=prediction.action_outcomes.get(intent, 0.5),
            reversible=reversible,
            rationale=f"LLM nomination: {str(payload.get('why', ''))[:100]}",
        )

    # ------------------------------------------------------------------
    # Phase 2: imagination
    # ------------------------------------------------------------------
    def imagine(self, proposals: Sequence[ActionProposal], latent, predictive) -> List[ActionProposal]:
        """Roll each candidate forward and fold the result into its value."""
        for p in proposals:
            try:
                r = predictive.rollout(latent, p, horizon=self.rollout_horizon)
            except Exception:
                logger.exception("Rollout failed for %r", p.intent)
                continue
            # Reward stable, plausible trajectories; punish runaway drift.
            p.value = (
                1.2 * (r["expected_success"] - 0.5)
                + 0.5 * r["stability"]
                - 0.7 * r["drift"]
                + 0.4 * p.confidence
            )
            p.expected_success = r["expected_success"]
            p.rationale += f" | rollout drift={r['drift']:.2f} stab={r['stability']:.2f}"
        return list(proposals)

    # ------------------------------------------------------------------
    # Phase 3: narration
    # ------------------------------------------------------------------
    def narrate(self, name: str, query: str, facts: Dict[str, Any], memory: Dict[str, str],
                working: str, drives: Drives, modulation_block: str,
                trace_note: str = "") -> str:
        system_content = (
            f"You are {name}, a loyal personal assistant. Use the facts below verbatim where "
            "applicable. Do NOT invent facts. Do NOT claim to have taken actions that are not "
            "listed in the facts. Answer the query, then one short closing sentence."
        )

        parts = [f"Working memory (most recent first is last):\n{working or 'empty'}\n"]
        parts.append(f"User query: {query}\n")
        for key, label in (("core", "Relevant Core Constraints"),
                           ("episodic", "Relevant Episodic Memory"),
                           ("entity", "Relevant Entities & Grounding")):
            if memory.get(key):
                parts.append(f"{label}:\n{memory[key]}\n")
        parts.append("Facts (use if present):\n" + json.dumps(facts, indent=2, default=str) + "\n")
        parts.append(drives.as_prompt_block() + "\n")
        parts.append(modulation_block + "\n")
        if trace_note:
            parts.append(f"[COGNITIVE NOTE] {trace_note}\n")
        parts.append("Follow the tone directives above. Respond in a small paragraph.")

        return self._call_llm("\n".join(parts), system=system_content)

    # ------------------------------------------------------------------
    def _call_llm(self, human: str, system: str = "") -> str:
        """Tolerant invocation across the wrapper shapes ASH's LLM class and
        langchain both present."""
        if self.llm is None:
            return ""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [SystemMessage(content=system), HumanMessage(content=human)]
        except Exception:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": human}]

        for attr in ("invoke", "chat", "generate", "complete"):
            fn = getattr(self.llm, attr, None)
            if not callable(fn):
                continue
            try:
                resp = fn(messages)
                if isinstance(resp, tuple):
                    return str(resp[0])
                return getattr(resp, "content", None) or str(resp)
            except Exception as e:
                logger.debug("LLM.%s failed: %s", attr, e)

        try:
            if callable(self.llm):
                return str(self.llm(human))
        except Exception:
            logger.exception("LLM callable failed")
        return ""
