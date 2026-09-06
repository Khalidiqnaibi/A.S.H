"""
src/brain/brain.py

The cognitive cycle. This is where the diagram becomes control flow.

    Homeostatic Modulator (limbic)
              |  latent drives (broadcast down)
              v
    Sensory Extractors -> Executive Core <-> Predictive Model
                              |                    ^
              fast reflexes   |                    | surprise
                              v                    |
         System 1 Policy  <---+--->  Episodic/Semantic Memory Engine
         (Actor-Critic)              System 2 (deliberation) sits behind
                                     the Executive on the slow path.

One turn = one `think()` call. Nine steps:

     1. Settle the previous turn's deferred reward (the critic learns).
     2. REFLEX CHECK -- exact recent repeat? Return the cached answer. Done.
     3. Sense -- fuse modalities, compute salience and novelty.
     4. Predict -- score last cycle's prediction, emit surprise.
     5. Modulate -- run emotion, derive latent drives, broadcast.
     6. Propose -- System 1 reflex arc + VLA (if it has something to say).
     7. Arbitrate + Gate -- Executive Core picks an action AND a pathway.
     8a. FAST  -> execute tool, render from template. No LLM. No deep retrieval.
     8b. SLOW  -> generate candidates, imagine rollouts, re-arbitrate, execute,
                  full memory retrieval, LLM narration.
     9. Commit -- memory write, outcome recorded, prediction staged for next turn.

Post-hoc escalation
-------------------
Step 8a can fail its own bet. If the fast path executes and the tool errors,
or the result contradicts what the predictive model expected, the cycle is
re-run as SLOW and the fast answer is discarded. That is a reflex that
overshot being caught by deliberation -- exactly the behavior the two-system
split exists to produce, and the reason `Pathway.ESCALATED` is tracked
separately in the trace.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .executive import ExecutiveCore
from .homeostasis import HomeostaticModulator
from .memory_engine import MemoryEngine
from .predictive import PredictiveModel
from .sensory import SensoryCortex
from .signals import (
    ActionClass, ActionProposal, BrainResponse, BrainTrace, Drives,
    Pathway, PerceptBundle, Prediction, Verdict,
)
from .system1 import System1Policy
from .system2 import System2Deliberator
from .vla import VLAChannel

logger = logging.getLogger("ash.brain")

# Action classes whose results may be served from the reflex cache. Anything
# that changes state is excluded -- a cached "deleted" is a lie the second time.
SAFE_TO_CACHE = {
    ActionClass.CONVERSATION,
    ActionClass.RETRIEVAL,
    ActionClass.COMPUTE,
}

CYCLE_COST = {
    Pathway.OBSERVE: 0.02,
    Pathway.REFLEX: 0.05,
    Pathway.FAST: 0.20,
    Pathway.SLOW: 1.00,
    Pathway.ESCALATED: 1.20,
}


def _fmt_tool_output(out: Any) -> str:
    """Template renderer for the fast path. Deliberately boring -- the whole
    point of FAST is that no language model is involved."""
    if out is None:
        return ""
    if isinstance(out, dict):
        for key in ("result", "output", "text", "value", "answer"):
            if key in out and out[key] is not None:
                return str(out[key])
        if out.get("ok") is False:
            return f"That didn't work: {out.get('error', 'unknown error')}"
        return ", ".join(f"{k}: {v}" for k, v in out.items() if k != "ok")
    return str(out)


class Brain:
    """Full cognitive runtime. `ASH` owns one of these."""

    def __init__(
        self,
        *,
        name: str = "A.S.H",
        user: str = "user",
        llm=None,
        embedder=None,
        classifier: Callable[[str], Dict[str, Any]],
        candidate_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        registry,
        memory_router,
        core_memory=None,
        emotion_engine,
        sentiment_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        shapes: Optional[Dict[str, Dict[str, Any]]] = None,
        enable_vision: bool = False,
        latent_dim: int = 384,
        concepts=None,
        concept_latent_dim: int = 256,
    ):
        self.name = name
        self.user = user

        self.sensory = SensoryCortex(embedder=embedder, enable_vision=enable_vision,
                                     latent_dim=latent_dim)

        # The associative layer. When present, the predictive model runs in
        # concept space rather than raw embedding space -- predicting the next
        # *conceptual* state is better-posed, because the surface form that
        # makes consecutive utterances look unrelated has been discarded.
        self.concepts = concepts
        self.concept_latent_dim = concept_latent_dim
        self.predictive = PredictiveModel(
            dim=concept_latent_dim if concepts is not None else latent_dim)
        self.homeostasis = HomeostaticModulator(emotion_engine)
        self.memory = MemoryEngine(memory_router)
        self.system1 = System1Policy(classifier=classifier, registry=registry, shapes=shapes)
        self.vla = VLAChannel(registry=registry)
        self.executive = ExecutiveCore(core_memory=core_memory, vla=self.vla)
        self.system2 = System2Deliberator(
            llm=llm, registry=registry, candidate_fn=candidate_fn, shapes=shapes,
        )

        self.registry = registry
        self.sentiment_fn = sentiment_fn
        self._last_cycle_cost = 0.5
        self.last_trace: Optional[BrainTrace] = None
        self.cycles = 0

        # Counters for introspection / the /brain status endpoint.
        self.pathway_counts: Dict[str, int] = {p.value: 0 for p in Pathway}

    # ==================================================================
    # Associative layer
    # ==================================================================
    def _conceive(self, bundle, learn: bool = True, surprise: float = 0.0):
        """Present the fused percept to the concept graph.

        Returns (activation, latent). The latent replaces the raw embedding
        as the predictive model's state when the graph is enabled. Failure
        here is non-fatal by design: the graph is an addition to a system
        that already worked without it, and a broken association should
        degrade ASH to its pre-concept behaviour rather than break a turn.
        """
        if self.concepts is None or bundle.fused_embedding is None:
            return None, bundle.fused_embedding
        try:
            d = self.homeostasis.drives
            act = self.concepts.perceive(
                bundle.fused_embedding, text=bundle.text, learn=learn,
                novelty=bundle.novelty, surprise=surprise,
                urgency=d.urgency, fatigue=d.fatigue)
            return act, self.concepts.net.latent(act, dim=self.concept_latent_dim)
        except Exception:
            logger.exception("Concept layer failed; falling back to raw embedding")
            return None, bundle.fused_embedding

    # ==================================================================
    # Public entry
    # ==================================================================
    def think(self, query: str, image: Any = None,
              importance: float = 0.5) -> BrainResponse:
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}
        trace = BrainTrace(query=query)
        self.cycles += 1

        # --- 1. Settle previous turn's reward -------------------------
        sentiment = self._sentiment(query)
        self.system1.settle_reward(query, sentiment.get("sentiment"))

        # --- 2. Reflex check ------------------------------------------
        cached = self.system1.reflex_lookup(query)
        if cached is not None:
            text, tool_out = cached
            trace.pathway = Pathway.REFLEX
            trace.notes.append("reflex cache hit -- no sensing, no retrieval, no LLM")
            trace.timings_ms["total"] = (time.perf_counter() - t_start) * 1000.0
            self.memory.push("user", query, salience=0.3)
            self.memory.push("ash", text, salience=0.3)
            self._finish_cycle(Pathway.REFLEX, trace)
            return BrainResponse(
                text=text, trace=trace, tool_output=tool_out,
                pathway=Pathway.REFLEX,
            )

        # --- 3. Sense --------------------------------------------------
        t0 = time.perf_counter()
        bundle = self.sensory.perceive(text=query, image=image)
        self.vla.note_perception(bundle)
        timings["sense"] = (time.perf_counter() - t0) * 1000.0
        trace.percepts = bundle.as_dict()

        # --- 3b. Conceive ----------------------------------------------
        t0 = time.perf_counter()
        activation, latent = self._conceive(
            bundle, surprise=(self.last_trace.prediction.get("surprise", 0.0)
                              if self.last_trace else 0.0))
        timings["conceive"] = (time.perf_counter() - t0) * 1000.0
        if activation is not None:
            trace.concepts = self.concepts.net.explain(activation)

        # --- 4. Predict ------------------------------------------------
        t0 = time.perf_counter()
        prelim = self.system1.propose(query, bundle, Drives(), Prediction())
        candidate_intents = [prelim.intent] if prelim.intent else []
        if activation is not None:
            candidate_intents += list(self.concepts.net.intent_activation(activation))
        prediction = self.predictive.observe(latent, candidate_intents)
        timings["predict"] = (time.perf_counter() - t0) * 1000.0
        trace.prediction = prediction.as_dict()

        # --- 5. Modulate (limbic broadcast) ----------------------------
        t0 = time.perf_counter()
        signals = self._telemetry(query, sentiment, prelim, memory_hits=0)
        drives = self.homeostasis.update(signals, cycle_cost=self._last_cycle_cost)
        timings["modulate"] = (time.perf_counter() - t0) * 1000.0
        trace.drives = drives.as_dict()

        # --- 6. Propose ------------------------------------------------
        t0 = time.perf_counter()
        proposals: List[ActionProposal] = [
            self.system1.propose(query, bundle, drives, prediction)
        ]
        vla_p = self.vla.propose(bundle, drives, prediction)
        if vla_p is not None:
            proposals.append(vla_p)
        if activation is not None:
            try:
                cp = self.concepts.proposer.propose(activation, bundle, drives, prediction)
                if cp is not None and not any(
                        p.intent == cp.intent and p.origin == "system1" for p in proposals):
                    proposals.append(cp)
            except Exception:
                logger.exception("Concept proposer failed")
        timings["propose"] = (time.perf_counter() - t0) * 1000.0

        # --- 7. Arbitrate + gate ---------------------------------------
        t0 = time.perf_counter()
        verdict = self.executive.arbitrate(proposals, bundle, drives, prediction)
        timings["arbitrate"] = (time.perf_counter() - t0) * 1000.0
        trace.proposals = [p.as_dict() for p in proposals]
        trace.verdict = verdict.as_dict()
        # The slow path overwrites trace.verdict with its own re-arbitration,
        # so keep the gate's actual fast-vs-slow reasoning here where it
        # won't be clobbered.
        trace.notes.append(
            f"gate -> {verdict.pathway.value}"
            + (f": {'; '.join(verdict.escalation_reasons[:3])}" if verdict.escalation_reasons else "")
        )

        # --- 8. Execute along the chosen pathway ------------------------
        if verdict.pathway == Pathway.FAST:
            response = self._run_fast(query, bundle, drives, prediction, verdict, trace,
                                      timings, latent=latent, activation=activation)
            if response is not None:
                trace.timings_ms = timings
                trace.timings_ms["total"] = (time.perf_counter() - t_start) * 1000.0
                self._finish_cycle(Pathway.FAST, trace)
                return response
            # Fast path bailed -> fall through to slow, marked as escalated.
            trace.escalated_from = Pathway.FAST.value

        response = self._run_slow(query, bundle, drives, prediction, verdict, trace, timings,
                                  importance=importance, latent=latent, activation=activation)
        trace.timings_ms = timings
        trace.timings_ms["total"] = (time.perf_counter() - t_start) * 1000.0
        final_pathway = Pathway.ESCALATED if trace.escalated_from else Pathway.SLOW
        trace.pathway = final_pathway
        response.pathway = final_pathway
        self._finish_cycle(final_pathway, trace)
        return response

    # ==================================================================
    # Fast path
    # ==================================================================
    def _run_fast(self, query, bundle: PerceptBundle, drives: Drives,
                  prediction: Prediction, verdict: Verdict, trace: BrainTrace,
                  timings: Dict[str, float], latent=None,
                  activation=None) -> Optional[BrainResponse]:
        """Execute a reflex. Returns None to signal 'escalate me'."""
        proposal = verdict.chosen
        if proposal is None or proposal.tool_name is None:
            trace.notes.append("fast path had nothing executable")
            return None

        t0 = time.perf_counter()
        out = self.system1.execute(proposal)
        timings["tool"] = (time.perf_counter() - t0) * 1000.0
        ok = bool(out.get("ok"))

        self.predictive.record_outcome(proposal.intent, ok)

        if not ok:
            # The reflex misfired. Do not paper over it with a template --
            # hand the whole thing to deliberation.
            trace.notes.append(f"fast tool '{proposal.tool_name}' failed: {out.get('error')}")
            self.system1.stage_reward(proposal, tool_success=False, escalated=True,
                                      bundle=bundle, drives=drives, prediction=prediction)
            return None

        # Shallow memory probe: hard core rules only.
        ctx = self.memory.retrieve(query, Pathway.FAST)
        trace.memory_hits = self.memory.hits(ctx)

        body = _fmt_tool_output(out)
        if not body.strip():
            trace.notes.append("fast tool returned nothing renderable")
            self.system1.stage_reward(proposal, tool_success=False, escalated=True)
            return None

        text = self._template_render(body, drives)

        trace.pathway = Pathway.FAST
        trace.tool_output = out
        trace.notes.append(
            f"reflex fired: {proposal.tool_name} (value={proposal.value:.2f}, "
            f"surprise={prediction.surprise:.2f}) -- no LLM invoked"
        )

        self.system1.reflex_store(query, text, out)
        self.system1.stage_reward(proposal, tool_success=True, escalated=False)
        self.memory.commit_turn(query, text, self.user, self.name, pathway=Pathway.FAST)
        self.predictive.commit(latent if latent is not None else bundle.fused_embedding,
                               proposal.intent)
        self._concept_feedback(activation, bundle, proposal, tool_success=True)

        return BrainResponse(
            text=text, trace=trace, tool_output=out,
            intent=proposal.intent, tool_used=proposal.tool_name,
            tool_success=True, pathway=Pathway.FAST,
        )

    @staticmethod
    def _template_render(body: str, drives: Drives) -> str:
        """Tone applied by rule, not by model. Three registers, chosen by
        the drive vector -- enough to not sound robotic, cheap enough to
        keep the fast path fast."""
        body = body.strip()
        if drives.urgency > 0.65 or drives.effort_budget < 0.3:
            return body
        if drives.social > 0.65:
            return f"{body} — anything else you need on that?"
        return body

    # ==================================================================
    # Slow path
    # ==================================================================
    def _run_slow(self, query, bundle: PerceptBundle, drives: Drives,
                  prediction: Prediction, verdict: Verdict, trace: BrainTrace,
                  timings: Dict[str, float], importance: float = 0.5,
                  latent=None, activation=None) -> BrainResponse:
        reasons = verdict.escalation_reasons or ["explicit deliberation"]
        logger.info("Brain: SLOW path -- %s", "; ".join(reasons[:3]))

        # --- generate + imagine ----------------------------------------
        t0 = time.perf_counter()
        candidates = self.system2.generate_candidates(query, bundle, drives, prediction)
        # Keep System 1's proposal in contention -- deliberation should be
        # able to conclude the reflex was right all along.
        if verdict.chosen is not None:
            candidates.append(verdict.chosen)
        candidates = self.system2.imagine(
            candidates, latent if latent is not None else bundle.fused_embedding,
            self.predictive)
        timings["deliberate"] = (time.perf_counter() - t0) * 1000.0

        # --- re-arbitrate over the richer candidate set -----------------
        t0 = time.perf_counter()
        verdict2 = self.executive.arbitrate(candidates, bundle, drives, prediction)
        timings["arbitrate2"] = (time.perf_counter() - t0) * 1000.0
        trace.proposals = [c.as_dict() for c in candidates]
        trace.verdict = verdict2.as_dict()

        chosen = verdict2.chosen
        tool_out, tool_ok, tool_name, intent = None, None, None, None

        if chosen is not None and chosen.tool_name:
            t0 = time.perf_counter()
            if chosen.action_class == ActionClass.PHYSICAL and self.vla.actuators:
                tool_out = self.vla.execute(chosen)
            else:
                tool_out = self.system1.execute(chosen)
            timings["tool"] = (time.perf_counter() - t0) * 1000.0
            tool_ok = bool(tool_out.get("ok"))
            tool_name = chosen.tool_name
            intent = chosen.intent
            self.predictive.record_outcome(intent, tool_ok)
            trace.tool_output = tool_out

        if verdict2.vetoed_by:
            trace.notes.append(f"vetoed by: {', '.join(verdict2.vetoed_by)}")
        if verdict2.deadlocked:
            trace.notes.append(f"board deadlocked (margin {verdict2.margin:.3f})")

        # --- full retrieval ---------------------------------------------
        t0 = time.perf_counter()
        ctx = self.memory.retrieve(query, Pathway.SLOW)
        trace.memory_hits = self.memory.hits(ctx)
        timings["retrieve"] = (time.perf_counter() - t0) * 1000.0

        # --- re-run the limbic update now that we know the outcome -------
        sentiment = self._sentiment(query)
        signals = self._telemetry(query, sentiment, chosen, trace.memory_hits,
                                  tool_used=tool_name, tool_success=tool_ok)
        drives = self.homeostasis.update(signals, cycle_cost=self._last_cycle_cost)
        trace.drives = drives.as_dict()
        emo_result = self.homeostasis.last_emotion_result
        modulation_block = (
            emo_result.modulation.as_prompt_block() if emo_result else "[EMOTIONAL STATE] neutral"
        )

        # --- narrate ------------------------------------------------------
        facts = {
            "intent": intent,
            "tool_used": tool_name,
            "tool_output": tool_out,
            "action_class": chosen.action_class.value if chosen else None,
            "blocked": verdict2.vetoed_by or None,
        }
        note = ""
        if verdict2.vetoed_by:
            note = (f"An action was blocked by {', '.join(verdict2.vetoed_by)}. Tell the user "
                    "plainly that you did not take it and why, without apologizing excessively.")
        elif verdict2.deadlocked:
            note = "Multiple interpretations scored equally. Ask one clarifying question."

        t0 = time.perf_counter()
        text = self.system2.narrate(
            self.name, query, facts, ctx, self.memory.working_block(),
            drives, modulation_block, trace_note=note,
        )
        timings["narrate"] = (time.perf_counter() - t0) * 1000.0

        if not (text or "").strip():
            text = _fmt_tool_output(tool_out) or "I couldn't put together a response for that."
            trace.notes.append("narration empty -- fell back to raw facts")

        # --- commit ---------------------------------------------------------
        if chosen is not None:
            self.system1.stage_reward(
                chosen, tool_success=tool_ok, escalated=bool(trace.escalated_from),
                bundle=bundle, drives=drives, prediction=prediction,
            )
        # Feed the resolved decision back into the board's uncertainty
        # tracker, so influence transfers to whichever member is actually
        # right rather than staying at whatever weights were configured.
        try:
            self.executive.learn_from_outcome(
                verdict2, outcome_good=(tool_ok is not False))
        except Exception:
            logger.exception("Executive outcome feedback failed")

        # Cache the answer even though we thought hard to get it. The reflex
        # cache is about the *result*, not the path -- a deliberated answer to
        # "what time is it" is just as reusable thirty seconds later. Only
        # side-effect-free classes are cached; anything that mutated state or
        # was blocked is deliberately not.
        # A deadlocked turn is still cacheable: the same ambiguous query
        # should get the same clarifying question, not a fresh coin-flip.
        cacheable = (
            chosen is not None
            and not verdict2.vetoed_by
            and chosen.action_class in SAFE_TO_CACHE
            and tool_ok is not False
            and bool((text or "").strip())
        )
        if cacheable:
            self.system1.reflex_store(query, text, tool_out)

        self.memory.commit_turn(query, text, self.user, self.name,
                                importance=importance, pathway=Pathway.SLOW)
        self.predictive.commit(latent if latent is not None else bundle.fused_embedding,
                               intent)
        self._concept_feedback(activation, bundle, chosen, tool_success=tool_ok,
                               summary=f"{query[:80]} -> {text[:120]}")

        return BrainResponse(
            text=text, trace=trace, tool_output=tool_out, intent=intent,
            tool_used=tool_name, tool_success=tool_ok, pathway=Pathway.SLOW,
        )

    # ==================================================================
    # Helpers
    # ==================================================================
    def _sentiment(self, query: str) -> Dict[str, Any]:
        if self.sentiment_fn is None:
            return {"sentiment": "unknown", "confidence": 0.0}
        try:
            return self.sentiment_fn(query) or {"sentiment": "unknown", "confidence": 0.0}
        except Exception:
            logger.exception("sentiment_fn failed")
            return {"sentiment": "unknown", "confidence": 0.0}

    def _telemetry(self, query, sentiment, proposal: Optional[ActionProposal],
                   memory_hits: int, tool_used=None, tool_success=None):
        from tools.emo import TelemetrySignals
        return TelemetrySignals(
            query_text=query,
            sentiment_label=sentiment.get("sentiment"),
            sentiment_score=sentiment.get("confidence", 0.0) or 0.0,
            intent=proposal.intent if proposal else None,
            intent_score=proposal.confidence if proposal else 0.0,
            tool_used=tool_used,
            tool_success=tool_success,
            memory_hits=memory_hits,
        )

    def _finish_cycle(self, pathway: Pathway, trace: BrainTrace):
        self._last_cycle_cost = CYCLE_COST.get(pathway, 0.5)
        self.pathway_counts[pathway.value] = self.pathway_counts.get(pathway.value, 0) + 1
        trace.pathway = pathway
        self.last_trace = trace
        if self.cycles % 20 == 0:
            self.system1.save()
            self.predictive.save()

    # ==================================================================
    # Ambient perception -- the OBSERVE pathway
    # ==================================================================
    def observe(self, text: str = "", image: Any = None,
                source: str = "ambient", salience: float = 0.3) -> BrainTrace:
        """Perceive without responding.

        This is the pathway an always-on ASH spends ~99% of its cycles in. It
        runs the sensory extractors and the predictive model -- so novelty and
        surprise stay calibrated against the *real* stream of the day rather
        than only against the handful of moments someone spoke -- updates the
        limbic state, and drops the observation into working memory.

        It deliberately does NOT: arbitrate, execute a tool, retrieve from the
        memory stores, call the LLM, or write an episode. Persistence of
        ambient events is the journal's job, and consolidation into episodic
        memory happens during sleep. Writing an episode per observation would
        bury the store in noise inside a day.

        Cost is one embedding plus a matrix multiply: sub-millisecond, which
        is what makes running it on every window switch viable.
        """
        t0 = time.perf_counter()
        trace = BrainTrace(query=text or f"<{source}>")
        trace.pathway = Pathway.OBSERVE
        self.cycles += 1

        bundle = self.sensory.perceive(text=text, image=image)
        if not bundle.percepts:
            trace.notes.append("nothing perceptible")
            return trace
        # Sensor-declared salience is evidence about the world; the extractor's
        # is evidence about the signal. Take the stronger claim.
        bundle.salience = max(bundle.salience, salience)
        self.vla.note_perception(bundle)
        trace.percepts = bundle.as_dict()

        activation, latent = self._conceive(bundle)
        if activation is not None:
            trace.concepts = self.concepts.net.explain(activation)

        prediction = self.predictive.observe(latent, [])
        trace.prediction = prediction.as_dict()

        # Limbic update on a cheap cycle. Ambient observation costs almost
        # nothing, so it accrues almost no fatigue -- but novelty and surprise
        # still move curiosity, which is how a long quiet stretch makes ASH
        # more inclined to speak up when something finally happens.
        signals = self._telemetry(text, {"sentiment": "unknown", "confidence": 0.0},
                                  None, memory_hits=0)
        drives = self.homeostasis.update(signals, cycle_cost=0.03)
        trace.drives = drives.as_dict()

        self.memory.push(source, text or f"<{source} percept>", salience=bundle.salience)
        self.predictive.commit(latent, None)

        trace.timings_ms["total"] = (time.perf_counter() - t0) * 1000.0
        self._finish_cycle(Pathway.OBSERVE, trace)
        return trace

    def _concept_feedback(self, activation, bundle, proposal, tool_success=None,
                          summary: str = ""):
        """Close the loop: reward the graph, index the episode, feed the guard.

        The reward is the critic's TD error, which is the same third factor
        the brain's own learning uses -- so the concept graph and the reflex
        critic are consolidating against one signal rather than two
        disagreeing ones.
        """
        if activation is None or self.concepts is None:
            return
        try:
            r = 0.0
            if tool_success is True:
                r += 0.8
            elif tool_success is False:
                r -= 0.8
            if proposal is not None:
                r += 0.4 * float(np.clip(proposal.value, -1.0, 1.0))
            self.concepts.reward(r)

            if summary:
                self.concepts.index.index(f"ep{self.cycles}", summary, activation)
                # One-shot episodic write. Separate from perception on
                # purpose: not every percept deserves an episode, and the
                # acetylcholine encoding gain decides how strongly this one
                # is laid down.
                self.concepts.encode_episode(
                    bundle.fused_embedding, text=summary, act=activation,
                    intent=(proposal.intent if proposal else None), reward=r)

            if proposal is not None and bundle.fused_embedding is not None:
                self.concepts.evaluator.observe(
                    bundle.fused_embedding, proposal.intent,
                    proposal.confidence, tool_success)
        except Exception:
            logger.exception("Concept feedback failed")

    # ==================================================================
    # Introspection
    # ==================================================================
    def status(self) -> Dict[str, Any]:
        total = max(1, sum(self.pathway_counts.values()))
        return {
            "cycles": self.cycles,
            "pathways": self.pathway_counts,
            "pathway_share": {k: round(v / total, 3) for k, v in self.pathway_counts.items()},
            "drives": self.homeostasis.drives.as_dict(),
            "fatigue": round(self.homeostasis.drives.fatigue, 3),
            "critic_weights": self.system1.critic.explain(),
            "critic_updates": self.system1.critic.updates,
            "forward_model_steps": self.predictive.forward.steps,
            "mean_prediction_error": round(self.predictive._err_mean, 4),
            "board": self.executive.board_summary(),
            "working_memory": len(self.memory.wm),
            "concepts": self.concepts.status() if self.concepts else None,
        }

    def sleep(self, seconds_idle: float = 1800.0) -> Dict[str, Any]:
        """Consolidation pass. Called by the maintenance scheduler."""
        from .consolidation import consolidate
        return consolidate(self, seconds_idle=seconds_idle)
