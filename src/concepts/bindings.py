"""
src/concepts/bindings.py

Where the associative layer touches the rest of ASH.

Three attachment points, chosen to match how the brain actually divides this
work rather than to maximise how much of the pipeline the graph owns:

  PROPOSER   The graph nominates an intent at the Executive board, competing
             with the classifier's proposal. It does NOT replace the
             classifier. That is not a hedge -- the basal ganglia run a fast
             stimulus-response habit path in parallel with cortical
             association, and the two compete downstream. The embedding
             classifier is the habit path. Having both is the faithful
             arrangement; replacing one with the other is not.

  RETRIEVAL  Memory recall by spreading activation instead of by cosine
             top-k. Episodes are linked to the concepts that were active when
             they were laid down, so recall follows the same associative
             structure everything else does, and gets better as the graph
             does.

  LATENT     The predictive model's state vector becomes the projected
             concept activation rather than the raw sentence embedding.
             Predicting the next conceptual state is a better-posed problem:
             the surface form that makes consecutive utterances look
             unrelated has already been discarded.

Seeding
-------
The graph does not start empty. Intents, tools, entities and CoreMemory rules
all become units, and prohibitive core rules get FROZEN inhibitory edges to
the intents they forbid. Those edges are exempt from every plasticity pass:
the network may learn anything at all, but it may not learn its way around a
constraint. This is the concept-layer equivalent of the Constitution's
absolute veto, and it exists for the same reason.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hippocampus import Hippocampus
from .model import Activation, Origin, SynType
from .network import ConceptNetwork
from .neuromod import Neuromodulators
from .plasticity import Plasticity
from .sleep import SleepCycle

logger = logging.getLogger("ash.concepts.bindings")


# ----------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------
def seed_from_ash(net: ConceptNetwork, embedder, registry=None, core_memory=None,
                  entity_memory=None, intent_catalog=None) -> Dict[str, int]:
    """Bootstrap the vocabulary from what ASH already knows."""
    counts = {"intents": 0, "tools": 0, "core": 0, "entities": 0, "links": 0}

    def embed(text: str) -> Optional[np.ndarray]:
        try:
            v = embedder.encode([text], convert_to_numpy=True,
                                normalize_embeddings=True)[0]
            return np.asarray(v, dtype=np.float32)
        except Exception:
            logger.exception("Embedding failed for %r", text[:40])
            return None

    # --- intents and tools ------------------------------------------
    intent_ids: Dict[str, int] = {}
    if registry is not None:
        try:
            for t in registry.list_tools():
                for intent in t.get("intents", []):
                    if intent in intent_ids:
                        continue
                    text = f"{intent.replace('_', ' ')}. {t.get('description', '')}"
                    v = embed(text)
                    if v is None:
                        continue
                    c = net.add_concept(v, label=intent, origin=Origin.INTENT,
                                        intent=intent, seed_text=text[:200])
                    intent_ids[intent] = c.id
                    counts["intents"] += 1
        except Exception:
            logger.exception("Seeding from tool registry failed")

    # Phrasing variants from the intent catalog widen each seed's basin
    # before any user has said anything.
    if intent_catalog is not None:
        try:
            for intent, phrases in _catalog_phrases(intent_catalog).items():
                cid = intent_ids.get(intent)
                if cid is None or not phrases:
                    continue
                c = net.concepts[cid]
                vecs = [embed(p) for p in phrases[:12]]
                vecs = [v for v in vecs if v is not None]
                if vecs:
                    m = np.mean(np.stack([c.prototype] + vecs), axis=0)
                    n = float(np.linalg.norm(m))
                    if n > 1e-9:
                        c.prototype = (m / n).astype(np.float32)
                        net._dirty = True
        except Exception:
            logger.exception("Seeding phrase variants failed")

    # --- core memory --------------------------------------------------
    if core_memory is not None:
        try:
            dump = core_memory.dump_all() if hasattr(core_memory, "dump_all") else {}
            items = dump.values() if isinstance(dump, dict) else (dump or [])
            for it in items:
                d = it if isinstance(it, dict) else getattr(it, "__dict__", {})
                text = str(d.get("text", "")).strip()
                if not text:
                    continue
                v = embed(text)
                if v is None:
                    continue
                hard = bool(d.get("hard")) or float(d.get("priority", 0) or 0) >= 9
                c = net.add_concept(v, label=f"rule:{text[:28]}", origin=Origin.CORE,
                                    seed_text=text, frozen=hard)
                counts["core"] += 1

                # A prohibitive hard rule gets a frozen inhibitory edge to
                # every intent it names. Spreading activation will actively
                # suppress those routes, and no amount of Hebbian learning
                # can weaken the edge.
                if hard and _is_prohibitive(text):
                    low = re.sub(r"[\s_\-]+", " ", text.lower())
                    for intent, iid in intent_ids.items():
                        tag = intent.replace("_", " ")
                        if tag and tag in low:
                            net.link(c.id, iid, weight=-1.0, bidirectional=False,
                                     frozen=True, kind=SynType.INHIBIT)
                            counts["links"] += 1
                            logger.info("Frozen inhibitory link: %r -/-> %s",
                                        text[:50], intent)
        except Exception:
            logger.exception("Seeding from core memory failed")

    # --- entities -------------------------------------------------------
    if entity_memory is not None:
        try:
            entities = []
            for attr in ("dump_all", "all_entities", "list_entities"):
                fn = getattr(entity_memory, attr, None)
                if callable(fn):
                    entities = fn()
                    break
            items = entities.values() if isinstance(entities, dict) else (entities or [])
            for e in items:
                d = e if isinstance(e, dict) else getattr(e, "__dict__", {})
                name = str(d.get("name", "")).strip()
                if not name:
                    continue
                text = f"{name} ({d.get('type', 'thing')})"
                v = embed(text)
                if v is None:
                    continue
                net.add_concept(v, label=name, origin=Origin.ENTITY, seed_text=text)
                counts["entities"] += 1
        except Exception:
            logger.exception("Seeding from entity memory failed")

    logger.info("Concept graph seeded: %s", counts)
    return counts


def _is_prohibitive(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in
               ("never", "must not", "do not", "don't", "forbidden", "refuse"))


def _catalog_phrases(catalog) -> Dict[str, List[str]]:
    """Best-effort extraction of example phrasings from the intent catalog.
    Written defensively because the catalog's internals are not this module's
    business and change shape as ASH grows."""
    out: Dict[str, List[str]] = {}
    for attr in ("commands", "entries", "items", "_commands", "catalog"):
        data = getattr(catalog, attr, None)
        if not data:
            continue
        try:
            it = data.items() if isinstance(data, dict) else enumerate(data)
            for key, val in it:
                d = val if isinstance(val, dict) else getattr(val, "__dict__", {})
                tag = d.get("tag") or d.get("intent") or (key if isinstance(key, str) else None)
                phrases = d.get("phrases") or d.get("examples") or d.get("utterances") or []
                if tag and phrases:
                    out[str(tag)] = [str(p) for p in phrases]
        except Exception:
            continue
        if out:
            break
    return out


# ----------------------------------------------------------------------
# Board proposer
# ----------------------------------------------------------------------
class ConceptProposer:
    """Turns concept activation into an ActionProposal for the Executive.

    Its confidence is deliberately discounted relative to the classifier's
    when the winning intent was reached purely by association rather than by
    direct match. An inferred route is a suggestion; a matched one is
    evidence. Conflating them is how an associative layer starts firing tools
    nobody asked for.
    """

    def __init__(self, net: ConceptNetwork, registry, shapes=None,
                 inferred_discount: float = 0.55):
        self.net = net
        self.registry = registry
        self.shapes = shapes or {}
        self.inferred_discount = inferred_discount

    def propose(self, act: Activation, bundle, drives, prediction):
        from src.brain.signals import ActionClass, ActionProposal
        from src.brain.system1 import classify_action

        scores = self.net.intent_activation(act)
        if not scores:
            return None

        # Inhibition: a negative-weight edge from a frozen core rule shows up
        # as suppression, so subtract it explicitly rather than relying on the
        # spread to have gone negative.
        for cid, a in act.spread.items():
            for dst, syn in self.net._out.get(cid, {}).items():
                if syn.weight() >= 0:
                    continue
                target = self.net.concepts.get(dst)
                if target is not None and target.intent in scores:
                    scores[target.intent] += a * syn.weight()

        intent, score = max(scores.items(), key=lambda kv: kv[1])
        if score <= 0.05:
            return None

        entry = self.registry.match(intent)
        if entry is None:
            return None

        cid = self.net._by_intent.get(intent)
        was_matched = cid is not None and cid in act.initial
        confidence = float(min(1.0, score if was_matched
                               else score * self.inferred_discount))

        cls, reversible = classify_action(intent, self.shapes.get(intent))
        route = "matched" if was_matched else "inferred by association"
        return ActionProposal(
            intent=intent, tool_name=entry.name, args=bundle.text,
            origin="associative", action_class=cls, confidence=confidence,
            expected_success=prediction.action_outcomes.get(intent, 0.5),
            reversible=reversible,
            rationale=f"concept graph: {self.net.describe(cid) if cid else intent} "
                      f"@ {score:.2f} ({route})",
        )


# ----------------------------------------------------------------------
# Associative memory index
# ----------------------------------------------------------------------
class ConceptMemoryIndex:
    """Links episodes to the concepts active when they were stored, then
    recalls by activation instead of by cosine.

    The practical difference: cosine retrieval finds episodes that *sound
    like* the query. Activation retrieval finds episodes that were laid down
    in a similar conceptual context, which is what you actually want when the
    query shares no vocabulary with the memory.
    """

    def __init__(self, net: ConceptNetwork, max_links_per_episode: int = 6):
        self.net = net
        self.max_links = max_links_per_episode
        self.by_concept: Dict[int, List[Tuple[str, float]]] = {}
        self.episodes: Dict[str, str] = {}

    def index(self, episode_id: str, summary: str, act: Activation):
        self.episodes[episode_id] = summary
        for cid, a in act.top(self.max_links):
            self.by_concept.setdefault(cid, []).append((episode_id, a))
            # Bound per-concept fan-out so a very general concept does not
            # accumulate the entire history and become useless for recall.
            lst = self.by_concept[cid]
            if len(lst) > 200:
                self.by_concept[cid] = sorted(lst, key=lambda kv: -kv[1])[:120]

    def recall(self, act: Activation, n: int = 5) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for cid, a in act.spread.items():
            for eid, strength in self.by_concept.get(cid, []):
                scores[eid] = scores.get(eid, 0.0) + a * strength
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
        return [(self.episodes.get(eid, eid), s) for eid, s in top]

    def status(self) -> Dict[str, Any]:
        return {"episodes": len(self.episodes),
                "indexed_concepts": len(self.by_concept)}


# ----------------------------------------------------------------------
# Regression guard
# ----------------------------------------------------------------------
class RoutingEvaluator:
    """Held-out query -> intent pairs, scored through the concept graph.

    Passed to `Plasticity.consolidate` so a sleep pass that degrades routing
    gets reverted. Pairs accumulate automatically from turns where the
    classifier was confident and the tool succeeded -- so the held-out set is
    built from ASH's own uncontested successes and needs no hand labelling.
    """

    def __init__(self, net: ConceptNetwork, capacity: int = 300):
        self.net = net
        self.pairs: List[Tuple[np.ndarray, str]] = []
        self.capacity = capacity

    def observe(self, vec: np.ndarray, intent: str, confidence: float,
                tool_success: Optional[bool]):
        if intent is None or tool_success is not True or confidence < 0.6:
            return
        self.pairs.append((np.asarray(vec, dtype=np.float32), intent))
        if len(self.pairs) > self.capacity:
            self.pairs = self.pairs[-self.capacity:]

    def score(self) -> float:
        """Fraction of held-out queries the graph still routes correctly.

        Growth and prototype drift are disabled during evaluation -- scoring
        must not itself modify the thing being scored.
        """
        if len(self.pairs) < 10:
            return 1.0
        correct = 0
        for vec, intent in self.pairs:
            act = self.net.perceive(vec, allow_growth=False, learn_prototypes=False)
            scores = self.net.intent_activation(act)
            if scores and max(scores.items(), key=lambda kv: kv[1])[0] == intent:
                correct += 1
        return correct / len(self.pairs)

    def status(self) -> Dict[str, Any]:
        return {"holdout_pairs": len(self.pairs)}


# ----------------------------------------------------------------------
class ConceptSystem:
    """Complementary learning systems, assembled.

    Two learners with different rates and a transfer channel between them:

        Hippocampus   one-shot, sparse, decays in days
        ConceptNetwork  slow, statistical, permanent  (cortex)
        SleepCycle    replays the former into the latter, interleaved

    plus the neuromodulators that decide which of the two is currently in
    charge. During waking, high acetylcholine favours encoding and cortical
    plasticity is suppressed. During sleep ACh falls, hippocampal output can
    drive cortex, and consolidation happens. That switch is the reason both
    systems can coexist without corrupting each other.
    """

    def __init__(self, embedder, registry, core_memory=None, entity_memory=None,
                 intent_catalog=None, shapes=None, dim: int = 384,
                 load: bool = True, seed: bool = True):
        self.net = ConceptNetwork(dim=dim)
        self.plasticity = Plasticity(self.net)
        self.hippocampus = Hippocampus(dim=dim)
        self.neuromod = Neuromodulators()
        self.plasticity.neuromod = self.neuromod
        self.sleep_cycle = SleepCycle(self.net, self.plasticity,
                                      self.hippocampus, self.neuromod)

        self.proposer = ConceptProposer(self.net, registry, shapes)
        self.index = ConceptMemoryIndex(self.net)
        self.evaluator = RoutingEvaluator(self.net)
        self.enabled = True
        self._last_sleep = time.time()

        # Schema lookup used by consolidation to fast-track associations that
        # fit existing structure. Wired here because it needs both stores.
        self.plasticity._schema_lookup = self._schema_for_pair

        restored = self.net.load() if load else False
        if not restored and seed:
            seed_from_ash(self.net, embedder, registry=registry,
                          core_memory=core_memory, entity_memory=entity_memory,
                          intent_catalog=intent_catalog)

    # ------------------------------------------------------------------
    def _schema_for_pair(self, key: Tuple[int, int]) -> float:
        """How much existing structure supports this pair.

        A pair whose endpoints are already well connected to a common
        neighbourhood is schema-consistent and consolidates faster. This is
        what makes the tenth example in a familiar domain cheap and the first
        one in a new domain expensive -- which is the correct asymmetry, and
        the opposite of what a uniform learning rate gives you.
        """
        a, b = key
        na = {d for d, _ in self.net.neighbours(a, 8)}
        nb = {d for d, _ in self.net.neighbours(b, 8)}
        if not na or not nb:
            return 0.0
        return len(na & nb) / max(1, min(len(na), len(nb)))

    # ------------------------------------------------------------------
    def perceive(self, vec, text: str = "", learn: bool = True,
                 novelty: Optional[float] = None, surprise: float = 0.0,
                 urgency: float = 0.3, fatigue: float = 0.0) -> Activation:
        act = self.net.perceive(vec, text=text)

        self.neuromod.update(
            novelty=act.novelty if novelty is None else novelty,
            surprise=surprise, urgency=urgency, fatigue=fatigue)

        if learn and self.enabled:
            self.plasticity.potentiate(act)
        return act

    def encode_episode(self, vec, text: str = "", act: Optional[Activation] = None,
                       intent: Optional[str] = None, reward: float = 0.0):
        """One-shot write to the hippocampus.

        Deliberately separate from `perceive`: not every percept deserves an
        episode. Encoding gain comes from acetylcholine, so in a familiar
        situation the trace starts weak and will fade unless something makes
        it matter.
        """
        if not self.enabled:
            return None
        concepts = dict(act.spread) if act is not None else {}
        return self.hippocampus.store(
            vec, text=text, concepts=concepts, intent=intent, reward=reward,
            gain=self.neuromod.encoding_gain())

    def recall(self, vec, n: int = 5):
        """Episodic recall by pattern completion. Works from one exposure."""
        return self.hippocampus.recall(vec, n=n)

    def reward(self, td_error: float):
        if not self.enabled:
            return
        self.neuromod.update(td_error=td_error)
        self.plasticity.reward(self.neuromod.reward_signal())

    # ------------------------------------------------------------------
    def sleep(self) -> Dict[str, Any]:
        hours = max(0.5, (time.time() - self._last_sleep) / 3600.0)
        self._last_sleep = time.time()
        report = self.sleep_cycle.run(evaluator=self.evaluator.score,
                                      hours_since_last=hours)
        self.net.save()
        return report

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "network": self.net.status(),
            "hippocampus": self.hippocampus.status(),
            "neuromod": self.neuromod.status(),
            "plasticity": self.plasticity.status(),
            "sleep": self.sleep_cycle.status(),
            "index": self.index.status(),
            "evaluator": self.evaluator.status(),
        }
