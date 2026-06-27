# memory_router.py
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .ner import NERExtractor
from .disambiguation import Disambiguator

from .core.core_manager import CoreMemoryEngine
from .entity.entity_manager import EntityManager
from .episodic.episodic_manager import EpisodicMemory

logger = logging.getLogger("ash.router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys as _sys
    logger.addHandler(logging.StreamHandler(stream=_sys.stderr))

# ----------------------------------------------------------------------
# Lightweight relevance scoring helpers.
#
# Neither CoreMemoryEngine.retrieve() nor EpisodicMemory.retrieve() hands
# back the similarity score it computed internally -- they each do their own
# embedding search, re-rank (hard/priority for core; the EpisodeRanker for
# episodic), and return bare CoreRule / Episode objects with the score
# already thrown away. So once those objects reach the router, the only
# relevance signal we can compute ourselves is lexical overlap against the
# query. `_extract_native_score` is kept as a defensive no-op in case a
# future version of either index starts attaching a score to what it
# returns -- today it will always return None and we fall back to lexical.
# ----------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "for", "and", "or", "but", "with", "this", "that",
    "it", "i", "you", "your", "my", "me", "we", "us", "do", "does", "did",
    "have", "has", "had", "what", "when", "where", "who", "why", "how",
    "can", "could", "would", "should", "will", "just", "so", "if", "as",
    "by", "from", "about", "into", "than", "then", "there", "here", "up",
    "out", "not", "no", "yes", "ok", "okay", "im", "youre",
    "today", "tomorrow", "yesterday", "now", "currently", "still", "right",
}


class MemoryRouter:
    """
    Route incoming text (user utterances, uploads, system events) to:
      - CoreMemory (policies, constraints, permanent facts)
      - EntityMemory (people, organizations, devices)
      - EpisodicMemory (time-based events)

    Expected collaborator interfaces (matching core_manager.py / entity_manager.py /
    episodic_manager.py):
      - core_mem: CoreMemoryEngine — .add(key, payload), .retrieve(query, top_k) -> List[CoreRule]
      - entity_mem: EntityManager — .ingest(entity_data), .find_matching_entities(mentions) -> List[Entity],
        and a `.store` with `.all() -> List[Entity]`
      - episodic_mem: EpisodicMemory — .add_episode(summary, event_type, related_entities, importance),
        .retrieve(query, top_k) -> List[Episode]
      - ner: NERExtractor — .extract(text) -> List[{"text", "label", ...}]
      - disamb: Disambiguator — .most_likely(mention, candidates, top_k) -> List[{"candidate", "score"}]
    """

    def __init__(
        self,
        core_mem : CoreMemoryEngine,
        entity_mem : EntityManager,
        episodic_mem : EpisodicMemory,
        ner: Optional[NERExtractor] = None,
        disamb: Optional[Disambiguator] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.core = core_mem
        self.entity = entity_mem
        self.episodic = episodic_mem
        self.ner = ner or NERExtractor()
        self.disambiguator = disamb or Disambiguator()
        self.config = {
            "entity_link_threshold": 0.6,  # candidate score above this => auto-link (write path)
            "core_keyword_min_count": 2,   # heuristics: if core keywords appear >= this, store in core
            "save_min_importance": 0.2,

            # --- retrieval tuning ---
            # How big a candidate pool to pull from core/episodic before we
            # filter it down. Bigger multiplier = more recall to choose from.
            "retrieval_fetch_multiplier": 3,
            # Minimum blended relevance score (0..1) a core/episodic memory
            # needs to be kept, unless it's flagged "always keep" below.
            "retrieval_relevance_threshold": 0.12,
            # Core rules at/above this priority (or marked `hard`) are always
            # surfaced, since they govern behavior regardless of topic.
            "core_always_include_priority": 8,
            # Episodes at/above this importance are always surfaced.
            "episodic_always_include_importance": 0.85,
            # Minimum Disambiguator confidence to treat a fuzzy entity match
            # as relevant for *retrieval*. Lower than entity_link_threshold,
            # since showing a borderline-relevant entity for grounding is
            # cheap, while auto-merging two entities at write time is not.
            "entity_retrieval_threshold": 0.45,

            **(config or {})
        }
        # core keywords: words that usually indicate core facts/rules (customize)
        self.core_keywords = set(["policy", "constraint", "rule", "standard", "limit", "goal", "mission", "aim"])

    # ----------------------
    # Public entry
    # ----------------------
    def route_utterance(self, text: str, source: str = "chat", importance: float = 0.5, actor: str = None) -> Dict[str, Any]:
        """
        Main call to route a text. Returns a dict describing action taken.
        Steps:
          1) run NER
          2) decide if it's core (policy/goal)
          3) if entities are detected: generically map them and pass to self.entity.ingest()
          4) append a clean interaction timeline window to episodic memory
        """
        start = time.time()
        ents = self.ner.extract(text)
        logger.info("NER extracted %d entities", len(ents))

        # Quick heuristic: core detection
        low_text = text.lower()
        core_score = sum(1 for kw in self.core_keywords if kw in low_text)
        is_core = core_score >= self.config["core_keyword_min_count"]

        result = {"routed_as": None, "details": {}, "time": 0.0}

        # If core -> add to core memory (idempotent)
        if is_core or importance >= 0.9:
            key = self._core_key_from_text(text)
            payload = {
                "text": text,
                "summary": text[:512],
                "importance": importance,
                "source": source,
                "actor": actor
            }
            try:
                self.core.add(key, payload)
                result["routed_as"] = "core"
                result["details"] = {"key": key}
                logger.info("Stored CORE memory key=%s", key)
                result["time"] = time.time() - start
                return result
            except Exception as e:
                logger.exception("Core memory add failed: %s", e)

        # Generically process ANY entity type via unified entity ingestion architecture
        linked_entities = []
        for ent in ents:
            mention = ent["text"]
            # Map labels generically (spaCy PERSON -> person, ORG -> organization, GPE -> location, etc.)
            raw_label = ent.get("label", "unknown").lower()

            # Formulate the payload data payload strictly adhering to Entity model constraints
            entity_payload = {
                "entity_type": raw_label,
                "primary_identifiers": {
                    "name": mention.lower().strip()
                },
                "attributes": {
                    "canonical_name": mention,
                    "last_seen_context": text,
                    "source": source,
                    "updated_at": time.time()
                }
            }

            try:
                # Let your specialized manager execute resolution, matching, or creation safely
                resolved_ent = self.entity.ingest(entity_payload)
                linked_entities.append(resolved_ent.entity_id)
                logger.info("Ingested entity: '%s' [%s] -> ID: %s", mention, raw_label, resolved_ent.entity_id)
            except Exception as ex:
                logger.warning("Entity ingestion workflow failed for mention '%s': %s", mention, ex)

        # Always store an episodic record for the event
        try:
            summary = self._summarize_for_episode(text, ents, actor)
            ep = self.episodic.add_episode(
                summary=summary,
                event_type="interaction",
                related_entities=linked_entities,
                importance=importance
            )
            result["routed_as"] = "episodic"
            result["details"] = {"episode_id": ep.episode_id, "linked_entities": linked_entities}
            logger.info("Added episode %s (linked %d entities)", ep.episode_id, len(linked_entities))
        except Exception as e:
            logger.exception("Episodic add failed: %s", e)
            result["routed_as"] = "failed"

        result["time"] = time.time() - start
        return result

    # ----------------------
    # Helpers
    # ----------------------
    def _core_key_from_text(self, text: str) -> str:
        # naive key: take first 6 words normalized
        k = "_".join(text.lower().strip().split()[:6])
        return f"core_{k}"

    def _summarize_for_episode(self, text: str, ents: List[Dict[str, Any]], actor: Optional[str]):
        # very small summarizer: mention entities and a short truncated text
        ent_texts = [e["text"] for e in ents]
        ent_block = ", ".join(ent_texts) if ent_texts else ""
        actor_block = f"{actor}: " if actor else ""
        s = f"{actor_block}{text[:400]}"
        if ent_block:
            s += f" [mentions: {ent_block}]"
        return s

    # ----------------------
    # Relevance scoring (layer-agnostic)
    # ----------------------
    def _tokenize(self, text: str) -> set:
        if not text:
            return set()
        return {
            w for w in _TOKEN_RE.findall(text.lower())
            if w not in _STOPWORDS and len(w) > 1
        }

    def _lexical_relevance(self, query_tokens: set, text: str) -> float:
        """Cheap, dependency-free relevance score in [0, 1].

        Blends how much of the *query* the memory covers (recall) with how
        much of the *memory* is actually about the query (precision), so a
        short, on-topic memory scores higher than a long memory that merely
        happens to share one word with the query.
        """
        if not query_tokens:
            return 0.0
        text_tokens = self._tokenize(text)
        if not text_tokens:
            return 0.0
        overlap = query_tokens & text_tokens
        if not overlap:
            return 0.0
        recall = len(overlap) / len(query_tokens)
        precision = len(overlap) / len(text_tokens)
        return (0.7 * recall) + (0.3 * precision)

    def _extract_native_score(self, obj: Any) -> Optional[float]:
        """See module docstring: today CoreRule/Episode never carry a score,
        so this is a forward-compatible no-op. Kept narrow and defensive so
        it can never raise."""
        for attr in ("score", "similarity", "relevance", "sim_score"):
            val = getattr(obj, attr, None)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    def _score_item(self, query_tokens: set, text: str, native_score: Optional[float] = None) -> float:
        lexical = self._lexical_relevance(query_tokens, text)
        if native_score is None:
            return lexical
        native = max(0.0, min(1.0, native_score))
        return (0.5 * native) + (0.5 * lexical)

    def _format_timestamp(self, ts: Any) -> str:
        try:
            return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

    def _all_entities(self) -> List[Any]:
        """Reach into EntityManager.store.all() the same way
        find_matching_entities() does internally -- this is the only way to
        enumerate entities for fuzzy matching / id lookup, since EntityManager
        doesn't expose a search-by-fuzzy-name or get-by-id method itself."""
        store = getattr(self.entity, "store", None)
        all_fn = getattr(store, "all", None) if store is not None else None
        if not callable(all_fn):
            return []
        try:
            return list(all_fn())
        except Exception as e:
            logger.warning("Failed to list entities from entity store: %s", e)
            return []

    def _entity_to_candidate(self, ent: Any) -> Dict[str, Any]:
        attrs = getattr(ent, "attributes", {}) or {}
        primary = getattr(ent, "primary_identifiers", {}) or {}
        canonical = attrs.get("canonical_name") or primary.get("name", "")
        aliases = [v for v in primary.values() if v]
        return {
            "id": getattr(ent, "entity_id", None),
            "name": canonical,
            "aliases": aliases,
            "metadata": ent,
        }

    # ----------------------
    # Per-layer retrieval builders
    # ----------------------
    def _build_core_block(self, query: str, query_tokens: set, top_k: int, min_relevance: float) -> str:
        fetch_k = max(top_k * self.config["retrieval_fetch_multiplier"], top_k)
        try:
            # CoreMemoryEngine.retrieve() already does its own embedding
            # search + hard/priority re-rank; asking for more than we'll
            # ultimately keep just widens the semantic candidate pool.
            core_rules = self.core.retrieve(query, top_k=fetch_k) or []
        except Exception as e:
            logger.warning("Failed to retrieve core memory: %s", e)
            return ""

        always_priority = self.config["core_always_include_priority"]
        scored = []
        for rule in core_rules:
            text = getattr(rule, "text", "") or ""
            score = self._score_item(query_tokens, text, self._extract_native_score(rule))
            priority = getattr(rule, "priority", 0) or 0
            is_hard = bool(getattr(rule, "hard", False))
            always_keep = is_hard or priority >= always_priority
            if always_keep or score >= min_relevance:
                scored.append((always_keep, priority, score, rule))

        if not scored:
            return ""

        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        scored = scored[:top_k]

        grouped: Dict[str, List[str]] = {}
        for _, _, _, rule in scored:
            category = getattr(rule, "category", "general") or "general"
            grouped.setdefault(category, []).append(getattr(rule, "text", ""))

        core_block = "[CORE MEMORY]\n"
        for category, texts in grouped.items():
            core_block += f"\n[{category.upper()}]\n"
            for t in texts:
                core_block += f"- {t}\n"
        return core_block

    def _build_episodic_block(self, query: str, query_tokens: set, top_k: int, min_relevance: float):
        fetch_k = max(top_k * self.config["retrieval_fetch_multiplier"], top_k)
        try:
            # EpisodicMemory.retrieve() internally over-fetches from its own
            # index (top_k * 2) and runs it through EpisodeRanker already;
            # bumping the top_k we pass in widens that candidate pool too.
            recent_episodes = self.episodic.retrieve(query, top_k=fetch_k) or []
        except Exception as e:
            logger.warning("Failed to retrieve episodic memory: %s", e)
            return "", []

        always_importance = self.config["episodic_always_include_importance"]
        scored = []
        for ep in recent_episodes:
            summary = getattr(ep, "summary", "") or ""
            score = self._score_item(query_tokens, summary, self._extract_native_score(ep))
            importance = getattr(ep, "importance", 0.5) or 0.5
            always_keep = importance >= always_importance
            if always_keep or score >= min_relevance:
                scored.append((always_keep, score, importance, ep))

        if not scored:
            return "", []

        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        scored = scored[:top_k]

        lines = []
        kept_episodes = []
        for _, _, _, ep in scored:
            summary = getattr(ep, "summary", "")
            when = self._format_timestamp(getattr(ep, "timestamp", None))
            prefix = f"[{when}] " if when else ""
            lines.append(f"- {prefix}{summary}")
            kept_episodes.append(ep)

        episodic_block = "[RELEVANT EPISODES]\n" + "\n".join(lines)
        return episodic_block, kept_episodes

    def _build_entity_block(self, query: str, query_tokens: set, top_k: int, min_relevance: float, linked_episodes: List[Any]) -> str:
        try:
            ents = self.ner.extract(query)
        except Exception as e:
            logger.warning("NER extraction failed during entity retrieval: %s", e)
            ents = []
        mentions = [e["text"] for e in ents]

        all_entities = self._all_entities()
        entities_by_id = {getattr(e, "entity_id", None): e for e in all_entities}

        candidates: Dict[Any, Any] = {}
        match_scores: Dict[Any, float] = {}  # entity_id -> best confidence found for it

        if mentions:
            # 1) Exact name / canonical_name matches -- guaranteed relevant,
            # the user (or the assistant) named them directly.
            try:
                exact = self.entity.find_matching_entities(mentions) or []
            except Exception as e:
                logger.warning("find_matching_entities failed: %s", e)
                exact = []
            for ent in exact:
                eid = getattr(ent, "entity_id", None)
                if eid is not None:
                    candidates[eid] = ent
                    match_scores[eid] = 1.0

            # 2) Fuzzy / semantic matches via the Disambiguator, for
            # nicknames, typos, or partial mentions that exact matching
            # would otherwise miss entirely.
            if all_entities:
                entity_candidates = [self._entity_to_candidate(e) for e in all_entities]
                threshold = self.config["entity_retrieval_threshold"]
                for mention in mentions:
                    try:
                        ranked = self.disambiguator.most_likely(mention, entity_candidates, top_k=top_k)
                    except Exception as e:
                        logger.debug("Disambiguator failed for mention '%s': %s", mention, e)
                        continue
                    for r in ranked:
                        score = r.get("score", 0.0) or 0.0
                        if score < threshold:
                            continue
                        cand = r.get("candidate", {}) or {}
                        eid = cand.get("id")
                        ent = cand.get("metadata")
                        if eid is None or ent is None:
                            continue
                        candidates[eid] = ent
                        match_scores[eid] = max(match_scores.get(eid, 0.0), score)

        # 3) Cross-layer recall: entities tied to episodes we already deemed
        # relevant, so grounding surfaces even when the query doesn't name
        # the entity directly (e.g. "how's that going?").
        for ep in linked_episodes:
            for eid in (getattr(ep, "related_entities", None) or []):
                if eid in candidates:
                    continue
                ent = entities_by_id.get(eid)
                if ent is not None:
                    candidates[eid] = ent
                    match_scores[eid] = max(match_scores.get(eid, 0.0), 0.5)  # linked, not named

        if not candidates:
            return ""

        scored = []
        for eid, ent in candidates.items():
            attrs = getattr(ent, "attributes", {}) or {}
            canonical = attrs.get("canonical_name") or getattr(ent, "primary_identifiers", {}).get("name", "Unknown")
            context = attrs.get("last_seen_context", "") or ""
            lexical = self._lexical_relevance(query_tokens, f"{canonical} {context}")
            match_score = match_scores.get(eid, 0.0)
            # Either an explicit name/fuzzy match or lexical overlap is
            # enough to justify keeping the entity for grounding.
            combined = max(match_score, lexical)
            if match_score >= 0.99 or combined >= min_relevance:
                scored.append((match_score, combined, ent))

        if not scored:
            return ""

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        scored = scored[:top_k]

        entity_block = "[RELEVANT ENTITIES]\n"
        for _, _, entity in scored:
            attrs = getattr(entity, "attributes", {}) or {}
            canonical = attrs.get("canonical_name", getattr(entity, "primary_identifiers", {}).get("name", "Unknown"))
            ent_type = getattr(entity, "entity_type", "unknown")
            context = attrs.get("last_seen_context", "")
            context_snippet = (context[:200] + "...") if len(context) > 200 else context
            entity_block += f"- Entity: {canonical} ({ent_type})\n"
            if context_snippet:
                entity_block += f"  Last Context: {context_snippet}\n"

        return entity_block

    def retrieve_context(
        self,
        query: str,
        top_k_episodes: int = 8,
        top_k_core: int = 8,
        top_k_entities: int = 8,
        min_relevance: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Query all three stages of memory to construct a unified context for the LLM.

        Compared to a naive "ask each store for its top few" approach, this:
          1) Pulls a wider candidate pool from core/episodic (retrieval_fetch_multiplier),
             since both CoreMemoryEngine.retrieve and EpisodicMemory.retrieve discard
             their internal similarity scores before returning -- a bigger top_k is the
             only way to widen what we get to choose from.
          2) Re-scores every core/episodic candidate against the query with a shared
             lexical relevance function, so only genuinely on-topic memories make it
             into the final context.
          3) Always keeps memories that matter regardless of topic overlap: hard/
             high-priority core rules, very high-importance episodes, and entities
             explicitly named (exactly or via the Disambiguator) in the query.
          4) Uses the Disambiguator for fuzzy/semantic entity matching (nicknames,
             typos, partial names) instead of relying solely on EntityManager's
             exact-string find_matching_entities, and cross-links episodic ->
             entity so grounding for an entity can surface even when it isn't
             named in the current query but is tied to a relevant episode.
        """
        min_relevance = self.config["retrieval_relevance_threshold"] if min_relevance is None else min_relevance
        query_tokens = self._tokenize(query)

        core_block = self._build_core_block(query, query_tokens, top_k_core, min_relevance)
        episodic_block, kept_episodes = self._build_episodic_block(query, query_tokens, top_k_episodes, min_relevance)
        entity_block = self._build_entity_block(query, query_tokens, top_k_entities, min_relevance, kept_episodes)

        return {
            "core": core_block,
            "episodic": episodic_block,
            "entity": entity_block
        }