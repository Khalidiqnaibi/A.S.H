# memory_router.py
import logging
import time
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


class MemoryRouter:
    """
    Route incoming text (user utterances, uploads, system events) to:
      - CoreMemory (policies, constraints, permanent facts)
      - EntityMemory (people, organizations, devices)
      - EpisodicMemory (time-based events)
    Requires instances:
      - core_mem: CoreMemory (must implement .add(key, payload) and .get(key))
      - entity_mem: EntityMemory (must implement .find_candidates(name), .create_entity(...), .update_entity(...))
      - episodic_mem: EpisodicMemory (must implement .add_episode(summary, event_type, related_entities, importance), .retrieve)
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
            "entity_link_threshold": 0.6,  # candidate score above this => auto-link
            "core_keyword_min_count": 2,   # heuristics: if core keywords appear >= this, store in core
            "save_min_importance": 0.2,
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