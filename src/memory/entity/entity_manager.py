# entity_manager.py

import uuid
from .entity_model import Entity
from .entity_store import EntityStore
from .entity_resolver import EntityResolver


class EntityManager:

    def __init__(self, path=None):
        self.store = EntityStore(path)
        self.resolver = EntityResolver(self.store)

    def ingest(self, entity_data: dict) -> Entity:

        incoming = Entity(
            entity_id=str(uuid.uuid4()),
            entity_type=entity_data["entity_type"],
            primary_identifiers=entity_data.get("primary_identifiers", {}),
            secondary_identifiers=entity_data.get("secondary_identifiers", {}),
            attributes=entity_data.get("attributes", {}),
        )

        decision, existing = self.resolver.resolve(incoming)

        if decision == "update":
            existing.update(incoming.attributes)
            self.store.update(existing)
            return existing

        if decision == "create":
            self.store.add(incoming)
            return incoming

        if decision == "ambiguous":
            # Fallback: create as new entity instead of crashing
            import logging
            logging.getLogger("ash.entity").warning(
                "Ambiguous entity match for '%s' — creating new entry.",
                entity_data.get("primary_identifiers", {}).get("name", "unknown")
            )
            self.store.add(incoming)
            return incoming

    def find_matching_entities(self, mentions: list) -> list:
        matched = []

        def canon(s: str) -> str:
            # case, punctuation and spacing do not make another entity: "a.s.h" / "ash", "backend & genai developer" / "backend/genai developer"
            import re as _re
            return _re.sub(r"[^a-z0-9]", "", str(s or "").lower())
        lower_mentions = {canon(m) for m in mentions} - {""}
        for entity in self.store.all():
            name = canon(entity.primary_identifiers.get("name", ""))
            canonical = canon(entity.attributes.get("canonical_name", ""))
            if (name and name in lower_mentions) or (canonical and canonical in lower_mentions):
                matched.append(entity)
        return matched