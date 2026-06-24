# entity_manager.py

import uuid
from .entity_model import Entity
from .entity_store import EntityStore
from .entity_resolver import EntityResolver


class EntityManager:

    def __init__(self, path="entities.json"):
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
            raise Exception("Ambiguous entity match. Manual confirmation required.")