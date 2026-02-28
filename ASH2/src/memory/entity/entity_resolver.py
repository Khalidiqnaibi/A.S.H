# entity_resolver.py

from typing import Optional, Tuple
from .entity_model import Entity
from .entity_store import EntityStore


class EntityResolver:

    def __init__(self, store: EntityStore):
        self.store = store

    def resolve(self, incoming: Entity) -> Tuple[str, Optional[Entity]]:
        """
        Returns:
            ("update", existing_entity)
            ("create", None)
            ("ambiguous", None)
        """

        candidates = []

        for entity in self.store.all():
            score = self._score_match(entity, incoming)
            if score > 0.5:
                candidates.append((score, entity))

        if not candidates:
            return "create", None

        candidates.sort(reverse=True, key=lambda x: x[0])

        if len(candidates) == 1 or candidates[0][0] > 0.8:
            return "update", candidates[0][1]

        return "ambiguous", None

    def _score_match(self, existing: Entity, incoming: Entity) -> float:
        score = 0.0

        # Primary identifier match (strong)
        for key, value in incoming.primary_identifiers.items():
            if value and existing.primary_identifiers.get(key) == value:
                score += 0.6

        # Secondary identifier match
        for key, value in incoming.secondary_identifiers.items():
            if value and existing.secondary_identifiers.get(key) == value:
                score += 0.2

        # Name fallback (weak)
        name1 = existing.attributes.get("full_name")
        name2 = incoming.attributes.get("full_name")

        if name1 and name2 and name1.lower() == name2.lower():
            score += 0.2

        return score