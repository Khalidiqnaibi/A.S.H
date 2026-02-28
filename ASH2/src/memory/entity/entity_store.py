# entity_store.py

import json
from typing import Dict, List, Optional
from .entity_model import Entity


class EntityStore:
    def __init__(self, path: str = "entities.json"):
        self.path = path
        self.entities: Dict[str, Entity] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
                for eid, data in raw.items():
                    self.entities[eid] = Entity(**data)
        except FileNotFoundError:
            self.entities = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(
                {eid: vars(ent) for eid, ent in self.entities.items()},
                f,
                indent=2,
            )

    def add(self, entity: Entity):
        self.entities[entity.entity_id] = entity
        self._save()

    def update(self, entity: Entity):
        self.entities[entity.entity_id] = entity
        self._save()

    def get(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def all(self) -> List[Entity]:
        return list(self.entities.values())