# entity_store.py

import logging
from typing import Dict, List, Optional

from dataclasses import asdict
from .entity_model import Entity
from ..paths import memory_file
from ..atomic import load_json, log_io_failure, write_json

logger = logging.getLogger("ash.memory.entities")


class EntityStore:
    def __init__(self, path: Optional[str] = None):
        self.path = memory_file("entities.json", path)
        self.entities: Dict[str, Entity] = {}
        self._load()

    def _load(self):
        # An empty or half-written file (the power went mid-save) is set aside and the last good copy used, instead of
        # the whole daemon refusing to start.
        raw = load_json(self.path, default=None, logger=logger)
        self.entities = {}
        if not isinstance(raw, dict):
            return
        for eid, data in raw.items():
            try:
                self.entities[eid] = Entity(**data)
            except Exception:
                logger.warning("Skipped an entity it could not read (%s)", eid)

    def _save(self):
        try:
            write_json(self.path, {eid: asdict(ent) for eid, ent in self.entities.items()}, indent=2)
        except OSError as e:                      # disk full or locked: keep it in memory, try again on the next change
            log_io_failure(logger, "entity memory", e)

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