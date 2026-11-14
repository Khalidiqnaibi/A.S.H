# core_store.py

import logging
from typing import Dict, List, Optional
from .core_model import CoreRule
from ..paths import memory_file
from ..atomic import load_json, log_io_failure, write_json

logger = logging.getLogger("ash.memory.core")


class CoreStore:

    def __init__(self, path: Optional[str] = None):
        self.path = memory_file("core_memory.json", path)
        self.rules: Dict[str, CoreRule] = {}
        self._load()

    def _load(self):
        raw = load_json(self.path, default=None, logger=logger)
        self.rules = {}
        if not isinstance(raw, dict):
            return
        for rid, data in raw.items():
            try:
                self.rules[rid] = CoreRule(**data)
            except Exception:
                logger.warning("Skipped a core rule it could not read (%s)", rid)

    def _save(self):
        try:
            write_json(self.path, {rid: vars(rule) for rid, rule in self.rules.items()}, indent=2)
        except OSError as e:
            log_io_failure(logger, "core memory", e)

    def add(self, rule: CoreRule):
        self.rules[rule.rule_id] = rule
        self._save()

    def update(self, rule: CoreRule):
        self.rules[rule.rule_id] = rule
        self._save()

    def delete(self, rule_id: str):
        if rule_id in self.rules:
            del self.rules[rule_id]
            self._save()

    def all(self) -> List[CoreRule]:
        return list(self.rules.values())

    def get(self, rule_id: str) -> Optional[CoreRule]:
        return self.rules.get(rule_id)