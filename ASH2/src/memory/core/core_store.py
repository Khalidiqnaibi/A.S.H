# core_store.py

import json
from typing import Dict, List, Optional
from .core_model import CoreRule


class CoreStore:

    def __init__(self, path: str = "core_memory.json"):
        self.path = path
        self.rules: Dict[str, CoreRule] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
                for rid, data in raw.items():
                    self.rules[rid] = CoreRule(**data)
        except FileNotFoundError:
            self.rules = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(
                {rid: vars(rule) for rid, rule in self.rules.items()},
                f,
                indent=2,
            )

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