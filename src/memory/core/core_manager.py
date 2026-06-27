# core_manager.py

from typing import List
from .core_store import CoreStore
from .core_model import CoreRule
from .core_index import CoreIndex


class CoreMemoryEngine:

    def __init__(self, embedder, path="core_memory.json"):
        self.store = CoreStore(path)
        self.index = CoreIndex(embedder)
        self._rebuild_index()

    def _rebuild_index(self):
        self.index.rebuild(self.store.all())

    # ----------------------------
    # Modification Methods
    # ----------------------------

    def add_rule(self, category: str, text: str, priority: int = 5, hard: bool = False):
        rule = CoreRule.create(category, text, priority, hard)
        self.store.add(rule)
        self._rebuild_index()
        return rule

    def update_rule(self, rule_id: str, new_text: str):
        rule = self.store.get(rule_id)
        if not rule:
            raise ValueError("Rule not found")

        rule.text = new_text
        self.store.update(rule)
        self._rebuild_index()

    def delete_rule(self, rule_id: str):
        self.store.delete(rule_id)
        self._rebuild_index()

    # ----------------------------
    # Retrieval
    # ----------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[CoreRule]:

        scored = self.index.search(query, top_k=top_k)

        # Re-rank: Hard rules first, then priority, then similarity
        scored.sort(
            key=lambda x: (
                not x[1].hard,        # hard=True first
                -x[1].priority,       # higher priority first
                -x[0],                # higher similarity first
            )
        )

        return [rule for _, rule in scored]

    # ----------------------------
    # Prompt Builder
    # ----------------------------

    def build_prompt_block(self, query: str) -> str:
        rules = self.retrieve(query)

        grouped = {}

        for rule in rules:
            grouped.setdefault(rule.category, []).append(rule.text)

        output = "[CORE MEMORY]\n"

        for category, texts in grouped.items():
            output += f"\n[{category.upper()}]\n"
            for t in texts:
                output += f"- {t}\n"

        return output

    # ----------------------------
    # Compatibility Methods
    # ----------------------------

    def add(self, key: str, payload: dict):
        import time
        existing = self.store.get(key)

        # Backward compatible: callers that only ever passed
        # {"text":..., "importance":...} keep getting exactly the old
        # behavior (category="constraint", hard=False, priority from
        # importance). Callers that now classify the statement (see
        # MemoryRouter._classify_core_candidate) can additionally pass
        # "category" / "hard", and optionally override "priority" directly.
        category = payload.get("category", "constraint")
        hard = bool(payload.get("hard", False))
        if "priority" in payload:
            priority = int(payload["priority"])
        else:
            priority = int(payload.get("importance", 0.5) * 10)

        if existing:
            existing.text = payload.get("text", existing.text)
            existing.category = payload.get("category", existing.category)
            existing.hard = payload.get("hard", existing.hard)
            if "priority" in payload or "importance" in payload:
                existing.priority = priority
            existing.updated_at = time.time()
            self.store.update(existing)
        else:
            rule = CoreRule(
                rule_id=key,
                category=category,
                text=payload.get("text", ""),
                priority=priority,
                hard=hard
            )
            self.store.add(rule)
        self._rebuild_index()

    def dump_all(self) -> dict:
        return {rule.rule_id: {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "text": rule.text,
            "priority": rule.priority,
            "hard": rule.hard,
            "created_at": rule.created_at,
            "updated_at": rule.updated_at
        } for rule in self.store.all()}