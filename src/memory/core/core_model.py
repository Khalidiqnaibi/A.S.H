# core_model.py

from dataclasses import dataclass, field
from typing import Optional
import time
import uuid


@dataclass
class CoreRule:
    rule_id: str
    category: str  # "identity", "goal", "constraint", "standard"
    text: str
    priority: int = 5          # higher = more important
    hard: bool = False         # cannot be overridden
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def create(category: str, text: str, priority: int = 5, hard: bool = False):
        return CoreRule(
            rule_id=str(uuid.uuid4()),
            category=category,
            text=text,
            priority=priority,
            hard=hard,
        )