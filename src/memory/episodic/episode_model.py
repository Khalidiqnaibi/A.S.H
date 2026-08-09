# episode_model.py

from dataclasses import dataclass, field
from typing import List, Optional
import uuid
import time


@dataclass
class Episode:
    episode_id: str
    timestamp: float
    summary: str
    event_type: str  # "interaction", "decision", "observation", etc.

    related_entities: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0–1 scale

    created_at: float = field(default_factory=time.time)

    # Pruning heuristic fields.
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    @staticmethod
    def create(summary: str, event_type: str,
               related_entities=None, importance=0.5):
        now = time.time()
        return Episode(
            episode_id=str(uuid.uuid4()),
            timestamp=now,
            summary=summary,
            event_type=event_type,
            related_entities=related_entities or [],
            importance=importance,
            last_accessed=now,
            access_count=0,
        )