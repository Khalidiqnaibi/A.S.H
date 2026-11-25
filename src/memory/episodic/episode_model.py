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
    # Ids of the concepts that were most active when this was laid down (src/concepts), so recall can follow what the
    # moment was about and not only its words. Concept ids are never reused. Empty for older episodes.
    concepts: List[int] = field(default_factory=list)
    # How the person sounded when they said it: the sentiment reading, -1 (bad) .. +1 (good), None when it was not read (older episodes,
    # ASH's own lines). So "how have I been lately" is a fact of memory, not a guess.
    mood: Optional[float] = None

    created_at: float = field(default_factory=time.time)

    # Pruning heuristic fields.
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    @staticmethod
    def create(summary: str, event_type: str,
               related_entities=None, importance=0.5, concepts=None, mood=None):
        now = time.time()
        return Episode(
            episode_id=str(uuid.uuid4()),
            timestamp=now,
            summary=summary,
            event_type=event_type,
            related_entities=related_entities or [],
            importance=importance,
            concepts=[int(c) for c in (concepts or [])][:8],
            mood=(round(max(-1.0, min(1.0, float(mood))), 3) if mood is not None else None),
            last_accessed=now,
            access_count=0,
        )