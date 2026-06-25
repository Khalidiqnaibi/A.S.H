# episodic_manager.py

from typing import List
from .episode_model import Episode
from .episode_store import EpisodeStore
from .episode_index import EpisodeIndex
from .episode_ranker import EpisodeRanker


class EpisodicMemory:

    def __init__(self, embedder, path="episodic_memory.json"):
        self.store = EpisodeStore(path)
        self.index = EpisodeIndex(embedder)
        self.ranker = EpisodeRanker()
        self._rebuild_index()

    def _rebuild_index(self):
        self.index.rebuild(self.store.all())

    # ---------------------
    # Add Episode
    # ---------------------

    def add_episode(self, summary, event_type, related_entities=None, importance=0.5):
        ep = Episode.create(
            summary=summary,
            event_type=event_type,
            related_entities=related_entities,
            importance=importance
        )
        self.store.add(ep)
        
        # Fast path: incremental update in RAM instead of full file re-encoding!
        self.index.append_single(ep) 
        return ep

    # ---------------------
    # Retrieve
    # ---------------------

    def retrieve(self, query, top_k=5) -> List[Episode]:

        scored = self.index.search(query, top_k=top_k * 2)
        ranked = self.ranker.rank(scored)
        return [ep for _, ep in ranked[:top_k]]