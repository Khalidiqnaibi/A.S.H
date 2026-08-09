# episodic_manager.py

import logging
import time
from typing import List
from .episode_model import Episode
from .episode_store import EpisodeStore
from .episode_index import EpisodeIndex
from .episode_ranker import EpisodeRanker

logger = logging.getLogger("ash.memory.episodic")


class EpisodicMemory:

    def __init__(self, embedder, path="episodic_memory.jsonl"):
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
        results = [ep for _, ep in ranked[:top_k]]

        # Mark these as used, for the pruning heuristic in prune()
        # below. Mutated in place; store.mark_accessed() just keeps
        # the store's dict pointing at the same (already-updated)
        # objects -- see its docstring for why this doesn't hit disk
        # on every retrieval.
        now = time.time()
        for ep in results:
            ep.last_accessed = now
            ep.access_count += 1
        self.store.mark_accessed(results)

        return results

    # ---------------------
    # Prune (maintenance-phase only)
    # ---------------------

    def prune(self, min_age_seconds: float = 7 * 86400) -> int:
        """
        Drop episodes that are both old (created >= min_age_seconds
        ago) AND never retrieved (access_count == 0). An episode
        that's old but has been retrieved even once survives --
        access_count, not recency, is what earns a spot; a heavily-
        referenced old memory should outlive a fresh one nobody's
        touched.

        Meant to run during ASH's maintenance/"sleep mode" phase
        (charging/WiFi/idle -- see the maintenance scheduler this
        wires into), not on the hot conversational path: it does a
        full rewrite of the store (store._save_all()) and rebuilds
        the embedding index, both of which are too expensive to run
        per-turn.

        Returns the number of episodes pruned.
        """
        now = time.time()
        survivors = {}
        pruned_count = 0

        for eid, ep in self.store.episodes.items():
            age = now - ep.timestamp
            never_used = ep.access_count == 0

            if age >= min_age_seconds and never_used:
                pruned_count += 1
                continue  # drop it
            survivors[eid] = ep

        self.store.episodes = survivors
        self.store._save_all()  # full rewrite: removing rows requires it,
                                 # and this is also where any in-RAM-only
                                 # access_count/last_accessed bumps from
                                 # mark_accessed() finally hit disk.
        self._rebuild_index()   # index must be rebuilt after pruning

        if pruned_count:
            logger.info("Pruned %d unused episode(s) older than %.0fs", pruned_count, min_age_seconds)

        return pruned_count