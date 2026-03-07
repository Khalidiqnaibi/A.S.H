# episode_index.py

import numpy as np
from typing import List
from .episode_model import Episode


class EpisodeIndex:

    def __init__(self, embedder):
        self.embedder = embedder
        self.episodes: List[Episode] = []
        self.vectors = None

    def rebuild(self, episodes: List[Episode]):
        self.episodes = episodes
        summaries = [ep.summary for ep in episodes]
        vecs = self.embedder.encode(summaries)
        self.vectors = vecs        

    def search(self, query: str, top_k=5):
        if self.vectors is None:
            return []

        q = self.embedder.embed(query)
        sims = self._cosine(q, self.vectors)
        idxs = np.argsort(-sims)[:top_k]

        return [(sims[i], self.episodes[i]) for i in idxs]

    def _cosine(self, q, M):
        q = q / np.linalg.norm(q)
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return M @ q