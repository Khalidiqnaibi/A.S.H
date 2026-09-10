# episode_index.py

import logging

import numpy as np
from typing import List
from .episode_model import Episode

logger = logging.getLogger("ash.memory.episode_index")


class EpisodeIndex:

    def __init__(self, embedder):
        self.embedder = embedder
        self.episodes: List[Episode] = []
        self.vectors = None
        if embedder is None:
            logger.warning(
                "EpisodeIndex: no embedding model available -- semantic search over "
                "episodic memory is disabled for this run (falls back to exact/keyword "
                "matching upstream in MemoryRouter). Check earlier logs for why the "
                "embedder failed to load (e.g. no internet on first run, model not "
                "cached locally)."
            )

    def rebuild(self, episodes: List[Episode]):
        self.episodes = episodes
        
        # Guard clause: If there are no episodes, do not invoke the embedder
        if not episodes or self.embedder is None:
            self.vectors = None
            return

        summaries = [ep.summary for ep in episodes]
        vecs = self.embedder.encode(summaries)
        
        # Convert to numpy array if sentence-transformers returned a list or torch tensor
        self.vectors = np.array(vecs)   

    def append_single(self, episode):
        self.episodes.append(episode)
        if self.embedder is None:
            return
        # Encode only the *one* new sentence
        new_vec = self.embedder.encode([episode.summary]) 
        
        if self.vectors is None:
            self.vectors = new_vec
        else:
            # Concatenate it to the bottom of your existing matrix
            self.vectors = np.vstack([self.vectors, new_vec]) 

    def search(self, query: str, top_k=5):
        if self.vectors is None or self.embedder is None:
            return []

        q = self.embedder.encode(query)
        sims = self._cosine(q, self.vectors)
        idxs = np.argsort(-sims)[:top_k]

        return [(sims[i], self.episodes[i]) for i in idxs]

    def _cosine(self, q, M):
        # Stabilize denominator with a small epsilon threshold
        q_norm = np.linalg.norm(q)
        q = q / (q_norm if q_norm > 0 else 1e-9)
        
        m_norms = np.linalg.norm(M, axis=1, keepdims=True)
        m_norms = np.where(m_norms == 0, 1e-9, m_norms)
        M = M / m_norms
        
        return M @ q