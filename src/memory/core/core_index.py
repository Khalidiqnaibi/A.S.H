# core_index.py

import logging

import numpy as np
from typing import List
from .core_model import CoreRule

logger = logging.getLogger("ash.memory.core_index")


class CoreIndex:

    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = []
        self.rules = []
        if embedder is None:
            logger.warning(
                "CoreIndex: no embedding model available -- semantic search over "
                "core memory is disabled for this run (falls back to exact/keyword "
                "matching upstream in MemoryRouter). Check earlier logs for why the "
                "embedder failed to load (e.g. no internet on first run, model not "
                "cached locally)."
            )

    def rebuild(self, rules: List[CoreRule]):
        self.rules = rules
        self.vectors = []

        if self.embedder is None:
            return

        for rule in rules:
            vec = self.embedder.encode(rule.text)
            self.vectors.append(vec)

        if self.vectors:
            self.vectors = np.vstack(self.vectors)

    def search(self, query: str, top_k: int = 5):
        if not self.rules or self.embedder is None:
            return []

        q_vec = self.embedder.encode(query)

        sims = self._cosine_similarity(q_vec, self.vectors)
        ranked = np.argsort(-sims)

        results = []
        for idx in ranked[:top_k]:
            results.append((sims[idx], self.rules[idx]))

        return results

    def _cosine_similarity(self, q, matrix):
        q = q / np.linalg.norm(q)
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix @ q