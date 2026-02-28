# core_index.py

import numpy as np
from typing import List
from .core_model import CoreRule


class CoreIndex:

    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = []
        self.rules = []

    def rebuild(self, rules: List[CoreRule]):
        self.rules = rules
        self.vectors = []

        for rule in rules:
            vec = self.embedder.embed(rule.text)
            self.vectors.append(vec)

        if self.vectors:
            self.vectors = np.vstack(self.vectors)

    def search(self, query: str, top_k: int = 5):
        if not self.rules:
            return []

        q_vec = self.embedder.embed(query)

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