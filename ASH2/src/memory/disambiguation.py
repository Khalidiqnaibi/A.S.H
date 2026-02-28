# disambiguation.py
# import sys
import logging
from typing import List, Dict, Any, Optional
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    ST_AVAILABLE = False

# import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger("ash.disamb")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys as _sys
    logger.addHandler(logging.StreamHandler(stream=_sys.stderr))


class Disambiguator:
    """
    Resolve entity mention to a canonical entity id in EntityMemory.
    If embeddings are available (sentence-transformers), uses them for semantic similarity.
    Fallback: string similarity ratio.
    """

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = embed_model_name
        self.model = None
        if ST_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info("Loaded sentence-transformers model: %s", self.model_name)
            except Exception as e:
                logger.warning("Failed to load embedder: %s", e)
                self.model = None

    def _string_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def most_likely(self, mention: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        candidates: list of {"id": id, "name": name, "aliases": [...], "metadata": ...}
        returns candidates with score sorted desc.
        """
        if not candidates:
            return []

        if self.model:
            try:
                qv = self.model.encode([mention], convert_to_numpy=True, normalize_embeddings=True)[0]
                names = [c.get("name", "") for c in candidates]
                texts = [c.get("name", "") + " " + " ".join(c.get("aliases", [])) for c in candidates]
                vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                sims = (vecs @ qv).tolist()
                ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
                return [{"candidate": r[0], "score": float(r[1])} for r in ranked[:top_k]]
            except Exception as e:
                logger.warning("Embed disambiguation failed: %s", e)
                # fallback to string
        # fallback string similarity
        scored = []
        for c in candidates:
            score = max(self._string_similarity(mention, c.get("name", "")),
                        max((self._string_similarity(mention, a) for a in c.get("aliases", [])), default=0.0))
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"candidate": s[0], "score": float(s[1])} for s in scored[:top_k]]