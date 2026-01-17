# ASH2/tools/classification.py
"""
Semantic intent & command router + sentiment tool.

Requirements:
  pip install sentence-transformers transformers
Configure with ASH_AI_BASE env var (defaults to ./ash_ai folder).
"""
from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import json
import pickle
import logging
import dotenv

import numpy as np
from langchain.tools import tool

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    print("[CLASSIFY] sentence-transformers not available:", e, file=sys.stderr, flush=True)

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    print("[CLASSIFY] transformers pipeline not available:", e, file=sys.stderr, flush=True)

# config
dotenv.load_dotenv()

BASE_PATH = os.environ.get("ASH_AI_BASE", os.path.join(os.getcwd(), r"C:\Users\khaaf\Documents\GitHub\A.S.H\ASH2\data"))
INTENTS_FILE = os.path.join(BASE_PATH, "intents.json")
INTENT_EMB_FNAME = os.path.join(BASE_PATH, "embeddings_intents.pkl")

EMBEDDING_MODEL_NAME = os.environ.get("ASH_EMBED_MODEL", "all-MiniLM-L6-v2")
SENTIMENT_MODEL_NAME = os.environ.get("ASH_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

INTENT_THRESHOLD = 0.1

logger = logging.getLogger("ash.classify")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.warning("JSON file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def _intent_description_from_block(block: Dict[str, Any]) -> str:
    if not isinstance(block, dict):
        return ""
    if "description" in block and block["description"]:
        return str(block["description"])
    patterns = block.get("patterns", [])
    sample = " ; ".join(patterns[:6])
    return f"Intent `{block.get('tag','?')}` — example phrases: {sample}"


class EmbeddingCatalog:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.labels = []
        self.descriptions = []
        self.embeddings = None
        self._source_mtime = None
        self._load_model()

    def _load_model(self):
        if SentenceTransformer is None:
            logger.error("SentenceTransformer not available. Install sentence-transformers.")
            return
        try:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            self.model = None

    def build_from_json_blocks(self, blocks: List[Dict[str, Any]]):
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        labels = []
        descs = []
        for b in blocks:
            tag = b.get("tag", "").strip()
            if not tag:
                continue
            labels.append(tag)
            desc = _intent_description_from_block(b)
            descs.append(desc)
        self.labels = labels
        self.descriptions = descs
        emb = self.model.encode(descs, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb
        logger.info("Built embeddings for %d labels", len(labels))

    def save_cache(self, fname: str):
        try:
            with open(fname, "wb") as fh:
                pickle.dump({
                    "labels": self.labels,
                    "descriptions": self.descriptions,
                    "embeddings": self.embeddings
                }, fh)
            logger.info("Saved embedding cache to %s", fname)
        except Exception:
            logger.exception("Failed to save embedding cache")

    def load_cache(self, fname: str) -> bool:
        if not os.path.exists(fname):
            return False
        try:
            with open(fname, "rb") as fh:
                data = pickle.load(fh)
                self.labels = data.get("labels", [])
                self.descriptions = data.get("descriptions", [])
                self.embeddings = data.get("embeddings", None)
            logger.info("Loaded embedding cache from %s", fname)
            return True
        except Exception:
            logger.exception("Failed to load embedding cache")
            return False

    def most_similar(self, text: str, top_k: int = 3):
        if self.model is None or self.embeddings is None or len(self.embeddings) == 0:
            return []
        q_emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = (self.embeddings @ q_emb).tolist()
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.labels[i], float(sims[i])) for i in idxs]

_INTENT_CATALOG = EmbeddingCatalog()

def _ensure_catalogs_loaded(force: bool = False):
    intents_json = _load_json(INTENTS_FILE) or {}

    intents_blocks = intents_json.get("intents", [])

    try:
        intents_mtime = os.path.getmtime(INTENTS_FILE) if os.path.exists(INTENTS_FILE) else 0
    except Exception:
        intents_mtime = 0

    if ( _INTENT_CATALOG.embeddings is None) or force:
        loaded = _INTENT_CATALOG.load_cache(INTENT_EMB_FNAME)
        if not loaded or (_INTENT_CATALOG and intents_mtime and (getattr(_INTENT_CATALOG, "_source_mtime", None) or 0) < intents_mtime):
            if SentenceTransformer is None:
                logger.warning("Embedding model missing: cannot build intent catalog")
            else:
                _INTENT_CATALOG._load_model()
                _INTENT_CATALOG.build_from_json_blocks(intents_blocks)
                _INTENT_CATALOG.save_cache(INTENT_EMB_FNAME)
        _INTENT_CATALOG._source_mtime = intents_mtime

def classify_intent(query: str) -> Dict[str, Any]:
    logger.info("classify_intent called")
    _ensure_catalogs_loaded()
    if SentenceTransformer is None:
        logger.warning("Embedding model not available; returning 'conversation'")
        return {"intent": "conversation", "score": 0.0}

    candidates = _INTENT_CATALOG.most_similar(query, top_k=1)
    if not candidates:
        return {"intent": "conversation", "score": 0.0}
    intent, score = candidates[0]
    logger.info("Intent candidate: %s (score=%.3f)", intent, score)
    if score < INTENT_THRESHOLD:
        return {"intent": "conversation", "score": float(score)}
    return {"intent": intent, "score": float(score)}

def get_intent_candidates(query: str, top_k: int = 3):
    logger.info("get_intent_candidates called")
    _ensure_catalogs_loaded()
    if SentenceTransformer is None:
        return []
    raw = _INTENT_CATALOG.most_similar(query, top_k=top_k)
    return [{"intent": i, "score": s} for i, s in raw]

def sentiment_tool(text: str) -> Dict[str, Any]:
    logger.info("sentiment_tool called")
    if pipeline is None:
        logger.warning("transformers pipeline missing; cannot run sentiment locally")
        return {"sentiment": "unknown", "confidence": 0.0}

    global _SENT_PIPE
    try:
        _SENT_PIPE
    except NameError:
        try:
            logger.info("Loading sentiment pipeline: %s", SENTIMENT_MODEL_NAME)
            _SENT_PIPE = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME, device=-1)
        except Exception as e:
            logger.exception("Failed to load sentiment model: %s", e)
            return {"sentiment": "unknown", "confidence": 0.0}

    try:
        out = _SENT_PIPE(text)
        if not out:
            return {"sentiment": "unknown", "confidence": 0.0}
        best = out[0]
        label = best.get("label") or best.get("score") and "POSITIVE"
        return {"sentiment": label, "confidence": float(best.get("score", 0.0))}
    except Exception:
        logger.exception("Sentiment pipeline failed")
        return {"sentiment": "unknown", "confidence": 0.0}

def classify_and_route(query: str) -> Dict[str, Any]:
    top = classify_intent(query)
    out = {"intent": top.get("intent"), "intent_score": top.get("score"), "command": None, "command_score": 0.0}

    return out

try:
    _ensure_catalogs_loaded()
except Exception:
    logger.debug("Initial catalog build failed (will retry on demand)", exc_info=True)

if __name__ == "__main__":
    logger.info("classification.py standalone test")
    q = input("Query> ")
    print("Intent:", classify_intent(q))
    print("Candidates:", get_intent_candidates(q, top_k=5))
    print("Route summary:", classify_and_route(q))
    s = input("Sentiment test> ")
    print("Sentiment:", sentiment_tool(s))
