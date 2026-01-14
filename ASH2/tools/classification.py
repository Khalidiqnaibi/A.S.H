# classification.py
"""
Semantic intent & command router + sentiment tool.

Usage:
  from classification import classify_intent, classify_command, sentiment_tool

Notes:
  - Requires: sentence-transformers, transformers
    pip install sentence-transformers transformers
  - Embeddings cached to disk (embeddings_intents.pkl / embeddings_commands.pkl).
  - Configure ASH_AI_BASE env var to point to the folder containing your JSON assets;
    defaults to CWD/ash_ai.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import json
import pickle
import time
import logging

import numpy as np

# LangChain tool decorator (keeps signature compatible)
from langchain.tools import tool

# Try to import sentence-transformers & transformers
try:
    from sentence_transformers import SentenceTransformer, util as sutil
except Exception as e:
    SentenceTransformer = None
    sutil = None
    print("[CLASSIFY] sentence-transformers not available:", e, file=sys.stderr, flush=True)

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    print("[CLASSIFY] transformers pipeline not available:", e, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Configuration: base folder where your JSON / pickles live
# ---------------------------------------------------------------------------
BASE_PATH = os.environ.get(
    "ASH_AI_BASE",
    os.path.join(os.getcwd(), "ash_ai")  # default ./ash_ai
)
INTENTS_FILE = os.path.join(BASE_PATH, "intents.json")
COMMAND_FILE = os.path.join(BASE_PATH, "command.json")
# cache files
INTENT_EMB_FNAME = os.path.join(BASE_PATH, "embeddings_intents.pkl")
COMMAND_EMB_FNAME = os.path.join(BASE_PATH, "embeddings_commands.pkl")

# model names
EMBEDDING_MODEL_NAME = os.environ.get("ASH_EMBED_MODEL", "all-MiniLM-L6-v2")
SENTIMENT_MODEL_NAME = os.environ.get("ASH_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

# similarity threshold defaults (tune as needed)
INTENT_THRESHOLD = 0.50
COMMAND_THRESHOLD = 0.55

# Logging helper
logger = logging.getLogger("ash.classify")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# ---------------------------------------------------------------------------
# Utilities: load JSON, build descriptions
# ---------------------------------------------------------------------------
def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.warning("JSON file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _intent_description_from_block(block: Dict[str, Any]) -> str:
    """
    Build a semantic description for an intent block.
    If the block has a 'description' field, prefer it.
    Otherwise, combine patterns into a short description.
    """
    if not isinstance(block, dict):
        return ""
    if "description" in block and block["description"]:
        return str(block["description"])
    patterns = block.get("patterns", [])
    # keep first few patterns as representative description
    sample = " ; ".join(patterns[:6])
    return f"Intent `{block.get('tag','?')}` — example phrases: {sample}"


def _command_description_from_block(block: Dict[str, Any]) -> str:
    # similar approach for commands JSON
    if not isinstance(block, dict):
        return ""
    if "description" in block and block["description"]:
        return str(block["description"])
    patterns = block.get("patterns", []) or block.get("examples", [])
    sample = " ; ".join(patterns[:6])
    return f"Command `{block.get('tag','?')}` — examples: {sample}"


# ---------------------------------------------------------------------------
# Embedding model loader and caching
# ---------------------------------------------------------------------------
class EmbeddingCatalog:
    """
    Loads embedding model, builds embeddings for a list of labels/descriptions,
    caches them to disk and reloads when source JSON mtime changes.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._load_model()
        # storage
        self.labels: List[str] = []
        self.descriptions: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._source_mtime: Optional[float] = None

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

    def build_from_json_blocks(self, blocks: List[Dict[str, Any]], is_command: bool = False):
        """
        blocks: list of intent/command blocks, each block must have 'tag' and optionally 'description'/'patterns'
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        labels = []
        descs = []
        for b in blocks:
            tag = b.get("tag", "").strip()
            if not tag:
                continue
            labels.append(tag)
            desc = _command_description_from_block(b) if is_command else _intent_description_from_block(b)
            descs.append(desc)
        self.labels = labels
        self.descriptions = descs
        # produce normalized embeddings (unit vectors) for fast cos sim
        emb = self.model.encode(descs, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb
        logger.info("Built embeddings for %d labels", len(labels))

    def save_cache(self, fname: str):
        """Save (labels, descriptions, embeddings) to disk."""
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
        """Return True if loaded from cache successfully."""
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

    def most_similar(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.model or self.embeddings is None:
            return []
        q_emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = (self.embeddings @ q_emb).tolist()  # dot product because normalized
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.labels[i], float(sims[i])) for i in idxs]


# ---------------------------------------------------------------------------
# Build catalogs (intents + commands) with caching logic
# ---------------------------------------------------------------------------
_INTENT_CATALOG = EmbeddingCatalog()
_COMMAND_CATALOG = EmbeddingCatalog()

def _ensure_catalogs_loaded(force: bool = False):
    """
    Load JSON, check mtime, (re)build embeddings if needed,
    or load caches when possible.
    """
    # load source files
    intents_json = _load_json(INTENTS_FILE) or {}
    commands_json = _load_json(COMMAND_FILE) or {}

    intents_blocks = intents_json.get("intents", [])
    commands_blocks = commands_json.get("intents", []) or commands_json.get("commands", [])

    # decide whether to (re)build intent embeddings
    try:
        intents_mtime = os.path.getmtime(INTENTS_FILE) if os.path.exists(INTENTS_FILE) else 0
    except Exception:
        intents_mtime = 0
    try:
        commands_mtime = os.path.getmtime(COMMAND_FILE) if os.path.exists(COMMAND_FILE) else 0
    except Exception:
        commands_mtime = 0

    # load or rebuild intent catalog
    if (not _INTENT_CATALOG.embeddings) or force:
        loaded = _INTENT_CATALOG.load_cache(INTENT_EMB_FNAME)
        if not loaded or (_INTENT_CATALOG and intents_mtime and (getattr(_INTENT_CATALOG, "_source_mtime", None) or 0) < intents_mtime):
            if SentenceTransformer is None:
                logger.warning("Embedding model missing: cannot build intent catalog")
            else:
                _INTENT_CATALOG._load_model()
                _INTENT_CATALOG.build_from_json_blocks(intents_blocks, is_command=False)
                _INTENT_CATALOG.save_cache(INTENT_EMB_FNAME)
        _INTENT_CATALOG._source_mtime = intents_mtime

    # load or rebuild command catalog
    if (not _COMMAND_CATALOG.embeddings) or force:
        loaded = _COMMAND_CATALOG.load_cache(COMMAND_EMB_FNAME)
        if not loaded or (_COMMAND_CATALOG and commands_mtime and (getattr(_COMMAND_CATALOG, "_source_mtime", None) or 0) < commands_mtime):
            if SentenceTransformer is None:
                logger.warning("Embedding model missing: cannot build command catalog")
            else:
                _COMMAND_CATALOG._load_model()
                _COMMAND_CATALOG.build_from_json_blocks(commands_blocks, is_command=True)
                _COMMAND_CATALOG.save_cache(COMMAND_EMB_FNAME)
        _COMMAND_CATALOG._source_mtime = commands_mtime


# ---------------------------------------------------------------------------
# LangChain tools: classification + sentiment
# ---------------------------------------------------------------------------

@tool
def classify_intent(query: str) -> Dict[str, Any]:
    """
    Classify the user's top-level intent using semantic embeddings.

    Returns: { "intent": <tag>, "score": <float> }
    If the top score is below INTENT_THRESHOLD, returns "conversation" as safe fallback.
    """
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


@tool
def get_intent_candidates(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Return top_k intent tag candidates with scores (useful for debugging and UI).
    """
    logger.info("get_intent_candidates called")
    _ensure_catalogs_loaded()
    if SentenceTransformer is None:
        return []
    raw = _INTENT_CATALOG.most_similar(query, top_k=top_k)
    return [{"intent": i, "score": s} for i, s in raw]


@tool
def classify_command(query: str) -> Dict[str, Any]:
    """
    Classify the user's command (only meaningful when top-level intent is 'command').
    Returns { "command": <tag>, "score": <float> }.
    If confidence < COMMAND_THRESHOLD, returns {command: None, score: <score>}.
    """
    logger.info("classify_command called")
    _ensure_catalogs_loaded()
    if SentenceTransformer is None:
        return {"command": None, "score": 0.0}

    candidates = _COMMAND_CATALOG.most_similar(query, top_k=1)
    if not candidates:
        return {"command": None, "score": 0.0}
    cmd, score = candidates[0]
    logger.info("Command candidate: %s (score=%.3f)", cmd, score)
    if score < COMMAND_THRESHOLD:
        return {"command": None, "score": float(score)}
    return {"command": cmd, "score": float(score)}


# Sentiment tool (downloadable HF model)
@tool
def sentiment_tool(text: str) -> Dict[str, Any]:
    """
    Run a local HuggingFace sentiment-analysis pipeline.
    Returns { "sentiment": "POSITIVE"|"NEGATIVE"|"NEUTRAL", "confidence": float }.
    Model: controlled by SENTIMENT_MODEL_NAME env var (default distilbert sst).
    """
    logger.info("sentiment_tool called")
    if pipeline is None:
        logger.warning("transformers pipeline missing; cannot run sentiment locally")
        return {"sentiment": "unknown", "confidence": 0.0}

    # create pipeline lazily to avoid heavy import on module load
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


# ---------------------------------------------------------------------------
# Optional helper: convenience combined call used by router code
# ---------------------------------------------------------------------------
def classify_and_route(query: str) -> Dict[str, Any]:
    """
    Synchronous helper for use by router code:
      - obtains top-level intent,
      - if intent == 'command' attempts to classify command tag,
      - returns a small dict summarizing the decisions.
    """
    top = classify_intent(query)
    out = {"intent": top.get("intent"), "intent_score": top.get("score"), "command": None, "command_score": 0.0}
    if out["intent"] and out["intent"].lower().startswith("command"):
        cmd = classify_command(query)
        out["command"] = cmd.get("command")
        out["command_score"] = cmd.get("score")
    return out


# ---------------------------------------------------------------------------
# Module-level initialization: ensure catalogs exist (non-blocking)
# ---------------------------------------------------------------------------
try:
    _ensure_catalogs_loaded()
except Exception:
    # non-fatal: will rebuild on first call
    logger.debug("Initial catalog build failed (will retry on demand)", exc_info=True)


# ---------------------------------------------------------------------------
# Demo / quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("classification.py standalone test")
    q = input("Query> ")
    print("Intent:", classify_intent(q))
    print("Candidates:", get_intent_candidates(q, top_k=5))
    print("Route summary:", classify_and_route(q))
    s = input("Sentiment test> ")
    print("Sentiment:", sentiment_tool(s))
