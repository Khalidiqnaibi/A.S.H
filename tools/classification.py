# tools/classification.py
"""
Semantic intent & command router + sentiment tool.

Requirements:
  pip install sentence-transformers transformers
Configure with ASH_AI_BASE env var (defaults to ./ash_ai folder).
"""
from typing import List, Dict, Any, Tuple, Optional
import os
import re
import sys
import json
import pickle
import logging
import dotenv

# MUST run before importing sentence_transformers/transformers below.
# huggingface_hub reads HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE into a frozen
# module-level constant (see huggingface_hub/constants.py) the instant it's
# first imported -- and sentence_transformers imports it transitively. A
# .env setting HF_HUB_OFFLINE=1 loaded AFTER that point is silently too
# late: os.environ gets updated, but huggingface_hub already locked in
# False, so it retries the network (and hangs for ~60s doing 5 backoff
# retries) on an air-gapped machine regardless of what .env says.
dotenv.load_dotenv()

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

BASE_PATH = os.environ.get("ASH_AI_BASE", os.path.join(os.getcwd(), ".\data"))
INTENTS_FILE = os.path.join(BASE_PATH, "intents.json")
INTENT_EMB_FNAME = os.path.join(BASE_PATH, "embeddings_intents.pkl")

EMBEDDING_MODEL_NAME = os.environ.get("ASH_EMBED_MODEL", "all-MiniLM-L6-v2")
SENTIMENT_MODEL_NAME = os.environ.get("ASH_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

INTENT_THRESHOLD = 0.21

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


# Each intent is matched through its description AND every example phrase, not
# the description alone: a query is compared with the phrases people actually
# say ("what is 12 times 7" scored 0.16 for calculate against the description,
# and that exact sentence is one of its examples). An intent's score is the
# mean of its two best-matching rows.
#
# The raw scores of that rule sit higher than the description-only scores every
# threshold downstream (the executive's match floor, the gate, INTENT_THRESHOLD)
# was tuned on, so they are mapped back onto that scale. Fitted by matching the
# medians of tool queries and of plain chat over a few dozen real queries; it
# changes how well intents are told apart, not what a given score means.
CATALOG_SCORE_SLOPE = 0.73
CATALOG_SCORE_OFFSET = -0.05
CATALOG_CACHE_VERSION = 2


class EmbeddingCatalog:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.labels = []
        self.descriptions = []
        self.embeddings = None
        self._source_mtime = None
        self._load_model()

    def get_model(self):
        return self.model

    def _load_model(self):
        if SentenceTransformer is None:
            logger.error("SentenceTransformer not available. Install sentence-transformers.")
            return
        try:
            logger.info("Loading embedding model: %s", self.model_name)
            # Explicit device= skips sentence-transformers' own
            # torch.cuda.is_available() auto-detect entirely (see
            # sentence_transformers.util.get_device_name -- it's only
            # called when device is None). That auto-detect is what
            # probes the NVIDIA driver, which can crash natively
            # (STATUS_ACCESS_VIOLATION) on a machine with no GPU or a
            # mismatched driver -- a hardware fault no try/except can
            # catch. Passing "cpu" up front avoids the probe altogether;
            # set ASH_USE_GPU=1 to opt back into auto-detection.
            device = None if os.environ.get("ASH_USE_GPU", "0") == "1" else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
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
            descs.append(_intent_description_from_block(b))
            for phrase in b.get("patterns", []) or []:
                phrase = str(phrase).strip()
                if phrase:
                    labels.append(tag)
                    descs.append(phrase)
        self.labels = labels
        self.descriptions = descs
        emb = self.model.encode(descs, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb
        logger.info("Built embeddings for %d labels", len(labels))

    def save_cache(self, fname: str):
        try:
            with open(fname, "wb") as fh:
                pickle.dump({
                    "version": CATALOG_CACHE_VERSION,
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
                if data.get("version") != CATALOG_CACHE_VERSION:
                    return False        # built by the old one-row-per-intent rule
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
        by_label: Dict[str, List[float]] = {}
        for label, sim in zip(self.labels, sims):
            by_label.setdefault(label, []).append(float(sim))
        scored = []
        for label, ss in by_label.items():
            best = sorted(ss, reverse=True)[:2]
            raw = sum(best) / len(best)
            scored.append((label, max(0.0, min(1.0, CATALOG_SCORE_SLOPE * raw + CATALOG_SCORE_OFFSET))))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

_INTENT_CATALOG = EmbeddingCatalog()

def _ensure_catalogs_loaded(force: bool = False):
    intents_json = _load_json(INTENTS_FILE) or {}
    intents_blocks = intents_json.get("intents", [])

    # Merge in per-tool shape files (tools/shapes/*.json) -- this is
    # what lets classify_and_route() recognize new tools without
    # anyone hand-editing intents.json. See tools/shape_loader.py.
    try:
        from tools.shapes.shape_loader import load_all_shapes, shapes_to_intent_blocks, shapes_max_mtime
        shape_blocks = shapes_to_intent_blocks(load_all_shapes())
        existing_tags = {b.get("tag") for b in intents_blocks}
        for b in shape_blocks:
            if b["tag"] in existing_tags:
                logger.warning("Shape tag '%s' collides with an intents.json tag -- shape wins", b["tag"])
                intents_blocks = [x for x in intents_blocks if x.get("tag") != b["tag"]]
            intents_blocks.append(b)
        shapes_mtime = shapes_max_mtime()
    except Exception:
        logger.exception("Failed to load tool shapes -- continuing with intents.json only")
        shapes_mtime = 0

    try:
        intents_mtime = os.path.getmtime(INTENTS_FILE) if os.path.exists(INTENTS_FILE) else 0
    except Exception:
        intents_mtime = 0
    intents_mtime = max(intents_mtime, shapes_mtime)

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

ARITHMETIC_SCORE = 0.85


def _is_arithmetic(query: str) -> bool:
    try:
        from tools.lesstools import looks_like_arithmetic
        return looks_like_arithmetic(query)
    except Exception:
        return False


# Structural evidence the embeddings are poor at: a file path in the request,
# a duration, a quantity with units. Each adds a fixed amount to the intents it
# points at -- it re-ranks and lifts candidates the catalog already found
# plausible, and cannot conjure one from nothing.
_PATH_HINT = re.compile(r"(?:[\w.\-]+[/\\][\w.\-]+|\b[\w\-]+\.[A-Za-z0-9]{1,6}\b)")
_UNIT_WORD = (r"km|kilomet\w+|mi|miles?|m|meters?|metres?|cm|mm|ft|feet|foot|inch(?:es)?|yd|yards?|"
              r"kg|kilogram\w*|g|grams?|lbs?|pounds?|oz|ounces?|l|liters?|litres?|ml|gal(?:lons?)?|"
              r"cups?|celsius|fahrenheit|[cf]")
_PRIORS = [
    (re.compile(r"\bremind(?:er)?\b|\bdon'?t\s+(?:let\s+me\s+)?forget\b", re.I),
     {"set_reminder": 0.20}),
    (re.compile(r"\bin\s+(?:an?|half\s+an?|\d+|one|two|three|five|ten|twenty|thirty)\s*"
                r"(?:seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?)\b|\btomorrow\b", re.I),
     {"set_reminder": 0.12}),
    (re.compile(r"\d\s*(?:" + _UNIT_WORD + r")\b.*\b(?:to|in|into)\b|\bhow\s+many\s+(?:" + _UNIT_WORD + r")\b",
                re.I),
     {"unit_convert": 0.20}),
    (re.compile(r"\b(?:cpu|ram|memory|temperature|temp|uptime|disk|battery|processor)\b", re.I),
     {"system_stats": 0.20}),
    (re.compile(r"\bsentiment\b|\b(?:mood|tone)\s+of\b", re.I), {"sentiment_analysis": 0.20}),
    (re.compile(r"\b(?:words?|characters?)\b.*\b(?:in|of|is|are|count)\b|\bword\s+count\b|\breading\s+time\b",
                re.I), {"word_count": 0.12}),
    (re.compile(r"\b(?:grep|search|find|look)\b", re.I), {"search_files": 0.10}),
    (re.compile(r"\b(?:ls|dir)\b|\b(?:list|show)\b.*\b(?:files|folders?|director\w+)\b|"
                r"\bwhat'?s\s+in\b.*\b(?:folder|directory)\b|\bfiles\s+in\b", re.I),
     {"list_directory": 0.20}),
]
_FILE_INTENTS = ("read_file", "file_info", "list_directory", "search_files")
_PATH_BOOST = 0.15
PRIOR_CAP = 0.9

# A request to destroy or change the machine: nothing here can do it, and a
# read-only tool answering "rm -rf /" with a directory listing is wrong
# whichever way it is read. These go to conversation, which can say no.
_DESTRUCTIVE = re.compile(
    r"^\s*(?:please\s+)?(?:delete|remove|erase|wipe|format|shut\s*down|shutdown|reboot|restart|kill|"
    r"drop|truncate|uninstall|rm|rmdir|del|mkfs|sudo)\b|\brm\s+-\w*r|\bdrop\s+table\b|\bformat\s+(?:the\s+)?"
    r"(?:disk|drive|hard\s*drive|c:)", re.I)

# Asking ASH how it is, said to a person and not to a box.
#
# The embedding cannot tell these apart from a hardware query and there is no
# amount of pattern-tuning that will teach it to: "how are you" and "how's the
# system doing" are a hair apart in MiniLM space, and system_stats.json has to
# keep the second one. So "what's your status" came back as
# `cpu_percent: 39.3, ram_used_percent: 93.0, ram_used_mb: 15145` -- the three
# phrases a person is most likely to open with, answered with telemetry.
#
# There is no tool that answers these. The right handler is the narrator,
# which already has the emotional modulation block and the drive vector, so
# ASH can say how it actually is. Vetoed to conversation the same way a
# destructive request is, and for the same reason: not "this tool scores
# lower", but "no tool is the answer to this".
_SELF_SMALL_TALK = re.compile(
    r"^\s*(?:so|and|but|hey|ok|okay)?[\s,]*"
    # "we" belongs here with "you": "how are we doing?" is the same question
    # asked more kindly, and it was in the live run that started all this.
    r"(?:how(?:'?s|'?re| is| are| have| has)?\s+(?:you|u|ya|we|yourself|things|it\s+going|"
    r"you\s+doing|we\s+doing)"
    r"|how(?:'?ve| have)\s+(?:you|we)\s+been"
    r"|are\s+you\s+(?:there|ok|okay|alright|all\s+right|around|awake|up|busy|free|"
    r"listening|still\s+there|with\s+me|good|fine|well|real|sure|sentient|alive)"
    r"|you\s+(?:there|ok|okay|alright|awake|around|good|still\s+there)"
    r"|what(?:'?s| is)\s+(?:your|ur)\s+status"
    r"|what(?:'?s| is)\s+(?:up|new|going\s+on)(?:\s+with\s+you)?"
    r"|(?:you\s+)?(?:still|there)\s*\?)"
    r"\b", re.I)
# ...unless it names something only the machine has, in which case it really
# is a hardware question: "how are you doing on memory", "are you up still --
# what's the uptime".
_MACHINE_NOUN = re.compile(
    r"\b(?:cpu|ram|memory|mem|temperature|temp|uptime|disk|storage|battery|processor|"
    r"system|machine|hardware|pi|server|load|fan|swap|gpu)\b", re.I)


def is_self_small_talk(query: str) -> bool:
    """"How are you" -- a question for the narrator, not for a tool."""
    q = query or ""
    return bool(_SELF_SMALL_TALK.search(q)) and not _MACHINE_NOUN.search(q)


# "hi ash", "hey ash, what time is it": the name is an address, not a topic.
_ADDRESS = re.compile(r"^\s*(?:(?:hey|hi|hello|ok|okay|yo)[,\s]+)?ash\b[,.!\s]*|[,\s]+ash[.!?\s]*$", re.I)


def strip_address(query: str) -> str:
    return _ADDRESS.sub("", query or "", count=1).strip()


def _apply_priors(query: str, scored):
    """scored: [(intent, score)] -> same shape, re-ranked by structural evidence."""
    q = query or ""
    boost: Dict[str, float] = {}
    for rx, targets in _PRIORS:
        if rx.search(q):
            for intent, amount in targets.items():
                boost[intent] = boost.get(intent, 0.0) + amount
    if _PATH_HINT.search(q) and not re.search(r"\d\s*(?:" + _UNIT_WORD + r")\b", q, re.I):
        for intent in _FILE_INTENTS:
            boost[intent] = boost.get(intent, 0.0) + _PATH_BOOST
    if not boost:
        return list(scored)
    out = [(i, min(PRIOR_CAP, sc + boost.get(i, 0.0))) for i, sc in scored]
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


# Asking ASH to look at the screen. "what you see" with no noun is a real
# request ("Yo ash, what you see?"), so this is phrases, not just "screen".
#
# It is a veto as well as a boost, for the same reason the small-talk one is:
# MiniLM puts "what do you think" (0.28) and "see you later" (0.28) close
# enough to "what do you see" to fire a tool, and this tool takes a screenshot
# of the user's screen. A conversational sentence must not do that on a
# near-miss in embedding space.
_SCREEN_ASK = re.compile(
    r"\b(?:screens?|screenshots?|monitor|desktop)\b"
    r"|\bwhat(?:'?s|\s+is|\s+am\s+i|\s+are\s+you|\s+do\s+you|\s+can\s+you|\s+you)?\s+"
    r"(?:you\s+)?(?:see(?:ing)?|looking\s+at)\b"
    r"|\bwhat'?s\s+open\b", re.I)


def _ranked(query: str, top_k: int):
    """Catalog candidates for `query`, with the structural priors applied."""
    cat = _INTENT_CATALOG
    wide = max(top_k, len(set(cat.labels)))          # priors may promote a lower-ranked intent
    ranked = _apply_priors(query, cat.most_similar(query, top_k=wide))
    if not _SCREEN_ASK.search(query or ""):
        ranked = [(i, sc) for i, sc in ranked if i != "screen_view"]
    return ranked[:top_k]


def classify_intent(query: str) -> Dict[str, Any]:
    logger.info("classify_intent called")
    _ensure_catalogs_loaded()
    if SentenceTransformer is None:
        logger.warning("Embedding model not available; returning 'conversation'")
        return {"intent": "conversation", "score": 0.0}

    query = strip_address(query)
    if not query or _DESTRUCTIVE.search(query) or is_self_small_talk(query):
        return {"intent": "conversation", "score": 0.0}
    if _is_arithmetic(query):
        return {"intent": "calculate", "score": ARITHMETIC_SCORE}
    candidates = _ranked(query, top_k=1)
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
    query = strip_address(query)
    if not query or _DESTRUCTIVE.search(query) or is_self_small_talk(query):
        return []
    out = [{"intent": i, "score": s} for i, s in _ranked(query, top_k=top_k)]
    if _is_arithmetic(query):
        out = [{"intent": "calculate", "score": ARITHMETIC_SCORE}] + [
            c for c in out if c["intent"] != "calculate"][:max(0, top_k - 1)]
    return out

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
            _SENT_PIPE = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME, device=-1, framework="pt")
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