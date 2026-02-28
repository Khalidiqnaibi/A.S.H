# ner.py
import re
# import sys
import logging
from typing import List, Dict, Any, Optional

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    SPACY_AVAILABLE = False

logger = logging.getLogger("ash.ner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys as _sys
    h = logging.StreamHandler(stream=_sys.stderr)
    logger.addHandler(h)


class NERExtractor:
    """
    Extract named entities from text.
    Primary: spaCy 'en_core_web_sm' (or user-specified model).
    Fallback: lightweight regex + heuristics that finds capitalized phrases and email/URLs/numbers.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                logger.info("Loaded spaCy model: %s", model_name)
            except Exception as e:
                logger.warning("spaCy model load failed: %s", e)
                self.nlp = None
        else:
            logger.info("spaCy not available: using fallback NER")

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Returns list of entities:
        [{ "text": "Khalid", "label": "PERSON", "start": 10, "end": 16, "confidence": 0.85 }, ...]
        Confidence is best-effort (spaCy has no built-in token-level confidence; we set 0.9 by default)
        """
        if self.nlp:
            doc = self.nlp(text)
            ents = []
            for ent in doc.ents:
                ents.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, "kb_id_", None) or 0.9
                })
            return ents

        # Fallback: capitalized phrase extractor + patterns
        ents = []
        # emails / urls
        for m in re.finditer(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text):
            ents.append({"text": m.group(1), "label": "EMAIL", "start": m.start(), "end": m.end(), "confidence": 0.85})
        for m in re.finditer(r"(https?://\S+)", text):
            ents.append({"text": m.group(1), "label": "URL", "start": m.start(), "end": m.end(), "confidence": 0.85})
        # capitalized sequences (names/organizations)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text):
            ents.append({"text": m.group(1), "label": "PROPER", "start": m.start(), "end": m.end(), "confidence": 0.6})
        # numbers/dates
        for m in re.finditer(r"\b(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{2}(?:am|pm)?)\b", text, flags=re.I):
            ents.append({"text": m.group(1), "label": "DATE_OR_TIME", "start": m.start(), "end": m.end(), "confidence": 0.7})
        return ents