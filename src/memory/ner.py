# ner.py
import os
import re
import logging
from typing import List, Dict, Any

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

# 1. LABELS WE CARE ABOUT: Ignore numbers (CARDINAL), dates, and percents. 
USEFUL_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC", "NORP"}

class NERExtractor:
    """
    Extract named entities from text.
    Primary: spaCy 'en_core_web_sm' with Custom Entity Rules & dynamic stop words.
    Fallback: lightweight regex + heuristics.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        
        # Load custom A.S.H stop words from a flat text file
        self.custom_stop_words = self._load_custom_stops("ignore_words.txt")

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                
                # Inject Custom Knowledge into the AI
                # This ensures it ALWAYS recognizes your specific ecosystem terms perfectly.
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                patterns = [
                    {"label": "PERSON", "pattern": [{"LOWER": "immortal"}]},
                    {"label": "PERSON", "pattern": [{"LOWER": "immortal0ne"}]},
                    {"label": "PERSON", "pattern": [{"LOWER": "khalid"}]},
                    {"label": "PRODUCT", "pattern": [{"LOWER": "a.s.h"}]},
                ]
                ruler.add_patterns(patterns)
                
                logger.info("Loaded spaCy model with custom Entity Ruler: %s", model_name)
            except Exception as e:
                logger.warning("spaCy model load failed: %s", e)
                self.nlp = None
        else:
            logger.info("spaCy not available: using fallback NER")

    def _load_custom_stops(self, filepath: str) -> set:
        """Loads specific ignore words from a raw text file."""
        if not os.path.exists(filepath):
            logger.info(f"'{filepath}' not found. Using empty custom stop list.")
            return set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read lines, strip whitespace, ignore empty lines and comments
                return {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}
        except Exception as e:
            logger.warning(f"Could not load custom stop words from {filepath}: {e}")
            return set()

    def _is_valid_entity(self, text: str, label: str) -> bool:
        """Strict gatekeeper to filter out junk entities, numbers, and stop words."""
        cleaned = text.strip().lower()
        
        # 1. Native AI Stop Words: Let spaCy do the heavy lifting
        if self.nlp and self.nlp.vocab[cleaned].is_stop:
            return False
            
        # 2. Custom Domain Stop Words: Check your external config
        if cleaned in self.custom_stop_words:
            return False
            
        # 3. Ignore pure numbers, ordinals, percentages
        if cleaned.isdigit() or label in {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "DATE", "TIME"}:
            return False
            
        # 4. Only keep highly relevant entity categories
        if self.nlp and label not in USEFUL_LABELS:
            if label not in {"PROPER", "EMAIL", "URL"}:
                return False
                
        # 5. Ignore single characters or isolated emojis
        if len(cleaned) <= 2:
            return False
            
        return True

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Returns list of cleaned entities:
        [{ "text": "Khalid", "label": "PERSON", "start": 10, "end": 16, "confidence": 0.9 }, ...]
        """
        ents = []
        
        user_match = re.match(r"^([A-Za-z0-9_.-]+):\s*", text)
        clean_text = text
        
        if user_match:
            speaker_name = user_match.group(1)
            ents.append({
                "text": speaker_name,
                "label": "USER",  # New dedicated label!
                "start": user_match.start(1),
                "end": user_match.end(1),
                "confidence": 1.0
            })
            # Remove prefix for general NER so it doesn't get confused
            clean_text = text[user_match.end():]
        
        if self.nlp:
            doc = self.nlp(clean_text)
            for ent in doc.ents:
                if self._is_valid_entity(ent.text, ent.label_):
                    ents.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.9
                    })
            
            # De-duplicate identical entities in the same string
            unique_ents = {e['text'].lower(): e for e in ents}
            return list(unique_ents.values())

        # --- FALLBACK NER LOGIC ---
        for m in re.finditer(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", clean_text):
            ents.append({"text": m.group(1), "label": "EMAIL", "start": m.start(), "end": m.end(), "confidence": 0.85})
        for m in re.finditer(r"(https?://\S+)", clean_text):
            ents.append({"text": m.group(1), "label": "URL", "start": m.start(), "end": m.end(), "confidence": 0.85})
            
        # Fallback proper noun capture
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", clean_text):
            matched_text = m.group(1)
            # Prevent capturing the very first word of a sentence if it's in custom stops
            if m.start() == 0 and matched_text.lower() in self.custom_stop_words:
                continue 
                
            if self._is_valid_entity(matched_text, "PROPER"):
                ents.append({"text": matched_text, "label": "PROPER", "start": m.start(), "end": m.end(), "confidence": 0.6})
                
        unique_ents = {e['text'].lower(): e for e in ents}
        return list(unique_ents.values())


if __name__ == "__main__":
    extr = NERExtractor()
    
    test_sentence = "Immortal: HOLLY YOU ARE WORKING. finally. still running real slow but dont worry your creator Immortal (Khalid) is gonna fix ya right up"
    
    print("\n--- TEST RUN ---")
    print(f"Input: {test_sentence}")
    results = extr.extract(test_sentence)
    
    print("\nExtracted Entities:")
    for r in results:
        print(f"- {r['text']} [{r['label']}]")
    