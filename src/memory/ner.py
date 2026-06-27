# ner.py
import os
import re
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

# 1. LABELS WE CARE ABOUT: Ignore numbers (CARDINAL), dates, and percents. 
USEFUL_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC", "NORP"}

class NERExtractor:
    """
    Extract named entities from text.
    Primary: spaCy 'en_core_web_sm' with Custom Entity Rules & dynamic stop words.
    State-Aware: Resolves pronouns to actual entity targets (I -> Immortal).
    Fallback: lightweight regex + heuristics.
    """

    def __init__(self, model_name: str = "en_core_web_sm", assistant_name: str = "A.S.H"):
        self.model_name = model_name
        self.assistant_name = assistant_name
        self.nlp = None
        
        # State trackers for Coreference Resolution
        self.last_speaker = None
        self.last_subject = None
                        
        # Load custom A.S.H stop words from a flat text file
        self.custom_stop_words = self._load_custom_stops("ignore_words.txt")

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                
                # Inject Custom Knowledge into the AI
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
                return {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}
        except Exception as e:
            logger.warning(f"Could not load custom stop words from {filepath}: {e}")
            return set()

    def _is_valid_entity(self, text: str, label: str) -> bool:
        """Strict gatekeeper to filter out junk entities, numbers, and stop words."""
        cleaned = text.strip().lower()
        
        # 1. Native AI Stop Words & Domain Stop Words
        if (self.nlp and self.nlp.vocab[cleaned].is_stop) or (cleaned in self.custom_stop_words):
            return False
            
        # 2. Ignore pure numbers, ordinals, percentages
        if cleaned.isdigit() or label in {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "DATE", "TIME"}:
            return False
            
        # 3. Only keep highly relevant entity categories
        if self.nlp and label not in USEFUL_LABELS:
            if label not in {"PROPER", "EMAIL", "URL"}:
                return False
                
        # 4. Ignore single characters or isolated emojis
        if len(cleaned) <= 2:
            return False
            
        return True
    
    def _resolve_pronoun(self, token_text: str) -> Optional[str]:
        """Simple state-based coreference resolution."""
        pronouns = {
            "i": self.last_speaker,
            "me": self.last_speaker,
            "my": self.last_speaker,
            "you": self.assistant_name if self.last_speaker != self.assistant_name else self.last_subject,
            "he": self.last_subject,
            "she": self.last_subject
        }
        return pronouns.get(token_text.lower())

    def extract(self, text: str) -> List[Dict[str, Any]]:
        ents = []
        
        # 1. SPEAKER DETECTION (Prefix split)
        speaker_match = re.match(r"^([A-Za-z0-9_.-]+):\s*", text)
        clean_text = text
        
        if speaker_match:
            speaker_name = speaker_match.group(1).strip()
            self.last_speaker = speaker_name
            
            # Specifically flag whether this is the User or the AI
            is_self = speaker_name.lower() == self.assistant_name.lower()
            label = "SPEAKER_SELF" if is_self else "SPEAKER_USER"
            
            ents.append({
                "text": speaker_name,
                "label": label,
                "start": speaker_match.start(1),
                "end": speaker_match.end(1),
                "confidence": 1.0
            })
            clean_text = text[speaker_match.end():]

        # 2. STANDARD ENTITY EXTRACTION
        if self.nlp:
            doc = self.nlp(clean_text)
            for ent in doc.ents:
                if self._is_valid_entity(ent.text, ent.label_):
                    self.last_subject = ent.text # Update target context state
                    
                    ents.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char + (len(text) - len(clean_text)),
                        "end": ent.end_char + (len(text) - len(clean_text)),
                        "confidence": 0.9
                    })
        
        # 3. TRANSPARENT PRONOUN RESOLUTION
        # Instead of saving "I", we swap it out for "Immortal" right here.
        for m in re.finditer(r'\b(i|me|my|you|he|she)\b', clean_text, re.IGNORECASE):
            clean_word = m.group(1).lower()
            resolved = self._resolve_pronoun(clean_word)
            if resolved:
                # Give it a label depending on if it resolved to the AI or a Person
                label = "PRODUCT" if resolved.lower() == self.assistant_name.lower() else "PERSON"
                ents.append({
                    "text": resolved, 
                    "label": label,
                    "start": m.start() + (len(text) - len(clean_text)),
                    "end": m.end() + (len(text) - len(clean_text)),
                    "confidence": 0.8
                })
    
        # 4. FALLBACK NER LOGIC
        for m in re.finditer(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", clean_text):
            ents.append({"text": m.group(1), "label": "EMAIL", "start": m.start(), "end": m.end(), "confidence": 0.85})
        for m in re.finditer(r"(https?://\S+)", clean_text):
            ents.append({"text": m.group(1), "label": "URL", "start": m.start(), "end": m.end(), "confidence": 0.85})
            
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", clean_text):
            matched_text = m.group(1)
            # Skip if it's just a capitalized stop word
            if m.start() == 0 and matched_text.lower() in self.custom_stop_words:
                continue 
                
            if self._is_valid_entity(matched_text, "PROPER"):
                self.last_subject = matched_text
                ents.append({"text": matched_text, "label": "PROPER", "start": m.start(), "end": m.end(), "confidence": 0.6})
                
        # 5. DEDUPLICATION
        unique_ents = {}
        for e in ents:
            key = e['text'].lower()
            if key not in unique_ents:
                unique_ents[key] = e
                
        return list(unique_ents.values())


if __name__ == "__main__":
    extr = NERExtractor()
    
    test_sentence = "Immortal: HOLLY YOU ARE WORKING. finally. still running real slow but dont worry your creator Khalid is gonna fix ya right up"
    
    print("\n--- TEST RUN ---")
    print(f"Input: {test_sentence}")
    results = extr.extract(test_sentence)
    
    print("\nExtracted Entities:")
    for r in results:
        print(f"- {r['text']} [{r['label']}]")