# entity_model.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid
import time


@dataclass
class Entity:
    entity_id: str
    entity_type: str  # "person", "organization", "device", etc.
    
    # Strong identifiers (high confidence)
    primary_identifiers: Dict[str, Optional[str]] = field(default_factory=dict)

    # Contextual identifiers (weak signals)
    secondary_identifiers: Dict[str, Optional[str]] = field(default_factory=dict)

    # Structured attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update(self, new_attributes: Dict[str, Any]):
        self.attributes.update(new_attributes)
        self.updated_at = time.time()