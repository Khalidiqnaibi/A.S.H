"""
tools/shared_model.py

One copy of each sentence-transformer model per process, loaded on the CPU.

all-MiniLM-L6-v2 was loaded three times at every start (twice by the intent
classifier, once by the entity disambiguator): about 90 MB each and eight
seconds, on a laptop that was reporting 94% memory.

The disambiguator also created its model with no device, which makes
sentence-transformers probe CUDA -- the probe tools/classification.py already
avoids because on a machine with a bad GPU driver it can crash the process
natively, where no try/except can catch it. Every load goes through here now,
on "cpu"; ASH_USE_GPU=1 opts back in.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("ash.shared_model")

_MODELS: Dict[str, Any] = {}
_LOCK = threading.Lock()
loads = 0                      # how many times a model was really loaded (for the tests)


def device() -> Optional[str]:
    return None if os.environ.get("ASH_USE_GPU", "0") == "1" else "cpu"


def load(name: str):
    """The model called `name`, loaded once. Raises if it cannot be loaded."""
    global loads
    with _LOCK:
        m = _MODELS.get(name)
        if m is not None:
            return m
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading embedding model: %s", name)
        m = SentenceTransformer(name, device=device())
        _MODELS[name] = m
        loads += 1
        return m
