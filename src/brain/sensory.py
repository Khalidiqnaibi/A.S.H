"""
src/brain/sensory.py

Sensory Extractors  (the "CNNs / ViTs" box).

Turns raw input of any modality into a `Percept`: an L2-normalized embedding
plus two scalars the rest of the brain actually cares about --

    salience -- how loudly this input demands attention (length, imperatives,
                question marks, entity density, explicit urgency markers)
    novelty  -- how different this is from the last N things we perceived

Novelty is the single most important output here. It feeds the Predictive
Model (novel input => higher expected surprise) and the Executive Core
(novel input => don't trust the reflex path). A brain that has seen the exact
same stimulus fifty times should not deliberate on the fifty-first.

Extractors are pluggable. `TextExtractor` uses the sentence-transformer that
classification.py already loaded -- no second model in RAM. `VisionExtractor`
is wired for a real ViT/CLIP checkpoint and degrades to a cheap
histogram-of-color descriptor if torch/timm aren't installed, so the vision
path is exercisable without pulling a GPU stack in.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .signals import Percept, PerceptBundle

logger = logging.getLogger("ash.brain.sensory")

_IMPERATIVE_HEADS = {
    "do", "make", "run", "stop", "start", "delete", "remove", "send", "call",
    "open", "close", "set", "turn", "move", "go", "give", "show", "find",
    "write", "read", "fix", "kill", "restart", "shutdown", "buy", "pay",
}
_URGENCY_MARKERS = {
    "now", "immediately", "asap", "urgent", "emergency", "quick", "quickly",
    "hurry", "right now", "fast",
}


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class BaseExtractor:
    modality = "base"

    def extract(self, raw: Any) -> Optional[Percept]:
        raise NotImplementedError


class TextExtractor(BaseExtractor):
    """Language cortex. Reuses the already-loaded MiniLM encoder."""

    modality = "text"

    def __init__(self, embedder=None, dim_fallback: int = 384):
        self.embedder = embedder
        self.dim_fallback = dim_fallback

    def _embed(self, text: str) -> np.ndarray:
        if self.embedder is not None:
            try:
                v = self.embedder.encode(
                    [text], convert_to_numpy=True, normalize_embeddings=True
                )[0]
                return np.asarray(v, dtype=np.float32)
            except Exception:
                logger.exception("Text embedder failed; falling back to hash embedding")
        # Deterministic hash embedding so the pipeline still runs with no model.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return _l2(rng.standard_normal(self.dim_fallback).astype(np.float32))

    @staticmethod
    def _salience(text: str, tokens: List[str]) -> float:
        s = 0.15
        low = text.lower()
        if tokens and tokens[0] in _IMPERATIVE_HEADS:
            s += 0.25
        if "?" in text:
            s += 0.15
        if "!" in text:
            s += 0.10
        if any(m in low for m in _URGENCY_MARKERS):
            s += 0.30
        # Proper-noun / entity density: capitalized tokens not at sentence start.
        caps = sum(1 for t in text.split()[1:] if t[:1].isupper())
        s += min(0.15, caps * 0.05)
        # Very long input is inherently demanding.
        s += min(0.15, len(tokens) / 200.0)
        return float(min(1.0, s))

    def extract(self, raw: Any) -> Optional[Percept]:
        text = "" if raw is None else str(raw)
        if not text.strip():
            return None
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        emb = self._embed(text)
        return Percept(
            modality="text",
            raw=text,
            embedding=emb,
            tokens=tokens,
            salience=self._salience(text, tokens),
            meta={"char_len": len(text), "token_len": len(tokens)},
        )


class VisionExtractor(BaseExtractor):
    """Visual cortex. Real ViT if available, cheap descriptor otherwise.

    Accepts a file path, a PIL image, or an HxWxC ndarray. The embedding is
    projected to `target_dim` so it can be concatenated/fused with text
    without the caller caring which backend produced it.
    """

    modality = "vision"

    def __init__(self, model_name: str = "google/vit-base-patch16-224", target_dim: int = 384):
        self.model_name = model_name
        self.target_dim = target_dim
        self._backend = None
        self._proj: Optional[np.ndarray] = None
        self._try_load()

    def _try_load(self):
        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
            import torch  # noqa: F401

            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._backend = "vit"
            logger.info("VisionExtractor: loaded ViT %s", self.model_name)
        except Exception as e:
            self._backend = "descriptor"
            logger.info("VisionExtractor: ViT unavailable (%s); using color/edge descriptor", e)

    def _project(self, v: np.ndarray) -> np.ndarray:
        if v.shape[0] == self.target_dim:
            return _l2(v)
        if self._proj is None or self._proj.shape != (v.shape[0], self.target_dim):
            rng = np.random.default_rng(1337)  # fixed seed => stable projection
            self._proj = rng.standard_normal((v.shape[0], self.target_dim)).astype(np.float32)
            self._proj /= np.sqrt(v.shape[0])
        return _l2(v @ self._proj)

    @staticmethod
    def _to_array(raw: Any) -> Optional[np.ndarray]:
        if isinstance(raw, np.ndarray):
            return raw
        try:
            from PIL import Image  # type: ignore

            img = Image.open(raw) if isinstance(raw, (str, bytes)) else raw
            return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            return None

    def _descriptor(self, arr: np.ndarray) -> np.ndarray:
        """Color histogram + gradient energy per 4x4 spatial cell.

        Not a ViT. It is, however, a real visual feature vector: it changes
        when the scene changes and stays put when it doesn't, which is all
        the novelty/surprise machinery downstream actually requires.
        """
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w, _ = arr.shape
        cells = []
        for gy in range(4):
            for gx in range(4):
                cell = arr[gy * h // 4:(gy + 1) * h // 4, gx * w // 4:(gx + 1) * w // 4]
                if cell.size == 0:
                    cells.append(np.zeros(6, dtype=np.float32))
                    continue
                mean = cell.reshape(-1, 3).mean(axis=0)
                std = cell.reshape(-1, 3).std(axis=0)
                cells.append(np.concatenate([mean, std]).astype(np.float32))
        gray = arr.mean(axis=2)
        gx_e = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
        gy_e = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
        return np.concatenate([np.concatenate(cells), np.array([gx_e, gy_e], dtype=np.float32)])

    def extract(self, raw: Any) -> Optional[Percept]:
        arr = self._to_array(raw)
        if arr is None:
            return None
        if self._backend == "vit":
            try:
                import torch

                inputs = self._processor(images=arr, return_tensors="pt")
                with torch.no_grad():
                    out = self._model(**inputs)
                vec = out.last_hidden_state[:, 0].squeeze(0).numpy().astype(np.float32)
            except Exception:
                logger.exception("ViT forward failed; falling back to descriptor")
                vec = self._descriptor(arr)
        else:
            vec = self._descriptor(arr)

        emb = self._project(np.asarray(vec, dtype=np.float32))
        energy = float(np.clip(np.abs(vec).mean() * 2.0, 0.0, 1.0))
        return Percept(
            modality="vision",
            raw=None,
            embedding=emb,
            salience=0.4 + 0.4 * energy,
            meta={"backend": self._backend, "shape": list(arr.shape)},
        )


class SensoryCortex:
    """Runs every registered extractor, fuses the results, scores novelty.

    Fusion is drive-free weighted averaging by salience -- deliberately dumb.
    The interesting behavior lives downstream; this layer's only job is to
    hand the rest of the brain one comparable vector per cycle.
    """

    def __init__(self, embedder=None, history: int = 64, enable_vision: bool = False,
                 latent_dim: int = 384):
        self.latent_dim = latent_dim
        self.text = TextExtractor(embedder=embedder, dim_fallback=latent_dim)
        self.vision: Optional[VisionExtractor] = (
            VisionExtractor(target_dim=latent_dim) if enable_vision else None
        )
        self._history: Deque[np.ndarray] = deque(maxlen=history)
        self._recent_hashes: Deque[str] = deque(maxlen=history)

    def enable_vision(self):
        if self.vision is None:
            self.vision = VisionExtractor(target_dim=self.latent_dim)

    def _novelty(self, emb: Optional[np.ndarray]) -> float:
        """1 - max cosine similarity against recent percepts. First-ever
        input is maximally novel."""
        if emb is None or not self._history:
            return 1.0
        sims = [float(np.dot(emb, h)) for h in self._history if h.shape == emb.shape]
        if not sims:
            return 1.0
        return float(np.clip(1.0 - max(sims), 0.0, 1.0))

    def perceive(self, text: str = "", image: Any = None, extra: Optional[Dict[str, Any]] = None) -> PerceptBundle:
        percepts: List[Percept] = []

        p_text = self.text.extract(text)
        if p_text is not None:
            percepts.append(p_text)

        if image is not None:
            if self.vision is None:
                self.enable_vision()
            p_vis = self.vision.extract(image) if self.vision else None
            if p_vis is not None:
                percepts.append(p_vis)

        if not percepts:
            return PerceptBundle(percepts=[], fused_embedding=None, text=text or "")

        embs = [(p.embedding, max(p.salience, 0.05)) for p in percepts if p.embedding is not None]
        if embs:
            dim = embs[0][0].shape[0]
            acc = np.zeros(dim, dtype=np.float32)
            wsum = 0.0
            for v, w in embs:
                if v.shape[0] != dim:
                    continue
                acc += v * w
                wsum += w
            fused = _l2(acc / wsum) if wsum > 0 else None
        else:
            fused = None

        novelty = self._novelty(fused)
        for p in percepts:
            p.novelty = novelty

        if fused is not None:
            self._history.append(fused)

        bundle = PerceptBundle(
            percepts=percepts,
            fused_embedding=fused,
            text=text or "",
            salience=max(p.salience for p in percepts),
            novelty=novelty,
            timestamp=time.time(),
        )
        if extra:
            for p in percepts:
                p.meta.update(extra)
        return bundle
