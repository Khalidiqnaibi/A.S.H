"""
src/senses/voiceprint.py

A rough voiceprint of one utterance, so ASH can tell the user's voice from a
guest's.

This is not speaker verification in the security sense -- there is no
speaker model installed, and a recording of the user would pass. It is the
statistics of the voice's spectral envelope (MFCCs, their means and spreads
over the utterance), which differ enough between two people in the same
room to keep a guest saying "ash, open my email" from being treated as the
user. The vector is computed from the audio the microphone already has,
kept in memory with the utterance, and never written to disk; only the
enrolled average of the user's own utterances is saved, and "forget my
voice" deletes it.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger("ash.senses.voiceprint")

MIN_SECONDS = 0.8
N_MFCC = 20


def fingerprint(audio, sample_rate: int = 16000) -> Optional[List[float]]:
    """A unit vector describing the voice in `audio` (int16 or float), or None
    if there is too little of it, or librosa is not installed."""
    try:
        a = np.asarray(audio)
        if a.size < MIN_SECONDS * sample_rate:
            return None
        if a.dtype.kind == "i":
            a = a.astype(np.float32) / 32768.0
        else:
            a = a.astype(np.float32)
        if float(np.sqrt(np.mean(a * a))) < 1e-3:
            return None                                  # silence
        import librosa
        m = librosa.feature.mfcc(y=a, sr=sample_rate, n_mfcc=N_MFCC)
        if m.shape[1] < 10:
            return None
        m = m[1:]                                        # c0 is loudness, not voice
        v = np.concatenate([m.mean(axis=1), m.std(axis=1)])
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n == 0.0:
            return None
        return [float(x) for x in v / n]
    except ImportError:
        return None
    except Exception:
        logger.exception("Could not compute a voiceprint")
        return None
