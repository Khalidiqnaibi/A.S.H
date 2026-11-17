"""
tests/semantic_fake.py -- a stand-in for the sentence encoder, for tests: words that mean the same thing share a dimension, so a
paraphrase the examples never contained still lands near them (which is what the real encoder does, in a form a test can control).
No model is loaded.
"""

import hashlib
import re

import numpy as np

GROUPS = [
    "thanks thank thx ty tysm cheers appreciate appreciated grateful obliged",
    "later bye goodbye cya ttyl peace leaving gotta off heading catch",
    "night goodnight gn sleep bedtime",
    "k kk kay okay noted gotcha understood alright fine",
    "feel feeling feelings emotional mood emotions state vibe heart mind mental",
    "status stats cpu memory ram disk battery usage load performance resources",
    "trivia quiz riddle puzzle question",
    "today todays daily",
    "memory remember remembered recall remembering conversations talked said",
    "notes note wrote written jotted",
    "wikipedia web internet google online encyclopedia",
    "people family friends contacts",
    "who whose whom",
    "weather rain temperature forecast",
    "play game start begin",
]
_IDX = {}
for i, g in enumerate(GROUPS):
    for w in g.split():
        _IDX[w] = i
DIM = 96
_G = len(GROUPS)


def _word_vec(w):
    v = np.zeros(DIM, dtype=np.float32)
    if w in _IDX:
        v[_IDX[w]] += 1.0
        return v
    h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
    v[_G + h % (DIM - _G)] += 1.0
    return v


def embed(texts):
    out = []
    for t in texts:
        words = re.findall(r"[a-z']+", str(t).lower())
        v = np.zeros(DIM, dtype=np.float32)
        for w in words:
            v += _word_vec(w)
        out.append(v)
    return np.stack(out)
