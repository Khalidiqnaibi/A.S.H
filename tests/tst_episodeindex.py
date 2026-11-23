"""
tst_episodeindex.py -- the episode index keeps its episodes and its vectors in step, even when embedding fails.

It appended the episode first and embedded after: an embedder that raised once left one more episode than vectors, and every search
after that returned the wrong episode for each hit.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_episodeindex.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.episodic.episode_index import EpisodeIndex  # noqa: E402
from src.memory.episodic.episode_model import Episode  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


class Emb:
    """One axis per word of a small vocabulary, so a query finds the episode that says its word."""
    VOCAB = ["alpha", "bravo", "charlie", "delta"]

    def __init__(self):
        self.fail = False

    def encode(self, x):
        if self.fail:
            raise RuntimeError("model busy")
        one = isinstance(x, str)
        rows = []
        for t in ([x] if one else x):
            rows.append([1.0 if w in t.lower() else 0.0 for w in self.VOCAB])
        a = np.array(rows, dtype=np.float32)
        return a[0] if one else a


def ep(i, text):
    return Episode(episode_id=str(i), timestamp=float(i), summary=text, event_type="interaction")


def main():
    print("\n=== E1. Appending keeps episodes and vectors in step ===")
    e = Emb()
    ix = EpisodeIndex(e)
    ix.append_single(ep(1, "alpha one"))
    ix.append_single(ep(2, "bravo two"))
    check("two in, two vectors", len(ix.episodes) == 2 and len(ix.vectors) == 2)
    e.fail = True
    ix.append_single(ep(3, "charlie three"))              # the embedder raises: nothing half-added
    check("a failed embed leaves the index in step", len(ix.episodes) == len(ix.vectors), f"{len(ix.episodes)} vs {len(ix.vectors)}")
    e.fail = False
    ix.append_single(ep(4, "delta four"))
    check("and the next one is the right row", len(ix.episodes) == len(ix.vectors)
          and ix.search("delta", 1)[0][1].episode_id == "4", str([(round(float(s), 2), x.episode_id) for s, x in ix.search("delta", 3)]))
    check("search still finds the early ones", ix.search("bravo", 1)[0][1].episode_id == "2")

    print("\n=== E2. Out of step is put right, not carried ===")
    e = Emb()
    ix = EpisodeIndex(e)
    ix.episodes = [ep(1, "alpha one"), ep(2, "bravo two")]   # loaded without vectors
    ix.append_single(ep(3, "charlie three"))
    check("episodes with no vectors yet: all indexed together", len(ix.vectors) == 3 and ix.search("charlie", 1)[0][1].episode_id == "3")
    ix.vectors = ix.vectors[:2]                                # one lost
    ix.append_single(ep(4, "delta four"))
    check("one vector short: rebuilt, in step again", len(ix.vectors) == len(ix.episodes) == 4
          and ix.search("delta", 1)[0][1].episode_id == "4")

    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
