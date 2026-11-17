"""
tst_nevermind.py -- "never mind" (said, or the console's button) takes the turn back: the episode, what was learned, the working memory.

It used to drop only the spoken answer; everything the turn wrote stayed, and later answers were built on it.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_nevermind.py
"""

import os
import sys
import tempfile
import time
from collections import deque
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import tst_freemoment as tf  # noqa: E402
from src.memory.episodic.episodic_manager import EpisodicMemory  # noqa: E402
from src.daemon.forget import UndoStack  # noqa: E402

check = tw.check


class Emb:
    def encode(self, x):
        if isinstance(x, str):
            return np.ones(4, dtype=np.float32)
        return np.ones((len(x), 4), dtype=np.float32)


def n1_episodes():
    print("\n=== N1. Episodes ===")
    m = EpisodicMemory(Emb(), path=os.path.join(tempfile.mkdtemp(), "e.jsonl"))
    a = m.add_episode("Immortal: before", "interaction")
    a.timestamp = time.time() - 100
    t = time.time() - 5
    m.add_episode("Immortal: a port is 8878", "interaction")
    m.add_episode("Immortal: and a version", "interaction")
    check("two were written in the turn", len(m.store.episodes) == 3)
    n = m.forget_since(t)
    check("both went, the earlier one stayed", n == 2 and [e.summary for e in m.store.episodes.values()] == ["Immortal: before"])
    check("the search index lost them too (rows and vectors agree)", len(m.index.episodes) == 1 and len(m.index.vectors) == 1)
    check("search still works", len(m.retrieve("anything")) == 1)
    again = EpisodicMemory(Emb(), path=m.store.path)
    check("and the file was rewritten without them", len(again.store.episodes) == 1)
    check("nothing to forget: 0", m.forget_since(time.time() + 10) == 0)


def n2_undo():
    print("\n=== N2. What was learned ===")
    u = UndoStack()
    log = []
    u.push("old", lambda: log.append("old"), now=100.0)
    u.push("a", lambda: log.append("a"), now=200.0)
    u.push("b", lambda: log.append("b"), now=201.0)
    check("steps since the turn started are undone, newest first", u.undo_since(150.0) == 2 and log == ["b", "a"], str(log))
    check("the older one is left", len(u) == 1)


def n3_runtime():
    print("\n=== N3. In the runtime ===")
    h = tf.rt_rig()
    rt = h.rt
    epi = EpisodicMemory(Emb(), path=os.path.join(tempfile.mkdtemp(), "e.jsonl"))
    rt.ash.episodic_memory = epi
    rt.brain = NS(memory=NS(wm=deque([NS(ts=1.0, text="old"), NS(ts=time.time(), text="this turn")], maxlen=20)))
    learned = []
    rt.undo.push("learned something", lambda: learned.append(1), now=time.time())
    epi.add_episode("Immortal: the thing I take back", "interaction")
    rt._recent_user.append("the thing I take back")
    rt._unwrite_turn(time.time() - 3, "the thing I take back")
    check("the episode is gone", len(epi.store.episodes) == 0)
    check("what was learned was undone", learned == [1])
    check("this turn's working-memory line is gone, the old one stays", [s.text for s in rt.brain.memory.wm] == ["old"])
    check("the line kept for checks is gone", "the thing I take back" not in rt._recent_user)
    check("counted", rt.stats.get("unwritten") == 1)


if __name__ == "__main__":
    n1_episodes()
    n2_undo()
    n3_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
