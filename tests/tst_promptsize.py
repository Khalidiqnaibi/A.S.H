"""
tst_promptsize.py -- a shorter, fresher prompt and a bounded answer.

From the session of 2026-11-06: "what do you see?" got "the YOLO pipeline as
we discussed in late June"; replies took 20-60 s, the model asked for up to
512 tokens that were clipped to ~150 afterwards.

  P1  Old episodes must be more on-topic than recent ones; important ones
      always get through
  P2  Fewer episodes and entities go into the prompt by default
  P3  The model is asked for no more than can be said aloud

Run:  PYTHONIOENCODING=utf-8 python tests/tst_promptsize.py
"""

import importlib
import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.memory.memory_router import MemoryRouter  # noqa: E402

check = tw.check
DAY = 86400.0


def router(episodes):
    r = MemoryRouter.__new__(MemoryRouter)
    r.config = {"retrieval_fetch_multiplier": 1, "retrieval_relevance_threshold": 0.12,
                "episodic_always_include_importance": 0.85, "episodic_stale_days": 14.0, "episodic_stale_extra": 0.10,
                "episodic_old_days": 60.0, "episodic_old_extra": 0.08}
    r.episodic = NS(retrieve=lambda q, top_k=8: episodes)
    return r


def ep(summary, days_old, importance=0.5, native=None):
    e = NS(summary=summary, timestamp=time.time() - days_old * DAY, importance=importance)
    if native is not None:
        e.score = native
    return e


def p1_stale():
    print("\n=== P1. Old episodes ===")
    q = "what do you see on the camera"
    toks = {"what", "see", "camera"}
    r = router([])
    check("fresh: no extra", r._staleness(ep("x", 1)) == 0.0)
    check("two weeks on: a little more", abs(r._staleness(ep("x", 20)) - 0.10) < 1e-9)
    check("two months on: more still", abs(r._staleness(ep("x", 90)) - 0.18) < 1e-9)
    check("no timestamp: no extra", r._staleness(NS(summary="x", timestamp=None, created_at=0)) == 0.0)

    weak = "you mentioned a camera pipeline for the YOLO detection project in late June"
    fresh = router([ep(weak, 2)])
    block, kept = fresh._build_episodic_block(q, toks, 5, 0.12)
    check("a weakly related episode from this week: kept", len(kept) == 1, block)
    old = router([ep(weak, 100)])
    block, kept = old._build_episodic_block(q, toks, 5, 0.12)
    check("the same, 100 days old: left out", kept == [] and block == "", block)
    strong = "you asked what do you see on the camera and the camera showed the desk"
    block, kept = router([ep(strong, 100)])._build_episodic_block(q, toks, 5, 0.12)
    check("an old episode that is squarely on topic: kept", len(kept) == 1)
    block, kept = router([ep(weak, 100, importance=0.9)])._build_episodic_block(q, toks, 5, 0.12)
    check("an old but important episode: always kept", len(kept) == 1)


def p2_sizes():
    print("\n=== P2. Fewer in the prompt ===")
    import inspect
    sig = inspect.signature(MemoryRouter.retrieve_context)
    check("episodes: 5", sig.parameters["top_k_episodes"].default == 5)
    check("entities: 5", sig.parameters["top_k_entities"].default == 5)
    check("core rules: 8 (the hard ones always come)", sig.parameters["top_k_core"].default == 8)


def p3_tokens():
    print("\n=== P3. The model is asked for a sayable amount ===")
    ash = importlib.import_module("src.py.ash")
    check("default 200 tokens", ash.DEFAULT_LLM_MAX_TOKENS == 200)
    llm = ash.build_llm()
    if llm is None:
        check("(no model backend here)", True)
        return
    check("passed to the model", getattr(llm, "_max_tokens", None) == ash.DEFAULT_LLM_MAX_TOKENS,
          str(getattr(llm, "_max_tokens", None)))
    os.environ["ASH_LLM_MAX_TOKENS"] = "320"
    try:
        importlib.reload(ash)
        check("and ASH_LLM_MAX_TOKENS changes it", ash.DEFAULT_LLM_MAX_TOKENS == 320)
    finally:
        os.environ.pop("ASH_LLM_MAX_TOKENS", None)
        importlib.reload(ash)


if __name__ == "__main__":
    for fn in (p1_stale, p2_sizes, p3_tokens):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
