"""
src/brain/memory_engine.py

Episodic / Semantic Memory Engine  (bottom-right box).

ASH already had a good multi-store memory system: CoreMemoryEngine (semantic
constraints), EntityManager (semantic entities), EpisodicMemory (timeline),
all fronted by MemoryRouter. This module does not reimplement any of it.

What it adds is the thing a brain has and a database doesn't: **working
memory**, and **retrieval depth that scales with the pathway**.

The old runtime called `retrieve_context()` on every single turn at full
depth, then handed the whole block to the LLM. That is the memory equivalent
of reading your entire diary before answering "what time is it". Here:

    REFLEX  -> no retrieval at all. Working memory only.
    FAST    -> working memory + a shallow, cheap probe (top-2, core only).
    SLOW    -> full three-store retrieval at configured depth.

Working memory is a short, decaying buffer of the last few turns plus
whatever was retrieved recently. It's what makes "and what about tomorrow?"
resolvable on the fast path without touching the stores.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from .signals import Pathway

logger = logging.getLogger("ash.brain.memory")

WM_CAPACITY = 8
WM_TTL = 600.0   # seconds before a working-memory slot decays out


@dataclass
class WorkingSlot:
    role: str          # "user" | "ash" | "tool" | "retrieved"
    text: str
    ts: float
    salience: float = 0.5

    def age(self) -> float:
        return time.time() - self.ts

    def active(self) -> bool:
        return self.age() < WM_TTL


class MemoryEngine:
    """Adapter over MemoryRouter + a working-memory buffer."""

    def __init__(self, router, capacity: int = WM_CAPACITY):
        self.router = router
        self.wm: Deque[WorkingSlot] = deque(maxlen=capacity)
        self._last_context: Dict[str, str] = {"core": "", "episodic": "", "entity": ""}
        self._last_context_ts = 0.0

    # ------------------------------------------------------------------
    # Working memory
    # ------------------------------------------------------------------
    def push(self, role: str, text: str, salience: float = 0.5):
        if not text:
            return
        self.wm.append(WorkingSlot(role=role, text=str(text), ts=time.time(), salience=salience))

    def working_block(self, max_chars: int = 900) -> str:
        parts: List[str] = []
        for slot in self.wm:
            if not slot.active():
                continue
            parts.append(f"{slot.role}: {slot.text}")
        block = "\n".join(parts)
        return block[-max_chars:] if len(block) > max_chars else block

    def clear_working(self):
        self.wm.clear()

    # ------------------------------------------------------------------
    # Retrieval, depth-tiered by pathway
    # ------------------------------------------------------------------
    def retrieve(self, query: str, pathway: Pathway) -> Dict[str, str]:
        if pathway == Pathway.REFLEX:
            return {"core": "", "episodic": "", "entity": ""}

        if pathway == Pathway.FAST:
            # Shallow probe: hard core rules are the only thing that can
            # change a fast answer's correctness, so that's all we pay for.
            try:
                ctx = self.router.retrieve_context(
                    query, top_k_core=2, top_k_episodes=0, top_k_entities=0
                )
            except Exception:
                logger.exception("Shallow retrieval failed")
                return {"core": "", "episodic": "", "entity": ""}
            return {"core": ctx.get("core", ""), "episodic": "", "entity": ""}

        # SLOW / ESCALATED: full depth.
        try:
            ctx = self.router.retrieve_context(query)
        except Exception:
            logger.exception("Full retrieval failed")
            ctx = {"core": "", "episodic": "", "entity": ""}

        self._last_context = ctx
        self._last_context_ts = time.time()
        for k, v in ctx.items():
            if v:
                self.push("retrieved", f"[{k}] {v[:200]}", salience=0.4)
        return ctx

    @staticmethod
    def hits(ctx: Dict[str, str]) -> int:
        return sum(1 for v in ctx.values() if v)

    # ------------------------------------------------------------------
    # Write path -- unchanged semantics, just funneled through here so the
    # brain has one place that touches persistent memory.
    # ------------------------------------------------------------------
    def commit_utterance(self, text: str, source: str = "chat",
                         importance: float = 0.5, actor: Optional[str] = None) -> Dict[str, Any]:
        try:
            return self.router.route_utterance(
                text=text, source=source, importance=importance, actor=actor
            )
        except Exception:
            logger.exception("route_utterance failed")
            return {}

    def commit_turn(self, user_text: str, ash_text: str, user_actor: str,
                    ash_actor: str, importance: float = 0.5,
                    pathway: Pathway = Pathway.SLOW) -> Dict[str, Any]:
        """Write both halves of a turn.

        Reflex turns are deliberately NOT persisted to episodic memory -- a
        cached repeat of "what time is it" is not an episode worth
        remembering, and writing it would flood the timeline with noise. The
        working buffer still sees it.
        """
        self.push("user", user_text, salience=0.6)
        self.push("ash", ash_text, salience=0.5)

        if pathway == Pathway.REFLEX:
            return {"skipped": "reflex turn not persisted"}

        # Slow-path turns are, by construction, the ones that required real
        # thought -- weight them a little heavier in the timeline.
        imp = importance + (0.15 if pathway in (Pathway.SLOW, Pathway.ESCALATED) else 0.0)
        out = self.commit_utterance(user_text, source="chat", importance=min(1.0, imp), actor=user_actor)
        self.commit_utterance(ash_text, source="ash", importance=min(1.0, imp), actor=ash_actor)
        return out
