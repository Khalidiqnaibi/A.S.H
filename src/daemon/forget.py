"""
src/daemon/forget.py

"Undo that" and "forget what I said about Omar".

ASH now learns and sets up a lot on its own -- a relation from a passing
sentence, a definition, a routine, a follow-up, a reminder, a watch, a
correction to its routing. Each of those leaves an undo step here, with a
plain label, and:

    "undo that" / "scratch that" / "forget that"
        reverses the most recent one, if it was in the last ten minutes, and
        says what it undid ("Undone: Omar is your cousin.")
    "forget what I said about Omar" / "forget Omar" / "forget ZXQ"
        removes that subject from what ASH knows -- people, relations between
        them, terms, routines, entity memory -- and closes the questions about
        it, so it is not asked straight back

Conversation history and the journal are not rewritten: what was said was
said. What ASH *believes* from it is what goes.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.forget")

_UNDO = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:undo(?: that| it| the last (?:one|thing))?|"
                   r"scratch that|forget that|never ?mind that|take that back|cancel that)\s*[.!]?\s*$", re.I)
_FORGET = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?forget\s+(?:what i (?:said|told you)\s+about\s+|"
                     r"(?:everything|all)\s+(?:i said\s+)?about\s+|about\s+)?(?:my\s+|the\s+)?"
                     r"(?P<s>[\w .'\-]{2,40}?)\s*[.!]?\s*$", re.I)
_NOT_SUBJECTS = frozenset("it that this everything all me you my voice them him her".split())


def wants_undo(text: str) -> bool:
    return bool(_UNDO.match(text or ""))


def wants_forget(text: str) -> Optional[str]:
    """The subject of "forget X", or None. "Forget my voice" and "forget it"
    are someone else's business."""
    m = _FORGET.match(text or "")
    if not m:
        return None
    s = m.group("s").strip(" .")
    if not s or s.lower() in _NOT_SUBJECTS or s.lower().endswith(" voice") or len(s.split()) > 5:
        return None
    return s


class UndoStack:
    def __init__(self, window_s: float = 600.0, clock: Callable[[], float] = time.time, size: int = 20):
        self.window_s = window_s
        self.clock = clock
        self.size = size
        self._steps: List[Tuple[float, str, Callable[[], None]]] = []

    def push(self, label: str, fn: Callable[[], None], now: Optional[float] = None):
        now = self.clock() if now is None else now
        self._steps.append((now, label, fn))
        del self._steps[:-self.size]

    def undo(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        while self._steps and now - self._steps[-1][0] > self.window_s:
            self._steps.pop()
        if not self._steps:
            return "There's nothing recent to undo."
        _t, label, fn = self._steps.pop()
        try:
            fn()
        except Exception:
            logger.exception("An undo step failed")
            return "I couldn't undo that one, sorry."
        return f"Undone: {label.rstrip('.')}."

    def __len__(self):
        return len(self._steps)


# ----------------------------------------------------------------------------
# Knowledge snapshots, for undoing what wondering learned
# ----------------------------------------------------------------------------
def snapshot(kb) -> Dict[str, Any]:
    import copy
    with kb._lock:
        return copy.deepcopy({k: getattr(kb, k) for k in ("people", "roles", "terms", "facts", "aliases",
                                                           "procedures")})


def entity_ids(entities) -> set:
    try:
        return {e.entity_id for e in entities.store.all()} if entities is not None else set()
    except Exception:
        return set()


def restorer(kb, snap: Dict[str, Any], entities=None, before_ids: Optional[set] = None) -> Callable[[], None]:
    """An undo step: put Knowledge back as it was, and drop entities the
    learning created (ones that existed before are left alone)."""
    def run():
        with kb._lock:
            for k, v in snap.items():
                setattr(kb, k, v)
            kb.save()
        if entities is not None and before_ids is not None:
            for eid in entity_ids(entities) - before_ids:
                remove_entity(entities, eid)
    return run


def remove_entity(entities, eid: str) -> bool:
    try:
        st = entities.store
        if eid in st.entities:
            del st.entities[eid]
            st._save()
            return True
    except Exception:
        logger.exception("Could not remove an entity")
    return False


def forget_subject(name: str, kb=None, entities=None, wonderer=None, now: Optional[float] = None) -> List[str]:
    """Remove `name` from everything ASH believes. Returns what went, as words."""
    from .wondering import norm
    n = norm(name)
    gone: List[str] = []
    if not n:
        return gone
    if kb is not None:
        with kb._lock:
            if n in kb.people:
                del kb.people[n]
                gone.append("who they are")
            for p in kb.people.values():
                if n in (p.get("between") or {}):
                    del p["between"][n]
                    if "how they know others" not in gone:
                        gone.append("how they know others")
            for role in [r for r, who in kb.roles.items() if norm(who) == n]:
                del kb.roles[role]
            for bare in [b for b, full in kb.aliases.items() if b == n or norm(full) == n]:
                del kb.aliases[bare]
            if n in kb.terms:
                del kb.terms[n]
                gone.append("what it means")
            if n in kb.procedures:
                del kb.procedures[n]
                gone.append("the routine")
            kb.save()
    if entities is not None:
        try:
            for e in list(entities.store.all()):
                names = [str(v) for v in (e.primary_identifiers.get("name"), (e.attributes or {}).get("canonical_name"),
                                         (e.attributes or {}).get("full_name")) if v]
                if any(norm(x) == n for x in names) and remove_entity(entities, e.entity_id):
                    if "the memory entry" not in gone:
                        gone.append("the memory entry")
        except Exception:
            logger.exception("Could not forget from entity memory")
    if wonderer is not None:
        try:
            now = wonderer.clock() if now is None else now
            with wonderer._lock:
                for w in list(wonderer.wonders.values()):
                    if w.status in ("pending", "asked") and (n in w.subjects or n == norm(w.subject)):
                        wonderer._close(w, "dismissed", now)
                        if "the questions about it" not in gone:
                            gone.append("the questions about it")
                wonderer._touch()
        except Exception:
            logger.exception("Could not close questions while forgetting")
    return gone
