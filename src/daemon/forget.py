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

What ASH *believes* from it is what goes: people, relations, terms, noted facts, tastes, the entity -- and the name is taken out of the
places that would bring it straight back (the summaries of remembered moments, other entries' context and relations, the concept graph's
copies of what was said). The moments themselves stay, as "someone". The raw journal is not rewritten (it is swept after a week).
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
        self._steps: List[Tuple[float, str, Callable[[], None], int]] = []
        self._uid = 0
        self.on_push: Optional[Callable[[int, str], None]] = None     # told of each step (the console's timeline)

    def push(self, label: str, fn: Callable[[], None], now: Optional[float] = None) -> int:
        now = self.clock() if now is None else now
        self._uid += 1
        self._steps.append((now, label, fn, self._uid))
        del self._steps[:-self.size]
        if self.on_push is not None:
            try:
                self.on_push(self._uid, label)
            except Exception:
                logger.debug("Undo listener failed", exc_info=True)
        return self._uid

    def undo_id(self, uid: int, now: Optional[float] = None) -> Optional[str]:
        """Undo one particular step (a button on the timeline), not just the last.
        None if it's gone: already undone, too old, or pushed out."""
        now = self.clock() if now is None else now
        for i, (t, label, fn, u) in enumerate(self._steps):
            if u == uid:
                if now - t > self.window_s:
                    return None
                del self._steps[i]
                try:
                    fn()
                except Exception:
                    logger.exception("An undo step failed")
                    return "I couldn't undo that one, sorry."
                return f"Undone: {label.rstrip('.')}."
        return None

    def undo(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        while self._steps and now - self._steps[-1][0] > self.window_s:
            self._steps.pop()
        if not self._steps:
            return "There's nothing recent to undo."
        _t, label, fn, _u = self._steps.pop()
        try:
            fn()
        except Exception:
            logger.exception("An undo step failed")
            return "I couldn't undo that one, sorry."
        return f"Undone: {label.rstrip('.')}."

    def undo_since(self, t: float) -> int:
        """Undo every step pushed at or after `t`, newest first (what a cancelled turn learned). Returns how many."""
        n = 0
        while self._steps and self._steps[-1][0] >= t:
            _t, _label, fn, _u = self._steps.pop()
            try:
                fn()
                n += 1
            except Exception:
                logger.exception("An undo step failed")
        return n

    def __len__(self):
        return len(self._steps)


# ----------------------------------------------------------------------------
# Knowledge snapshots, for undoing what wondering learned
# ----------------------------------------------------------------------------
def snapshot(kb) -> Dict[str, Any]:
    import copy
    with kb._lock:
        return copy.deepcopy({k: getattr(kb, k) for k in ("people", "roles", "terms", "facts", "aliases",
                                                           "procedures", "tastes") if hasattr(kb, k)})


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


def _name_rx(names) -> "re.Pattern":
    parts = sorted({str(x).strip() for x in names if x and len(str(x).strip()) >= 2}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in parts) + r")(?!\w)", re.I) if parts else re.compile(r"(?!x)x")


def _scrub(text: str, rx) -> str:
    """`text` with the name taken out: a "[mentions: ...]" tag that names it goes, and the name itself becomes "someone"."""
    if not text or not rx.search(text):
        return text
    text = re.sub(r"\s*\[mentions:[^\]]*\]", lambda m: "" if rx.search(m.group(0)) else m.group(0), text)
    return rx.sub("someone", text).strip()


def scrub_memories(rx, removed_ids: set, entities=None, episodic=None, concepts=None) -> List[str]:
    """The name out of the places that would bring it back: remembered moments, other entries, the concept graph's copies of the words."""
    gone: List[str] = []
    if episodic is not None:
        try:
            changed = 0
            with episodic.store.lock:
                for ep in list(episodic.store.episodes.values()):
                    new = _scrub(ep.summary, rx)
                    keep = [e for e in (ep.related_entities or []) if e not in removed_ids and not rx.search(str(e))]
                    if new != ep.summary or keep != list(ep.related_entities or []):
                        ep.summary, ep.related_entities = new, keep
                        changed += 1
                if changed:
                    episodic.store._save_all()
                    episodic._rebuild_index()           # the vectors were made from the old words
            if changed:
                gone.append(f"the name in {changed} remembered moment{'s' if changed != 1 else ''}")
        except Exception:
            logger.exception("Could not scrub episodes while forgetting")
    if entities is not None:
        try:
            changed = 0
            for e in list(entities.store.all()):
                attrs = e.attributes if isinstance(getattr(e, "attributes", None), dict) else None
                if not attrs:
                    continue
                hit = False
                for k in ("last_seen_context", "summary", "notes"):
                    if isinstance(attrs.get(k), str) and rx.search(attrs[k]):
                        attrs[k] = _scrub(attrs[k], rx)
                        hit = True
                rel = attrs.get("relations")
                if isinstance(rel, dict):
                    for k in [k for k in rel if rx.search(str(k))]:
                        del rel[k]
                        hit = True
                changed += int(hit)
            if changed:
                entities.store._save()
                gone.append(f"the name in {changed} other entr{'ies' if changed != 1 else 'y'}")
        except Exception:
            logger.exception("Could not scrub entities while forgetting")
    net = getattr(concepts, "net", None) if concepts is not None else None
    if net is not None:
        try:
            changed = 0
            for c in list(net.concepts.values()):
                if rx.search(getattr(c, "seed_text", "") or ""):
                    c.seed_text = _scrub(c.seed_text, rx)
                    changed += 1
                ex = [e for e in (getattr(c, "exemplars", None) or []) if not rx.search(str(e[1]))]
                if len(ex) != len(getattr(c, "exemplars", None) or []):
                    c.exemplars = ex
                    changed += 1
            cand = getattr(net, "candidates", None)
            if cand is not None:
                for k in list(cand.items):
                    keep = [t for t in (k.texts or []) if not rx.search(str(t))]
                    if len(keep) != len(k.texts or []):
                        k.texts = keep
                        changed += 1
            if changed:
                gone.append("what the concept graph kept of it")
        except Exception:
            logger.exception("Could not scrub the concept graph while forgetting")
    return gone


def forget_subject(name: str, kb=None, entities=None, wonderer=None, now: Optional[float] = None,
                   episodic=None, concepts=None) -> List[str]:
    """Remove `name` from everything ASH believes. Returns what went, as words."""
    from .wondering import norm
    n = norm(name)
    gone: List[str] = []
    if not n:
        return gone
    names = {str(name).strip()}
    removed_ids: set = set()
    felt: List[str] = []
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
            facts = getattr(kb, "facts", None)
            if facts:
                for k in [k for k, v in facts.items() if n in norm(k).split() or n == norm(k) or n == norm(str(v))]:
                    del facts[k]
                    if "what it noted about them" not in gone:
                        gone.append("what it noted about them")
            tastes = getattr(kb, "tastes", None)
            if tastes:
                # Both ways: "forget Omar" drops what Omar likes, and
                # "forget Interstellar" drops how anyone felt about it.
                if n in tastes:
                    del tastes[n]
                    gone.append("what they like")
                felt = []
                for k, t in tastes.items():
                    if n in (t.get("items") or {}):
                        del t["items"][n]
                        felt.append(k)
                if felt:
                    gone.append("how it was liked or disliked")
            kb.save()
        if tastes and felt:
            try:
                from .taste import mirror
                mirror(kb, felt)
            except Exception:
                logger.debug("Could not refresh tastes on entities", exc_info=True)
    if entities is not None:
        try:
            for e in list(entities.store.all()):
                names_of_e = [str(v) for v in (e.primary_identifiers.get("name"), (e.attributes or {}).get("canonical_name"),
                                               (e.attributes or {}).get("full_name")) if v]
                if any(norm(x) == n for x in names_of_e) and remove_entity(entities, e.entity_id):
                    removed_ids.add(e.entity_id)
                    names.update(names_of_e)
                    if "the memory entry" not in gone:
                        gone.append("the memory entry")
        except Exception:
            logger.exception("Could not forget from entity memory")
    gone.extend(scrub_memories(_name_rx(names), removed_ids, entities=entities, episodic=episodic, concepts=concepts))
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
