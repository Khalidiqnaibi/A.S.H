"""
src/concepts/gist.py

Replay compiles episodes into the concepts: when the same entity keeps turning up in the episodes a concept was active for
("clinic", "Hebron", "my dad" -> Afif), the concept learns it, so activating the concept brings the entity up without any episode
having to be found first. The episodes stay as they are; this is the slow store learning what the fast one keeps seeing.
"""

from typing import Any, Dict, Iterable


def compile_gist(net, episodes: Iterable[Any], min_episodes: int = 3, share: float = 0.6, keep: int = 4) -> int:
    """Set `entities` on every concept that has at least `min_episodes` episodes: the entity ids present in at least `share` of
    them (and at least two). Returns how many concepts changed."""
    by_concept: Dict[int, list] = {}
    for ep in episodes:
        for cid in (getattr(ep, "concepts", None) or []):
            by_concept.setdefault(int(cid), []).append(ep)
    changed = 0
    for cid, eps in by_concept.items():
        c = net.concepts.get(cid)
        if c is None or len(eps) < min_episodes:
            continue
        count: Dict[str, int] = {}
        for ep in eps:
            for eid in set(getattr(ep, "related_entities", None) or []):
                count[str(eid)] = count.get(str(eid), 0) + 1
        need = max(2, int(share * len(eps) + 0.999))
        got = {e: n for e, n in sorted(count.items(), key=lambda t: -t[1]) if n >= need}
        got = dict(list(got.items())[:keep])
        if got != c.entities:
            c.entities = got
            changed += 1
    return changed


def entities_of(net, concept_ids: Iterable[int]) -> list:
    """The entity ids the given concepts have learned, strongest first, without repeats."""
    seen: Dict[str, int] = {}
    for cid in concept_ids:
        c = net.concepts.get(int(cid))
        for e, n in (getattr(c, "entities", None) or {}).items():
            seen[e] = max(seen.get(e, 0), n)
    return [e for e, _n in sorted(seen.items(), key=lambda t: -t[1])]
