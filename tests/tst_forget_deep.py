"""
tst_forget_deep.py -- "forget Omar" takes the name out of the places that would bring it straight back.

forget_subject cleared Knowledge (people, roles, terms, procedures, tastes), the entity, and the open questions. It left the name in the facts
ASH had noted, in the summaries of remembered moments ("... [mentions: Omar]"), in other entries' context and relations (Sara's entry kept
"relations: {Omar: cousin}"), and in the concept graph's copies of the words -- so retrieving context could surface him again.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_forget_deep.py
"""

import os
import sys
import tempfile
import threading
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


class Store:
    def __init__(self):
        self.entities = {}
        self.saved = 0

    def all(self):
        return list(self.entities.values())

    def _save(self):
        self.saved += 1


def entity(eid, name, **attrs):
    return NS(entity_id=eid, primary_identifiers={"name": name}, attributes=dict(attrs), entity_type="person")


def main():
    from src.daemon.forget import forget_subject, _name_rx, _scrub
    from src.daemon.wondering import Knowledge

    print("\n=== F1. The scrub ===")
    rx = _name_rx({"Omar"})
    check("the name becomes 'someone'", _scrub("I told Omar about it", rx) == "I told someone about it")
    check("a [mentions: ...] tag that names him goes", _scrub("Lunch with the team [mentions: Omar, Sara]", rx) == "Lunch with the team")
    check("a tag that does not is kept", _scrub("Lunch [mentions: Sara]", rx) == "Lunch [mentions: Sara]")
    check("not inside another word", _scrub("an Omarian soup", rx) == "an Omarian soup")

    print("\n=== F2. Everywhere ===")
    kb = Knowledge(None)
    kb.facts["omar birthday"] = "May 3"
    kb.facts["wifi password"] = "hunter2"
    ents = NS(store=Store())
    ents.store.entities = {"e1": entity("e1", "Omar", last_seen_context="Omar likes tea"),
                           "e2": entity("e2", "Sara", last_seen_context="Sara and Omar went out", relations={"Omar": "cousin", "Lina": "friend"})}
    eps = {}
    for i, (summ, rel) in enumerate([("Khalid said Omar is late [mentions: Omar]", ["e1"]), ("Khalid parked on level 3", ["e2"]),
                                      ("Omar and Sara met", ["e1", "e2"])]):
        eps[str(i)] = NS(episode_id=str(i), summary=summ, related_entities=list(rel))
    rebuilt, saved = [], []
    epi = NS(store=NS(lock=threading.RLock(), episodes=eps, _save_all=lambda: saved.append(1)), _rebuild_index=lambda: rebuilt.append(1))
    cand = NS(texts=["call omar tonight", "buy milk"])
    cons = NS(net=NS(concepts={1: NS(seed_text="Omar is coming", exemplars=[(0.9, "Omar is coming"), (0.8, "dinner plans")]), 2: NS(seed_text="weather", exemplars=[])},
                     candidates=NS(items=[cand])))
    gone = forget_subject("Omar", kb=kb, entities=ents, episodic=epi, concepts=cons)
    check("the entity went, with what ASH believed", "the memory entry" in gone and "e1" not in ents.store.entities, str(gone))
    check("the noted fact about him went, the other stayed", "omar birthday" not in kb.facts and kb.facts.get("wifi password") == "hunter2")
    check("his name is out of the remembered moments (and the entity id from them)",
          eps["0"].summary == "Khalid said someone is late" and eps["2"].summary == "someone and Sara met"
          and eps["0"].related_entities == [] and eps["2"].related_entities == ["e2"] and eps["1"].summary == "Khalid parked on level 3", str([e.summary for e in eps.values()]))
    check("...and the file was rewritten and the search index rebuilt", saved and rebuilt)
    check("another entry's context and relations lost him, and kept the rest",
          "Omar" not in ents.store.entities["e2"].attributes["last_seen_context"] and ents.store.entities["e2"].attributes["relations"] == {"Lina": "friend"},
          str(ents.store.entities["e2"].attributes))
    check("the concept graph's copies of the words lost him", cons.net.concepts[1].seed_text == "someone is coming"
          and [e[1] for e in cons.net.concepts[1].exemplars] == ["dinner plans"] and cand.texts == ["buy milk"], str((cons.net.concepts[1].seed_text, cand.texts)))
    check("and it says so", any("remembered moment" in g for g in gone) and any("concept graph" in g for g in gone), str(gone))
    again = forget_subject("Omar", kb=kb, entities=ents, episodic=epi, concepts=cons)
    check("forgetting him again finds nothing more", not any("remembered moment" in g for g in again), str(again))
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
