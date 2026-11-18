"""
tst_entitygate.py -- a name the tagger finds becomes an entity only once it has recurred.

309 of 362 entities in ASH's memory were one-off capitalised words with nothing but a name ("Dimension", "Embedding", a
heading): nothing to know about them and a lot of noise to search through. Now a mention is a candidate until it has come up in
two different sentences; one that is already an entity is always updated; ASH's own words never create one.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_entitygate.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_MEMORY_DIR", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.memory.memory_router import MemoryRouter  # noqa: E402

check = tw.check


class Ent:
    def __init__(self, name, **payload):
        self.entity_id = f"id-{name}"
        self.primary_identifiers = {"name": name.lower()}
        self.attributes = dict(payload.get("attributes", {}))


class FakeEntities:
    def __init__(self):
        self.items = {}
        self.ingested = []

    def find_matching_entities(self, mentions):
        return [e for m in mentions for e in [self.items.get(m.lower().strip())] if e is not None]

    def ingest(self, payload):
        name = payload["primary_identifiers"]["name"]
        self.ingested.append(name)
        e = self.items.get(name) or Ent(name, attributes=payload["attributes"])
        self.items[name] = e
        return e


class FakeNER:
    def extract(self, text):
        return [{"text": w.strip(".,!?"), "label": "PERSON"} for w in text.split() if w[:1].isupper() and w.lower() not in ("i", "the")]


class FakeEpisodic:
    def add_episode(self, **kw):
        return type("Ep", (), {"episode_id": "e1"})()


def make(**cfg):
    ents = FakeEntities()
    r = MemoryRouter(core_mem=type("C", (), {"add": lambda *a, **k: None})(), entity_mem=ents, episodic_mem=FakeEpisodic(),
                     ner=FakeNER(), config=cfg)
    return r, ents


def g1_gate():
    print("\n=== G1. Seen once is a candidate; seen twice is an entity ===")
    r, ents = make()
    r.route_utterance("The lecturer said Dimension is a keyword.", source="chat")
    check("one mention: no entity", ents.ingested == [] and not ents.items, str(ents.ingested))
    r.route_utterance("The lecturer said Dimension is a keyword.", source="chat")
    check("the same sentence again is still one mention", ents.ingested == [], str(ents.ingested))
    r.route_utterance("Then we compared Dimension with Embedding.", source="chat")
    check("a second, different sentence: now it is kept", "dimension" in ents.items, str(list(ents.items)))
    check("...and the one that was not repeated is not", "embedding" not in ents.items)
    n = len(ents.ingested)
    r.route_utterance("Embedding again.", source="chat")
    r.route_utterance("Is Embedding a vector?", source="chat")
    check("it joins them when it recurs", "embedding" in ents.items)
    r.route_utterance("Dimension is back.", source="chat")
    check("an entity that exists is updated at once, every time", len(ents.ingested) > n and ents.ingested[-1] == "dimension")


def g2_exceptions():
    print("\n=== G2. What the gate leaves alone ===")
    r, ents = make(entity_min_mentions=1)
    r.route_utterance("I met Omar today.", source="chat")
    check("with the gate set to 1 it is as it was", "omar" in ents.items)
    r, ents = make()
    r.route_utterance("Answering Omar properly.", source="ash")
    r.route_utterance("Answering Omar properly again.", source="ash")
    check("ASH's own words never make an entity (as before)", not ents.items)
    r, ents = make(entity_candidate_days=-1.0)
    r.route_utterance("Omar said hi.", source="chat")
    r.route_utterance("Omar said bye.", source="chat")
    check("a mention older than the window does not count", not ents.items, str(ents.items))
    r, ents = make()
    for i in range(2500):
        r.route_utterance(f"One-off Wordnumber{i} here.", source="chat")
    check("the candidate list stays bounded", len(r._candidates) <= 2000, str(len(r._candidates)))
    r, ents = make()
    long = "Zebra " + "word " * 100
    r.route_utterance(long, source="chat")
    r.route_utterance("Zebra again " + "word " * 100, source="chat")
    check("...really", len(ents.items["zebra"].attributes["last_seen_context"]) <= 200)


def g3_names():
    print("\n=== G3. ASH's own name, punctuation, and one entity under several spellings ===")
    r, ents = make(entity_min_mentions=1)
    r.route_utterance("Ash, what do you think", source="chat")
    r.route_utterance("A.S.H is running", source="chat")
    check("ASH's own name is never an entity", ents.ingested == [], str(ents.ingested))
    r.route_utterance("I met Hebron today, Hebron.", source="chat")
    names = {n for n in ents.ingested}
    check("punctuation is not part of a name", all(n == n.strip(" ,.;:!?") for n in names), str(names))
    from src.memory.entity.entity_manager import EntityManager
    em = EntityManager.__new__(EntityManager)
    from types import SimpleNamespace as NS
    mk = lambda nm: NS(primary_identifiers={"name": nm.lower()}, attributes={"canonical_name": nm})
    em.store = NS(all=lambda: [mk("backend & genai developer"), mk("Frameworks"), mk("Samuel")])
    check("'backend/genai developer' finds 'backend & genai developer'", len(em.find_matching_entities(["backend/genai developer"])) == 1)
    check("'frameworks,' finds 'Frameworks'", len(em.find_matching_entities(["frameworks,"])) == 1)
    check("a different name finds nothing", em.find_matching_entities(["Sam"]) == [])
    check("an empty mention finds nothing", em.find_matching_entities(["", "..."]) == [])


if __name__ == "__main__":
    g1_gate()
    g2_exceptions()
    g3_names()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
