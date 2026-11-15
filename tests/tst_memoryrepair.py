"""
tst_memoryrepair.py -- repairing what the wrong answers left in memory, on a copy, with a backup.

  M1  Looking changes nothing
  M2  --apply: junk readings and this week's one-offs go, an older one-off and the people ASH was told about stay
  M3  The knowledge: a "keep that in mind" taste, a definition with the lost-comma filler, a question closed as answered with no answer

Run:  PYTHONIOENCODING=utf-8 python tests/tst_memoryrepair.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import memory_cleanup as mc  # noqa: E402

check = tw.check
NOW = time.time()


def ent(i, name, age_days, **extra):
    return {"entity_id": f"id{i}", "entity_type": "proper", "primary_identifiers": {"name": name.lower()}, "secondary_identifiers": {},
            "attributes": {"canonical_name": name, "last_seen_context": "x", "source": "chat", "updated_at": NOW - age_days * 86400, **extra}}


def build():
    d = tempfile.mkdtemp()
    ents = {e["entity_id"]: e for e in (
        ent(1, "Streamio", 0.2), ent(2, "Linda Iqnaibi", 0.2), ent(3, "PyTorch", 60), ent(4, "cpu_percent", 0.1), ent(5, "Hebron", 0.1),
        ent(6, "Dimension", 0.5), ent(7, "Omar", 0.2, relation="friend"), ent(8, "Embedding", 0.5))}
    json.dump(ents, open(os.path.join(d, "entities.json"), "w"))
    eps = [{"episode_id": "e1", "timestamp": NOW, "summary": "Immortal: hi", "related_entities": ["id1", "id2", "id3"]},
           {"episode_id": "e2", "timestamp": NOW, "summary": "Immortal: hello Linda Iqnaibi", "related_entities": ["id2", "id6"]},
           {"episode_id": "e3", "timestamp": NOW, "summary": "Immortal: Dimension again", "related_entities": ["id6"]}]
    with open(os.path.join(d, "episodic_memory.jsonl"), "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    os.makedirs(os.path.join(d, "state", "wonder"))
    json.dump({"people": {"linda iqnaibi": {"name": "Linda Iqnaibi", "relation": "sister", "between": {}}}, "roles": {"sister": "Linda Iqnaibi"},
               "terms": {"itzyman": {"term": "@ItzYman", "definition": "man is a friend and muting is something else"}},
               "tastes": {"khalid": {"name": "Khalid", "user": True, "items": {"marvel movies": {}, "shows keep that in mind": {}, "chess": {}}}}},
              open(os.path.join(d, "state", "wonder", "knowledge.json"), "w"))
    json.dump({"wonders": [{"key": "role:father", "status": "answered", "not_before": 5.0, "closed_at": 9.0},
                           {"key": "role:sister", "status": "answered", "not_before": 0, "closed_at": 1}],
               "closed": {"role:father": {"s": "answered", "t": 1}, "role:sister": {"s": "answered", "t": 1}}},
              open(os.path.join(d, "state", "wonder", "wonders.json"), "w"))
    return d


def m1_look():
    print("\n=== M1. Looking ===")
    d = build()
    before = {f: open(os.path.join(d, f), "rb").read() for f in ("entities.json", "episodic_memory.jsonl")}
    p = mc.plan(d)
    check("a plan is made", len(p["drop_entities"]) > 0 and p["knowledge"]["changes"])
    check("nothing is written by looking", all(open(os.path.join(d, f), "rb").read() == b for f, b in before.items()))


def m2_entities():
    print("\n=== M2. Entities ===")
    d = build()
    p = mc.plan(d)
    names = {mc._name(e): why for e, why in p["drop_entities"]}
    check("a reading of the machine goes", "cpu_percent" in names, str(names))
    check("a one-off from the last days goes (seen once, name only)", "Streamio" in names and "Embedding" in names, str(names))
    check("one seen in two sentences stays", "Dimension" not in names, str(names))
    check("an older one-off stays (it may be a real project)", "PyTorch" not in names)
    check("someone ASH was told about stays", "Linda Iqnaibi" not in names)
    check("one with something known about it stays", "Omar" not in names)
    check("the user's places stay", "Hebron" not in names)
    mc.apply(p)
    left = set(json.load(open(os.path.join(d, "entities.json"))))
    check("the file is rewritten without them", "id1" not in left and "id4" not in left and "id8" not in left and "id2" in left and "id3" in left, str(left))
    eps = [json.loads(line) for line in open(os.path.join(d, "episodic_memory.jsonl"))]
    check("no episode points at an entity that is gone", all("id1" not in e["related_entities"] and "id4" not in e["related_entities"] for e in eps)
          and "id2" in eps[0]["related_entities"], str(eps))
    check("a backup was made", any(f.startswith("entities.json.bak-") for f in os.listdir(d)))


def m3_knowledge():
    print("\n=== M3. What ASH knows ===")
    d = build()
    p = mc.plan(d)
    mc.apply(p)
    k = json.load(open(os.path.join(d, "state", "wonder", "knowledge.json")))
    check("the 'keep that in mind' taste is gone, real ones kept", set(k["tastes"]["khalid"]["items"]) == {"marvel movies", "chess"},
          str(k["tastes"]["khalid"]["items"]))
    check("the definition lost the filler", k["terms"]["itzyman"]["definition"] == "a friend and muting is something else", k["terms"]["itzyman"]["definition"])
    w = json.load(open(os.path.join(d, "state", "wonder", "wonders.json")))
    father = [x for x in w["wonders"] if x["key"] == "role:father"][0]
    sister = [x for x in w["wonders"] if x["key"] == "role:sister"][0]
    check("a question closed with no answer on record is asked again", father["status"] == "pending" and "role:father" not in w["closed"], str(father))
    check("one whose answer is on record stays closed", sister["status"] == "answered" and "role:sister" in w["closed"])
    check("backups of both files", any(f.startswith("knowledge.json.bak-") for f in os.listdir(os.path.join(d, "state", "wonder")))
          and any(f.startswith("wonders.json.bak-") for f in os.listdir(os.path.join(d, "state", "wonder"))))
    p2 = mc.plan(d)
    check("run again: nothing left to do", not p2["knowledge"]["changes"] and not p2["drop_entities"], str(p2["knowledge"]["changes"]))


if __name__ == "__main__":
    m1_look()
    m2_entities()
    m3_knowledge()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
