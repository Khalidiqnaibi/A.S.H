"""tst_hygiene.py -- what ASH keeps, kept clean: junk is not written (system text, "my own window", window hosts, labels and folders as entities), software is never a person,
a misheard name is mended; an audit reports what is wrong; a clean makes a backup first and only then drops the junk (episodes stay, their links to removed entities go).
All on temp files -- the user's memory is never touched. Offline.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_hygiene.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.daemon import hygiene as hy  # noqa: E402

check = tw.check


def ent(eid, name, label):
    return {"entity_id": eid, "entity_type": label, "primary_identifiers": {"name": name}, "attributes": {"canonical_name": name.title()}}


def make(tmp):
    eps = [
        {"episode_id": "1", "summary": "Immortal: you up ash?", "event_type": "interaction", "related_entities": [], "importance": 0.5},
        {"episode_id": "2", "summary": "[FREE MOMENT 21:04] ASH looked around", "event_type": "interaction", "related_entities": [], "importance": 0.5},
        {"episode_id": "3", "summary": "I'm looking at my own window", "event_type": "interaction", "related_entities": [], "importance": 0.5},
        {"episode_id": "4", "summary": "Khalid spent 12 minutes on ASH (ApplicationFrameHost)", "event_type": "activity", "related_entities": ["e1"], "importance": 0.3},
        {"episode_id": "5", "summary": "Khalid fixed the paddle ocr bug", "event_type": "interaction", "related_entities": ["e1", "e3", "e5"], "importance": 0.8},
    ]
    ents = {"e1": ent("e1", "khalid", "person"), "e2": ent("e2", "paddleocr", "person"), "e3": ent("e3", "users", "person"), "e4": ent("e4", "allama", "org"),
            "e5": ent("e5", "omar", "person"), "e6": ent("e6", "c++", "person")}
    ep, en = os.path.join(tmp, "episodic_memory.jsonl"), os.path.join(tmp, "entities.json")
    with open(ep, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(e) + "\n" for e in eps)
    with open(en, "w", encoding="utf-8") as fh:
        json.dump(ents, fh)
    return ep, en, eps, ents


def main():
    print("\n=== H1. What is not worth keeping ===")
    check("system text and 'my own window' are not events", not hy.keep_episode("[FREE MOMENT 21:04] x") and not hy.keep_episode("I'm looking at my own window") and not hy.keep_episode("spent 5 minutes (ApplicationFrameHost)"))
    check("a person's sentence is kept", hy.keep_episode("Immortal: you up ash?"))
    check("folders and labels are not entities", not hy.keep_entity("Users") and not hy.keep_entity("Fact") and not hy.keep_entity("C:/Users/x/a.txt") and not hy.keep_entity("5"))
    check("a name is kept", hy.keep_entity("Omar") and hy.keep_entity("Hebron"))
    check("software is a tool, never a person", hy.retype("PaddleOCR", "person") == "tool" and hy.retype("C++", "person") == "tool" and hy.retype("Omar", "person") == "person")
    check("a misheard name is mended", hy.fix_name("Allama") == "ollama" and hy.fix_name("Omar") == "Omar")
    check("mojibake is not a name", not hy.keep_entity("Khalid\u00c3\u00a9"))

    print("\n=== H2. Audit and clean ===")
    tmp = tempfile.mkdtemp()
    ep, en, eps, ents = make(tmp)
    a = hy.audit(eps, ents)
    check("the audit counts the junk", a["junk_episodes"] == 3 and "paddleocr (person)" in a["tools_typed_as_things"] and "users" in a["junk_entities"] and a["misheard"] == ["allama"], str(a))
    check("and says it in a line", "3 of 5 episodes" in hy.describe(a), hy.describe(a))
    r = hy.clean(ep, en, apply=False)
    check("a dry run changes nothing", not r["applied"] and len(open(ep, encoding="utf-8").read().splitlines()) == 5 and not [f for f in os.listdir(tmp) if ".bak-" in f])
    r = hy.clean(ep, en, apply=True)
    kept = [json.loads(l) for l in open(ep, encoding="utf-8") if l.strip()]
    ents2 = json.load(open(en, encoding="utf-8"))
    check("a backup of both files was made first", len([f for f in os.listdir(tmp) if ".bak-" in f]) == 2 and len(r["backups"]) == 2)
    check("the junk episodes are gone, the real ones stay", [e["episode_id"] for e in kept] == ["1", "5"], str([e["episode_id"] for e in kept]))
    check("links to a removed entity are dropped, the others stay", kept[1]["related_entities"] == ["e1", "e5"], str(kept[1]))
    check("tools retyped, the junk entity removed, the name mended", ents2["e2"]["entity_type"] == "tool" and ents2["e6"]["entity_type"] == "tool" and "e3" not in ents2
          and ents2["e4"]["primary_identifiers"]["name"] == "ollama", str(ents2.keys()))
    check("a second audit is clean", hy.describe(r["after"]) == "My memory looks clean.", hy.describe(r["after"]))

    print("\n=== H3. The asking ===")
    check("check / clean are told apart", hy.asks("check my memory") == "check" and hy.asks("clean up my memory") == "clean" and hy.asks("tidy your memory") == "clean" and hy.asks("what is in my memory") is None)

    print("\n=== H4. In the runtime (stores are fakes; nothing real is touched) ===")
    import tst_modellast as tm
    from src.memory.episodic.episode_model import Episode

    class Store:
        def __init__(self, path):
            import threading
            self.path, self.lock, self.episodes, self.saved = path, threading.RLock(), {}, 0

        def all(self):
            return list(self.episodes.values())

        def _save_all(self):
            self.saved += 1

    class Index:
        def rebuild(self, eps):
            pass

    class Epi:
        pass

    class Ent:
        def __init__(self, i, name, label):
            self.entity_id, self.entity_type, self.primary_identifiers, self.attributes = i, label, {"name": name}, {"canonical_name": name}

    class EStore:
        def __init__(self, path):
            self.path, self.entities, self.saved = path, {}, 0

        def _save(self):
            self.saved += 1
    tmp = tempfile.mkdtemp()
    ep, en, eps, ents = make(tmp)
    epi = Epi()
    epi.store, epi.index = Store(ep), Index()
    for e in eps:
        o = type("E", (), dict(e))()
        epi.store.episodes[e["episode_id"]] = o
    entm = Epi()
    entm.store = EStore(en)
    for k, v in ents.items():
        entm.store.entities[k] = Ent(k, v["primary_identifiers"]["name"], v["entity_type"])
    rt = tm.rig()
    rt.ash.episodic_memory, rt.ash.entity_memory = epi, entm
    used, said = rt.say("check my memory")
    check("'check my memory' reports, changes nothing", not used and said and "3 of 5 episodes" in said[-1] and len(epi.store.episodes) == 5, f"{used} {said}")
    used, said = rt.say("clean up my memory")
    check("'clean up my memory' says what it will do and asks", not used and said and said[-1].endswith("Go ahead?") and len(epi.store.episodes) == 5, f"{used} {said}")
    used, said = rt.say("yes")
    check("on a yes it cleans, with a backup", not used and said and said[-1].startswith("Done: 3 junk episodes") and len(epi.store.episodes) == 2 and "e3" not in entm.store.entities
          and entm.store.entities["e2"].entity_type == "tool" and len([f for f in os.listdir(tmp) if ".bak-" in f]) == 2, f"{used} {said}")
    rt = tm.rig()
    rt.ash.episodic_memory, rt.ash.entity_memory = epi, entm
    used, said = rt.say("clean up my memory")
    check("when it is clean it says so", not used and said and "looks clean" in said[-1], f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
