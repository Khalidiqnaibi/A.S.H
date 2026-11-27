"""tst_selfcare.py -- merging the two names, searching ASH's memory, "plan my day", "let's reflect", "any tips", "when did I last mention Omar". Offline; the entity
file is a temp copy, never the real one.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_selfcare.py"""
import json
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import selfcare as sc  # noqa: E402
from src.memory.entity.entity_manager import EntityManager  # noqa: E402

check = tw.check


def main():
    print("\n=== C1. Two names, one person ===")
    check("parsed", sc.parse_merge("merge Immortal into Khalid") == ("Immortal", "Khalid") and sc.parse_merge("merge Khalid into Khalid") is None)
    d = tempfile.mkdtemp()
    em = EntityManager(os.path.join(d, "entities.json"))
    em.ingest({"entity_type": "person", "primary_identifiers": {"name": "immortal"}, "attributes": {"canonical_name": "Immortal"}})
    em.ingest({"entity_type": "person", "primary_identifiers": {"name": "omar"}, "attributes": {"canonical_name": "Omar"}})
    n, backup = sc.merge_names(em, "Immortal", "Khalid", os.path.join(d, "backups"))
    names = sorted(e.attributes["canonical_name"] for e in em.store.all())
    check("renamed in memory, the old kept as an alias, one backup before", n == 1 and names == ["Khalid", "Omar"] and os.path.exists(backup), f"{n} {names} {backup}")
    check("the backup is the file as it was", "Immortal" in open(backup, encoding="utf-8").read())
    rt = tm.rig()
    rt.ash.user = "Immortal"
    rt.ash.user_aliases = set()
    rt.ash.set_user_name = lambda nm: (setattr(rt.ash, "user", nm), nm)[1]
    rt.ash.entity_memory = em
    rt.cfg.life_dir = os.path.join(d, "life")
    cfg = os.path.join(d, "daemon.json")
    json.dump({}, open(cfg, "w"))
    rt.config_path = cfg
    em2 = EntityManager(os.path.join(d, "e2.json"))
    rt.ash.entity_memory = em2
    used, said = rt.say("merge Immortal into Khalid")
    check("by voice: the name changes, the old one is an alias, said what happened", not used and rt.ash.user == "Khalid" and "Immortal" in rt.ash.user_aliases
          and said[-1].startswith("Done -- I call you Khalid now, Immortal is kept as an old name"), f"{used} {said}")
    rt2 = tm.rig()
    rt2.ash.user = "Khalid"
    used, said = rt2.say("merge origin into master")
    check("a git merge is not a name merge", not (said and said[-1].startswith("Done -- I call you")), str(said))

    print("\n=== C2. Searching memory ===")
    rt = tm.rig()
    rt.wonder.kb.people["omar"] = {"name": "Omar", "relation": "cousin", "between": {}}
    rt.say("remember that Omar owes me 20 dollars")
    rt.wonder.kb.store_term("Cassandra", "a wide-column database")
    s = sc.memory_search(rt, "Omar")
    check("people and notes, with what each is", "a person: Omar, your cousin" in s and "a note: " in s, s)
    check("terms", "a term: Cassandra is a wide-column database" in sc.memory_search(rt, "Cassandra"))
    check("nothing: said so", sc.memory_search(rt, "Zanzibar") == "Nothing about Zanzibar in what I've kept.")
    used, said = rt.say("search your memory for Omar")
    check("by voice, no model", not used and said[-1].startswith("About Omar:"), f"{used} {said}")

    print("\n=== C3. The plan, the reflection, the tips ===")
    now = time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 08:00", "%Y-%m-%d %H:%M"))
    p = sc.plan([(now + 7200, "Standup")], ["finish chapter 3"], ["call the bank"], ("Branch", 9000.0), now)
    check("fixed things first, then where yesterday stopped", p == "First, Standup at 10:00, then your target: finish chapter 3, then pick up Branch -- you had 2.5 h on it yesterday, and remember: call the bank.", p)
    check("nothing fixed: still says where to start", sc.plan([], [], [], None, now).startswith("Nothing fixed today"))
    r = sc.reflect([("Branch", 7200.0)], ["your interview went well"], ("Branch", 4), 5, 1)
    check("went well / harder / a question back", r == "Went well: 2.0 hours on Branch; 4 days in a row on Branch; your interview went well. Harder: 5 of my answers were slow; "
                                                       "1 time I got something wrong or you were frustrated. What would you change tomorrow?", r)
    rows = [{"source": "active_window", "ts": time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 10:05", "%Y-%m-%d %H:%M")) + i, "meta": {"process": "Code.exe"}} for i in range(5)] + \
           [{"source": "active_window", "ts": time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 22:00", "%Y-%m-%d %H:%M")), "meta": {"process": "opera.exe"}}]
    check("tips from the window record", sc.coach([rows]).startswith("You're most active around 10:00") and "Code.exe" in sc.coach([rows]), sc.coach([rows]))
    used, said = tm.rig().say("plan my day")
    check("'plan my day' in the runtime, no model", not used and said, f"{used} {said}")
    used, said = tm.rig().say("let's reflect on my day")
    check("'let's reflect' in the runtime, no model", not used and said, f"{used} {said}")

    print("\n=== C4. When did I last mention ... ===")
    t1 = time.time() - 3 * 86400
    rows = [("d1", [{"source": "microphone", "ts": t1, "text": "I need to call Omar about the exam"}, {"source": "active_window", "ts": t1 + 5, "text": "Omar - chat"}])]
    s = sc.last_mention(rows, "Omar")
    check("from what you said, not from window titles", s.startswith("You last mentioned Omar on") and "call Omar about the exam" in s, s)
    check("never: said so", sc.last_mention(rows, "Zed").startswith("I don't have you mentioning Zed"))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
