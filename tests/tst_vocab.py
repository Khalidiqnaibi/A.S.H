"""
tst_vocab.py -- the recogniser is told which words to lean towards (names, classes, "port"), locally only.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_vocab.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.senses.mic import MicrophoneSensor  # noqa: E402

check = tw.check


def v1_prompt():
    print("\n=== V1. The prompt ===")
    m = MicrophoneSensor(enabled=False)
    check("with no words, the prompt is what it always was", m.prompt_hint() == "Talking to Ash, my assistant.")
    m.hint_words = lambda: ["Khalid", "Linda", "NoSQL", "linda", "", "x" * 40]
    p = m.prompt_hint()
    check("words are listed, once each, long junk dropped", p == "Talking to Ash, my assistant. Words I use: Khalid, Linda, NoSQL.", p)
    m.hint_words = lambda: [f"Word{i}" for i in range(60)]
    check("at most 25 of them", len(m.prompt_hint().split(":")[1].split(",")) == 25, m.prompt_hint())
    m.hint_words = lambda: 1 / 0
    check("a failing source does not break hearing", m.prompt_hint() == "Talking to Ash, my assistant.")


def v2_runtime():
    print("\n=== V2. From the runtime ===")
    import tst_effects as te
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, knowledge_path=os.path.join(d, "k.json"), wonder_path=os.path.join(d, "w.json"),
                timetable=True, timetable_path=os.path.join(d, "tt.json"))
    rt.brain.user = "Khalid"
    rt.say("remember linda is my sister", source="console")
    rt.say("my NoSQL class is on Sunday from 10 to 11:30", source="console")
    words = rt.vocab_hint()
    check("the user, the people told about, the classes, and the words that were misheard",
          "Khalid" in words and "Linda" in words and "NoSQL" in words and "port" in words, str(words))
    mic = MicrophoneSensor(enabled=False)
    rt.periphery.sensors["microphone"] = mic
    rt._hint_the_microphone()
    check("the microphone is given the source", "Linda" in mic.prompt_hint(), mic.prompt_hint())


def v3_names_from_notes():
    print("\n=== V3. Names in the notes, things that keep coming up, apps ===")
    import tst_effects as te
    from types import SimpleNamespace as NS
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, notes=True, notes_path=os.path.join(d, "n.json"), knowledge_path=os.path.join(d, "k.json"), wonder_path=os.path.join(d, "w.json"))
    rt.brain.user = "Khalid"
    rt.say("remember that Lieth is my friend", source="console")
    rt.say("remember that Baha Natshah helps me with the chip", source="console")
    ent = lambda name, n: NS(attributes={"canonical_name": name, "mentions": n}, primary_identifiers={"name": name.lower()})
    rt.ash.entity_memory = NS(store=NS(entities={"1": ent("Stremio", 9), "2": ent("Once", 1), "3": ent("Opera", 5)}))
    words = rt.vocab_hint()
    check("a friend who is only in the notes is in the hints (it heard 'Lai' and 'life')", "Lieth" in words and "Baha Natshah" in words, str(words))
    check("things that keep coming up are, a one-off is not", "Stremio" in words and "Opera" in words and "Once" not in words, str(words))
    check("most useful first: the user, then the people and notes, long before the fixed words", words[0] == "Khalid" and words.index("Lieth") < words.index("ASH"), str(words))
    check("no duplicates, and the page's ordinary words are not names", len({w.lower() for w in words}) == len(words) and "Remember" not in words, str(words))
    check("a runtime with none of it still gives the fixed words", "ASH" in te.rig().vocab_hint())
    # "Who is Lai?" / "Who is life?" -> Wikipedia; it was Lieth, who is in the notes
    rt.facts = NS(enabled=True, hear=lambda t: "A German municipality. (Wikipedia)")
    for q in ("Who is Lai?", "Who is life?", "who is Leith"):
        rt.say(q, source="console")
        check(f"{q!r} is answered as Lieth, with what was heard said back", rt.out[-1].startswith("Did you mean Lieth?") and "friend" in rt.out[-1] and "Wikipedia" not in rt.out[-1], rt.out[-1])
    rt.say("who is Lieth", source="console")
    check("the exact name is still the people/notes rule's", not rt.out[-1].startswith("Did you mean") and "Wikipedia" not in rt.out[-1], rt.out[-1])
    rt.say("who is Lincoln", source="console")
    check("a real, unrelated name still goes to Wikipedia", "Wikipedia" in rt.out[-1], rt.out[-1])
    rt.say("who is Baha", source="console")
    check("a first name of a two-word name is not a guess about sound", not rt.out[-1].startswith("Did you mean"), rt.out[-1])



if __name__ == "__main__":
    v1_prompt()
    v2_runtime()
    v3_names_from_notes()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
