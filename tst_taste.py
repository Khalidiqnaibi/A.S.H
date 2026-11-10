"""
tst_taste.py -- what people like and dislike, learned from what is said.

ASH knew who people were and nothing about them: "I love Interstellar" and
"Ali likes jazz when he's sad" went to the model and were gone with the
conversation. taste.py keeps them in the knowledge store, so they are saved,
undone, forgotten and mirrored into entity memory by the code that already does
that for people, answers "what do I like?" by rule, and hands the model the
list only when the question is about choosing something.

  T1  Reading: what a sentence states, and the many that only look like it
  T2  Keeping: scoring, changing a mind, the occasion, who "he" is, who is not a person
  T3  Asking: "what do I like?", "what does Ali hate?", "do I like jazz?"
  T4  Using: the model gets the list for choosing something and for nothing else
  T5  Undo and forget, and entity memory follows
  T6  On the runtime: said once, kept, asked back, never from a guest, never to the model
  T7  The manual teaches it, and the control command shows it
  T8  "This song": what is playing, said and asked about

Run:  PYTHONIOENCODING=utf-8 python tst_taste.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
NOW = 1_800_000_000.0


def kb_with(entities=None, path=None):
    from src.daemon.wondering import Knowledge
    return Knowledge(path, entity_manager=entities, clock=lambda: NOW)


def taste(kb=None, user="Khalid", **cfg):
    from src.daemon.taste import Taste, TasteConfig
    return Taste(kb or kb_with(), user=user, cfg=TasteConfig.from_dict(cfg), clock=lambda: NOW)


def t1_reading():
    print("\n=== T1. Reading a sentence ===")
    from src.daemon.taste import read

    def one(text):
        got = read(text)
        return got[0] if got else None

    for text, item, sign in [
        ("I love Interstellar", "Interstellar", 1),
        ("I can't stand horror films", "horror films", -1),
        ("I really hate cilantro", "cilantro", -1),
        ("I'm obsessed with lo-fi", "lo-fi", 1),
        ("I don't like Mondays", "Mondays", -1),
        ("I never liked jazz", "jazz", -1),
        ("I am not a fan of country music", "country music", -1),
        ("I'm a huge fan of Hans Zimmer", "Hans Zimmer", 1),
        ("My favourite band is Radiohead", "Radiohead", 1),
        ("My favorite movie is Inception", "Inception", 1),
        ("Interstellar is amazing", "Interstellar", 1),
        ("The Boys is so good", "The Boys", 1),
        ("Honestly I love rock and roll", "rock and roll", 1),
        ("I skipped Tenet", "Tenet", -1),
        ("I binged Dark last weekend", "Dark", 1),
        ("I like to code", "code", 1),
    ]:
        f = one(text)
        check(f"reads {text!r}", f is not None and f.item == item and (f.strength > 0) == (sign > 0),
              f"{f.item if f else None} {f.strength if f else ''}")

    check("'love' is stronger than 'like'", one("I love jazz").strength > one("I like jazz").strength)
    check("'kind of' softens it", 0 < one("I kind of like jazz").strength < one("I like jazz").strength)
    check("'really' firms it up", one("I really hate cilantro").strength < one("I hate cilantro").strength)
    check("the kind comes from the words around it",
          one("I can't stand horror films").kind == "movie" and one("I'm obsessed with lo-fi").kind == "music"
          and one("I like pizza").kind == "food" and one("I like to code").kind == "activity")

    both = read("I love Interstellar and I hate Tenet")
    check("two statements in one sentence are both kept",
          [(f.item, f.strength > 0) for f in both] == [("Interstellar", True), ("Tenet", False)], str(both))
    two = read("I like pizza and sushi")
    check("a list is several tastes", [f.item for f in two] == ["pizza", "sushi"])
    check("a title with 'and' in it is one thing", [f.item for f in read("I watch Rick and Morty a lot")] == [])
    check("...kept whole when it is liked", [f.item for f in read("I love Rick and Morty")] == ["Rick and Morty"])

    f = one("I listen to lo-fi while I'm coding")
    check("a habit with its occasion", f.item == "lo-fi" and f.when == "coding" and f.by == "habit", str(f))
    f = one("Ali loves Jaws when he's sad")
    check("...and someone else's, with theirs", f.subject == "Ali" and f.item == "Jaws" and f.when == "sad", str(f))
    check("he and she are recorded as pronouns, for the caller to place",
          one("She hates spicy food").pronoun == "she" and one("She hates spicy food").subject == "")

    for text in ["I hate it when people interrupt", "I like how this works", "I love you ash", "Do you like pizza?",
                 "I like your idea", "I loved the movie", "I think I like it", "He likes it", "the weather is awful",
                 "What do I like?", "I wish I liked jazz", "If I liked jazz I would say so", "Would you like pizza",
                 "I love that", "I hate this", "I like what you did"]:
        check(f"not a taste: {text!r}", read(text) == [], str(read(text)))


def t2_keeping():
    print("\n=== T2. Keeping it ===")
    from src.daemon.taste import Taste

    kb = kb_with()
    t = taste(kb)
    said = t.observe("I love Interstellar", now=NOW)
    check("learned, as a sentence about the user", said == ["The user loves Interstellar."], str(said))
    e = kb.tastes["khalid"]["items"]["interstellar"]
    check("kept under the user, with the score and the kind",
          kb.tastes["khalid"]["user"] is True and e["score"] == 0.9 and e["n"] == 1, str(e))

    t.observe("I really love Interstellar", now=NOW + 1)
    e = kb.tastes["khalid"]["items"]["interstellar"]
    check("saying it again strengthens it, and counts", e["score"] >= 0.95 and e["n"] == 2, str(e))
    t.observe("I hate Interstellar now", now=NOW + 2)
    e = kb.tastes["khalid"]["items"]["interstellar"]
    check("saying the opposite replaces it: people change their minds",
          e["score"] < 0 and e["flips"] == 1, str(e))

    kb = kb_with()
    t = taste(kb)
    t.observe("I like jazz", now=NOW)
    t.observe("I like my jazz music", now=NOW + 1)
    check("a kind learned later is not lost", kb.tastes["khalid"]["items"]["jazz"]["kind"] == "music")

    t.observe("I listen to lo-fi when I'm coding", now=NOW + 2)
    t.observe("I listen to lo-fi while coding", now=NOW + 3)
    lofi = kb.tastes["khalid"]["items"]["lo fi"]
    check("the occasion is counted", lofi["when"] == {"coding": 2}, str(lofi))
    check("...and said back", "coding" in Taste.sentence("Khalid", True, lofi), Taste.sentence("Khalid", True, lofi))

    # someone else
    kb = kb_with()
    t = taste(kb)
    t.observe("Ali likes jazz", now=NOW)
    check("another person has a profile of their own",
          kb.tastes["ali"]["name"] == "Ali" and kb.tastes["ali"]["user"] is False
          and "jazz" in kb.tastes["ali"]["items"] and "jazz" not in kb.tastes.get("khalid", {}).get("items", {}))
    t.observe("He hates cilantro", now=NOW + 1, people=["Ali"])
    check("'he' is whoever was just mentioned", "cilantro" in kb.tastes["ali"]["items"])
    before = sum(len(v["items"]) for v in kb.tastes.values())
    t.observe("She loves pasta", now=NOW + 2, people=[])
    check("'she' with nobody mentioned is not guessed",
          sum(len(v["items"]) for v in kb.tastes.values()) == before)
    check("a guest's voice never writes the profile", taste().observe("I love pizza", guest=True) == [])
    check("off means off", taste(enabled=False).observe("I love pizza") == [])
    kbu = kb_with()
    taste(kbu).observe("Khalid loves Interstellar", now=NOW)
    check("the user's own name as the subject is the user",
          "interstellar" in kbu.tastes.get("khalid", {}).get("items", {}) and list(kbu.tastes) == ["khalid"])
    check("learn_from_others can be turned off", taste(learn_from_others=False).observe("Ali likes jazz") == [])

    # entity memory already knows Netflix and Python are not people
    class Ent:
        def __init__(self):
            from types import SimpleNamespace as NS
            self.store = NS(all=lambda: [NS(entity_id="1", entity_type="org", primary_identifiers={"name": "netflix"},
                                            attributes={"canonical_name": "Netflix"})])

        def ingest(self, d):
            pass
    kb = kb_with(Ent())
    t = taste(kb)
    check("a capitalised word that entity memory has as an organisation is not a person",
          t.observe("Netflix likes to cancel shows", now=NOW) == [] and t.observe("Netflix loves ads", now=NOW) == [])
    check("an unknown name still is", t.observe("Sara loves ads", now=NOW) != [])

    # persistence
    d = tempfile.mkdtemp()
    path = os.path.join(d, "k.json")
    kb = kb_with(path=path)
    taste(kb).observe("I love Interstellar. Ali likes jazz.", now=NOW)
    kb2 = kb_with(path=path)
    check("saved with the rest of what ASH knows, and reloaded",
          "interstellar" in kb2.tastes["khalid"]["items"] and "jazz" in kb2.tastes["ali"]["items"])
    check("the knowledge status counts them", kb2.status()["tastes"] == 2, str(kb2.status()))
    check("a store from before this existed still loads",
          kb_with(path=os.path.join(d, "none.json")).tastes == {})


def t3_asking():
    print("\n=== T3. Being asked ===")
    kb = kb_with()
    t = taste(kb)
    check("nothing yet: says so, and how to tell it",
          "haven't told me" in t.answer("what do I like?") and "I love" in t.answer("what do I like?"))
    for s in ["I love Interstellar", "I like pizza and sushi", "I can't stand horror films", "I don't like Mondays",
              "I'm obsessed with lo-fi", "Ali likes jazz", "He hates cilantro"]:
        t.observe(s, now=NOW, people=["Ali"])

    a = t.answer("what do I like?")
    check("what you like, loves apart from likes", "love Interstellar" in a and "lo-fi" in a and "like pizza" in a
          and "hate" not in a, a)
    a = t.answer("what do I hate?")
    check("what you hate", "horror films" in a and "dislike Mondays" in a and "Interstellar" not in a, a)
    check("by kind", "pizza" in t.answer("what food do I like?") and "Interstellar" not in t.answer("what food do I like?"))
    check("by what you do with it", "lo-fi" in t.answer("what do I like to listen to?"))
    check("'what are my favourites'", "Interstellar" in t.answer("what are my favourites?"))
    check("'what do you know about my taste'", "Interstellar" in t.answer("what do you know about my taste?"))
    a = t.answer("what does Ali like?")
    check("someone else's, in the third person", "Ali likes jazz" in a and "cilantro" not in a, a)
    check("...and what they hate", "Ali hates cilantro" in t.answer("what does Ali hate?"))
    check("someone with no profile", "don't know what Zed likes" in t.answer("what does Zed like?"))
    check("do I like it: yes", t.answer("do I like Interstellar?") == "You love Interstellar.", t.answer("do I like Interstellar?"))
    check("do I like it: no", t.answer("do I like horror films?") == "You hate horror films.", t.answer("do I like horror films?"))
    check("do I like it: never said", "haven't told me how you feel about opera" in t.answer("do I like opera?"))
    for s in ["what time is it", "I love Interstellar", "what do I have on today", "do you like pizza?", "what is jazz"]:
        check(f"not a taste question: {s!r}", t.answer(s) is None)


def t4_using():
    print("\n=== T4. Handing it to the model ===")
    kb = kb_with()
    t = taste(kb)
    for s in ["I love Interstellar", "I love lo-fi", "I can't stand horror films", "I listen to lo-fi while coding",
              "I like pizza", "Ali loves Jaws when he's sad"]:
        t.observe(s, now=NOW)

    c = t.context("what should I watch tonight?", NOW)
    check("choosing something gets the list, framed as background",
          "Interstellar" in c and "horror films" in c and "background only" in c and "never read this list" in c, c)
    check("the occasion goes with it", "while coding" in t.context("what should I listen to?", NOW))
    check("by kind: only music for music", "lo-fi" in t.context("recommend some music", NOW)
          and "pizza" not in t.context("recommend some music", NOW))
    check("someone mentioned brings theirs", "Jaws" in t.context("what should I watch with Ali?", NOW))
    for q in ["what time is it", "open notepad", "how are you", "what's the weather", "set a timer for 5 minutes",
              "tell me about black holes"]:
        check(f"nothing for {q!r}", t.context(q, NOW) == "")
    check("in_answers can be turned off", taste(kb, in_answers=False).context("what should I watch", NOW) == "")
    check("an empty profile adds nothing", taste().context("what should I watch", NOW) == "")


def t5_undo_forget():
    print("\n=== T5. Undo and forget ===")
    from src.daemon.forget import forget_subject, restorer, snapshot
    from src.memory.entity.entity_manager import EntityManager

    d = tempfile.mkdtemp()
    ents = EntityManager(os.path.join(d, "entities.json"))
    kb = kb_with(ents)
    t = taste(kb)
    t.observe("I love Interstellar", now=NOW)

    snap = snapshot(kb)
    t.observe("I hate Tenet", now=NOW + 1)
    check("learned", "tenet" in kb.tastes["khalid"]["items"])
    restorer(kb, snap)()
    check("undo takes back only the last thing",
          "tenet" not in kb.tastes["khalid"]["items"] and "interstellar" in kb.tastes["khalid"]["items"])

    def likes(name):
        for e in ents.store.all():
            if (e.attributes or {}).get("canonical_name", "").lower() == name.lower():
                return e.attributes.get("likes", ""), e.attributes.get("dislikes", "")
        return None
    check("a short likes line is on the person's entity", likes("Khalid") and "Interstellar" in likes("Khalid")[0],
          str(likes("Khalid")))
    t.observe("Ali likes jazz and hates cilantro", now=NOW + 2)

    gone = forget_subject("Interstellar", kb=kb, entities=ents)
    check("forget an item: it is gone from the profile", "interstellar" not in kb.tastes["khalid"]["items"], str(gone))
    check("...and says so", "how it was liked or disliked" in gone, str(gone))
    check("...and off the entity too", "Interstellar" not in (likes("Khalid") or ("", ""))[0], str(likes("Khalid")))
    check("...without touching anyone else's", "jazz" in kb.tastes["ali"]["items"])

    gone = forget_subject("Ali", kb=kb, entities=ents)
    check("forget a person: their tastes go with them", "ali" not in kb.tastes and "what they like" in gone, str(gone))
    check("...and the user's profile is still there", "khalid" in kb.tastes)

    t.observe("I like pizza", now=NOW + 3)
    n = t.forget("")
    check("forget everything, counted", n >= 1 and kb.tastes == {}, str(n))


def t6_runtime():
    print("\n=== T6. On the runtime ===")
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, taste=True, knowledge_path=os.path.join(d, "knowledge.json"),
                wonder_path=os.path.join(d, "wonders.json"))
    rt.brain.user = "Khalid"
    from src.daemon.taste import Taste, TasteConfig
    rt.taste = Taste(rt.wonder.kb, user="Khalid", cfg=TasteConfig(), clock=lambda: rt.t[0])
    seen = []
    orig = rt.ash.think

    def think(query, image=None, context=""):
        seen.append((query, context))
        return orig(query, image=image, context=context)
    rt.ash.think = think

    rt.say("I love Interstellar")
    check("said once, it is kept", "interstellar" in rt.wonder.kb.tastes.get("khalid", {}).get("items", {}),
          str(rt.wonder.kb.tastes))
    check("...and the utterance is still answered, as before", seen and seen[-1][0] == "I love Interstellar", str(seen))
    check("it can be undone, with a label that reads", any("love Interstellar" in lbl for lbl in
                                                          [s[1] for s in rt.undo._steps]), str(rt.undo._steps))

    n = len(seen)
    rt.say("what do I like?")
    check("asked back, it answers by rule", rt.out[-1].startswith("You love Interstellar"), rt.out[-1])
    check("the model was not asked", len(seen) == n, str(seen[n:]))

    rt.say("what should I watch tonight?")
    check("a question about choosing reaches the model with the list as background",
          "Interstellar" in seen[-1][1], str(seen[-1]))
    rt.say("what is the capital of france")
    check("anything else does not carry it", "[Tastes" not in seen[-1][1], str(seen[-1]))

    n = len(rt.wonder.kb.tastes.get("khalid", {}).get("items", {}))
    rt.t[0] += 3
    from src.senses.base import Modality, SensorEvent
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="I love pizza", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice does not write the profile",
          len(rt.wonder.kb.tastes.get("khalid", {}).get("items", {})) == n)
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="what do I like?", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("...and is not told what the user likes", not (rt.out and rt.out[-1].startswith("You love")), rt.out[-1])

    rt.say("who am I?")
    check("'who am I' mentions it", "Interstellar" in rt.out[-1], rt.out[-1])
    rt.say("forget Interstellar")
    check("forgotten by voice", "interstellar" not in rt.wonder.kb.tastes.get("khalid", {}).get("items", {}),
          str(rt.wonder.kb.tastes))
    check("the status reports it", rt.status()["taste"]["items"] == 0, str(rt.status().get("taste")))

    bare = te.rig(wonder=True, knowledge_path=os.path.join(d, "k2.json"), wonder_path=os.path.join(d, "w2.json"))
    check("without the setting there is no taste (a bare runtime behaves as it did)", bare.taste is None)


def t7_manual_and_control():
    print("\n=== T7. The manual and the control command ===")
    from src.daemon import manual
    from src.daemon.service import DEFAULT_CONFIG
    ids = {e[0]: e[2] for e in manual.ENTRIES}
    check("the manual has likes and dislikes, with the phrases that work",
          "taste" in ids and all(p in ids["taste"] for p in ("what do I like", "forget Interstellar", "Undo that")))
    check("...and says what it learns from, and that a guest never adds", "more weakly, from what you play" in ids["taste"]
          and "guest" in ids["taste"])
    check("the screen and what-you-do entries exist", "screen" in ids and "activity" in ids)
    check("the default config has the section, on", DEFAULT_CONFIG["taste"]["enabled"] is True)

    from src.daemon.service import ControlServer
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, taste=True, knowledge_path=os.path.join(d, "k.json"), wonder_path=os.path.join(d, "w.json"))
    rt.brain.user = "Khalid"
    from src.daemon.taste import Taste, TasteConfig
    rt.taste = Taste(rt.wonder.kb, user="Khalid", cfg=TasteConfig(), clock=lambda: rt.t[0])
    rt.taste.observe("I love Interstellar. Ali likes jazz.")
    cs = ControlServer.__new__(ControlServer)
    cs.runtime = rt
    got = cs.dispatch("profile")
    check("ashctl profile shows what it knows", got.get("ok") and "Interstellar" in got.get("report", "")
          and "Ali" in got.get("report", ""), str(got)[:160])
    got = cs.dispatch("profile forget Ali")
    check("...and forgets one person's", got.get("ok") and "ali" not in rt.wonder.kb.tastes, str(got))
    check("...leaving the rest", "khalid" in rt.wonder.kb.tastes)
    bare = te.rig(wonder=True, knowledge_path=os.path.join(d, "k3.json"), wonder_path=os.path.join(d, "w3.json"))
    cs.runtime = bare
    got = cs.dispatch("profile")
    check("...and says so when it is off", got.get("ok") is False and "not enabled" in got.get("error", ""), str(got))


def t8_this_song():
    print("\n=== T8. 'This song': what is playing, said and asked about ===")
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, taste=True, knowledge_path=os.path.join(d, "knowledge.json"),
                wonder_path=os.path.join(d, "wonders.json"))
    rt.brain.user = "Khalid"
    from src.daemon.taste import Taste, TasteConfig
    rt.taste = Taste(rt.wonder.kb, user="Khalid", cfg=TasteConfig(), clock=lambda: rt.t[0])
    from src.senses.base import Modality, SensorEvent

    def play(event, **meta):
        rt.submit(SensorEvent(source="now_playing", modality=Modality.TEXT, text="now playing", salience=0.1,
                              ts=rt.t[0], persistable=False, meta=dict(event=event, **meta)))
        rt._tick()

    def items():
        return rt.wonder.kb.tastes.get("khalid", {}).get("items", {})

    rt.say("what's playing?")
    check("nothing on: says so", rt.out[-1] == "I don't see anything playing.", rt.out[-1])
    rt.say("I love this song")
    check("'this song' with nothing playing is not guessed", not any("this" in k or "zq" in k for k in items()), str(list(items())))

    play("started", player="Spotify", kind="music", title="Midnight City", artist="M83", item="Midnight City by M83")
    for q in ("what's playing?", "what song is this?", "what am I listening to?", "who's singing?"):
        rt.say(q)
        check(f"{q!r} is answered from the player", rt.out[-1] == "That's Midnight City by M83 on Spotify.", rt.out[-1])
    rt.say("what's playing at the cinema?")
    check("a different question is left alone", "Midnight City" not in rt.out[-1], rt.out[-1])

    rt.say("I love this song")
    got = items().get("midnight city by m83")
    check("'I love this song' is kept as what is playing", got and got["score"] >= 0.75 and got["kind"] == "music", str(items()))
    rt.say("what's playing?")
    check("and asking again says you like it", rt.out[-1].endswith("You like it."), rt.out[-1])
    rt.say("do I like this?")
    check("'do I like this?' is about what is playing", rt.out[-1] == "You love Midnight City by M83.", rt.out[-1])

    play("changed", player="Spotify", kind="music", title="Flowers", artist="Miley Cyrus", item="Flowers by Miley Cyrus",
         finished="Midnight City by M83", finished_title="Midnight City", finished_artist="M83", finished_kind="music",
         finished_player="Spotify", finished_s=200, finished_how="changed", finished_skip=False,
         finished_start=rt.t[0] - 200, finished_end=rt.t[0])
    rt.say("I can't stand this")
    got = items().get("flowers by miley cyrus")
    check("it follows the song that is on now", got and got["score"] <= -0.75, str(items()))
    rt.say("what's playing?")
    check("...and says it is not liked", "You don't like it" in rt.out[-1] and "Flowers by Miley Cyrus" in rt.out[-1], rt.out[-1])
    before = dict(items())
    rt.say("I love this place")
    rt.say("I love this idea")
    check("this + something that is not what plays is not bound", set(items()) == set(before) or all(
        "flowers" not in k or v["score"] < 0 for k, v in items().items()), str(items()))

    play("stopped")
    rt.say("what's playing?")
    check("when it stops, nothing is playing", rt.out[-1] == "I don't see anything playing.", rt.out[-1])

    play("started", player="Spotify", kind="music", title="Song X", artist="", item="Song X")
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="what's playing?", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice is not told what you are playing", not rt.out[-1].startswith("That's"), rt.out[-1])


if __name__ == "__main__":
    t1_reading()
    t2_keeping()
    t3_asking()
    t4_using()
    t5_undo_forget()
    t6_runtime()
    t7_manual_and_control()
    t8_this_song()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
