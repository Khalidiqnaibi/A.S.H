"""
tst_notes.py -- "remember that I parked on level 3" ... "where did I park?", by rule, never the model.

  N1  what counts as a note (and what is a reminder, a person, or too vague)
  N2  said back the right way round: "I parked" -> "you parked", "my sister's" -> "your sister's"
  N3  finding: every word that matters, stems ("parked" = "park"), the newest first, one extra word only for
      "where did I ... the car?"
  N4  never the wrong note: not another person's birthday, not "how do I park a car?", not unrelated questions
  N5  "what did I tell you about X?", "what do you remember?"
  N6  kept on disk, found again after a restart; "forget X" drops it
  N7  in the runtime: kept and answered with no model call; undo; forget; why
  N8  not from overheard speech, not for a guest, held while a stranger is in view
  N9  people still go to remember.py ("remember Lieth is my friend"); a misspelled "remmeber" still works
  N10 "what do you know about me / the user / Khalid" and "who is my friend" answer exactly as "what do you remember" does
  N11 "who is Khalid" (the user) and "who is Lieth" (somebody in the notes) are answered from the notes, not looked up on Wikipedia
  N12 "khalid (me)", "my real name is Khalid" teach ASH another name the person goes by (kept), so N11 works for it

Run:  PYTHONIOENCODING=utf-8 python tests/tst_notes.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.notes import Notes, keys, stem, turn_round  # noqa: E402

check = tw.check
T = 1_800_000_000.0


def notes(path=None):
    n = Notes(path, clock=lambda: T)
    for b in ("I parked on level 3", "the wifi password is hunter2", "I'm allergic to peanuts",
              "my sister's birthday is on May 3rd", "the passports are in the top drawer"):
        n.keep_note(b, T)
    return n


def n1_what_is_a_note():
    print("\n=== N1. What counts as a note ===")
    for said, body in [("Ash, remember that I parked on level 3", "I parked on level 3"),
                       ("note that the wifi password is hunter2", "the wifi password is hunter2"),
                       ("keep in mind that I'm allergic to peanuts", "I'm allergic to peanuts"),
                       ("don't forget the passports are in the top drawer", "the passports are in the top drawer"),
                       ("can you remember my locker number is 42", "my locker number is 42"),
                       ("make a note: the plumber comes on Friday", "the plumber comes on Friday"),
                       ("my favorite number is 4242, remember that", "my favorite number is 4242"),
                       ("the plumber comes on Friday. Please remember that.", "the plumber comes on Friday")]:
        check(f"{said!r} -> {body!r}", Notes.wants_note(said) == body, str(Notes.wants_note(said)))
    for said in ("remember to call mom", "remember in 10 minutes to check the oven", "remember this", "remember that",
                 "remember me", "do you remember where I parked", "remember when we went to Paris", "what time is it",
                 "I remember the old house", "remember it", "what do you remember that", "yes remember that"):
        check(f"not a note: {said!r}", Notes.wants_note(said) is None, str(Notes.wants_note(said)))


def n2_turned_round():
    print("\n=== N2. Said back the right way round ===")
    for said, back in [("I parked on level 3", "you parked on level 3"), ("my sister's birthday is May 3rd", "your sister's birthday is May 3rd"),
                       ("I'm allergic to peanuts", "you're allergic to peanuts"), ("we keep the spare key under the mat", "you keep the spare key under the mat"),
                       ("the wifi password is hunter2", "the wifi password is hunter2")]:
        check(f"{said!r} -> {back!r}", turn_round(said) == back, turn_round(said))
    check("stems: parked = park = parking", stem("parked") == stem("park") == stem("parking") == "park")
    pairs = [("box", "boxes"), ("pass", "passes"), ("glass", "glasses"), ("come", "comes"), ("key", "keys"),
             ("watch", "watches"), ("dish", "dishes"), ("drawer", "drawers"), ("password", "passwords")]
    check("a word and its plural find each other", all(stem(a) == stem(b) for a, b in pairs),
          str([(a, b) for a, b in pairs if stem(a) != stem(b)]))
    check("filler is not a key", keys("where did I put my passport") == ["passport"], str(keys("where did I put my passport")))


def n3_finding():
    print("\n=== N3. Finding ===")
    n = notes()
    for q, want in [("where did I park", "you parked on level 3"), ("Ash, where did I park?", "you parked on level 3"),
                    ("where did I park the car", "you parked on level 3"), ("what's the wifi password", "hunter2"),
                    ("when is my sister's birthday", "May 3rd"), ("am I allergic to anything", "peanuts"),
                    ("where did I put my passport", "top drawer"), ("do you remember where I parked", "level 3"),
                    ("can you remind me where I parked", "level 3")]:
        got = n.recall(q, T)
        check(f"{q!r} -> ...{want}...", got is not None and want in got and got.startswith("You told me"), str(got))
    n.keep_note("I parked on level 5", T + 60)
    check("the newest note wins", "level 5" in (n.recall("where did I park", T + 120) or ""), str(n.recall("where did I park", T + 120)))
    check("and it says when", "this " in (n.recall("where did I park", T + 120) or "") or "(" in (n.recall("where did I park", T + 120) or ""))


def n4_never_the_wrong_note():
    print("\n=== N4. Never the wrong note ===")
    n = notes()
    for q in ("when is my birthday", "when is mom's birthday", "how do I park a car", "what's the weather password",
              "where are my keys", "where is the eiffel tower", "what time is it", "what should I cook tonight",
              "is it going to rain", "who is the president", "what can you do with notes", "can you use my notes"):
        check(f"{q!r} is left to the brain", n.recall(q, T) is None, str(n.recall(q, T)))


def n5_told_and_all():
    print("\n=== N5. What was I told ===")
    n = notes()
    got = n.recall("what did I tell you about the wifi", T)
    check("what did I tell you about the wifi", got == "You told me: the wifi password is hunter2.", str(got))
    got = n.recall("what did I tell you about my car", T)
    check("nothing told: said so (not handed to the model)", got == "You haven't told me anything about your car.", str(got))
    got = n.recall("what do you remember", T)
    check("what do you remember: the last few", got.startswith("You told me:") and "level 3" in got and "peanuts" in got, got)
    check("nothing at all: said so", Notes(None).recall("what do you remember") == "You haven't asked me to remember anything yet.")


def n6_disk_and_forget():
    print("\n=== N6. Kept on disk; forgotten on request ===")
    path = os.path.join(tempfile.mkdtemp(), "notes.json")
    notes(path)
    again = Notes(path, clock=lambda: T)
    check("found again after a restart", "level 3" in (again.recall("where did I park", T) or ""))
    gone = again.forget("where I parked")
    check("forget where I parked", gone == ["you parked on level 3"] and again.recall("where did I park", T) is None, str(gone))
    check("...and that is on disk too", Notes(path).recall("where did I park") is None)
    check("forgetting nothing that matches drops nothing", again.forget("the moon") == [] and len(again.items) == 4)
    check("no stray temp files", not [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp")])


def runtime(**cfg):
    import tst_effects as te
    rt = te.rig(notes=True, notes_path=os.path.join(tempfile.mkdtemp(), "notes.json"), **cfg)
    rt.brain.user = "Khalid"
    return rt


def say(rt, text, **kw):
    n = len(rt.ash.calls)
    k = len(rt.out)
    rt.say(text, **kw)
    return (rt.out[-1] if len(rt.out) > k else None), len(rt.ash.calls) > n


def n7_runtime():
    print("\n=== N7. In the runtime ===")
    rt = runtime()
    out, model = say(rt, "Ash, remember that I parked on level 3")
    check("kept, said back, no model", out == "Noted: you parked on level 3." and not model, f"{out!r} {model}")
    out, model = say(rt, "Ash, where did I park?")
    check("found, no model", out is not None and "level 3" in out and not model, f"{out!r} {model}")
    out, model = say(rt, "what should I cook tonight")
    check("an unrelated question still goes to the model", model, f"{out!r}")
    say(rt, "Ash, remember that the wifi password is hunter2")
    out, _ = say(rt, "undo that")
    check("undo drops the last note", rt.notes.recall("what's the wifi password") is None, f"{out!r}")
    out, model = say(rt, "Ash, forget where I parked")
    check("'forget where I parked' drops it", "level 3" in (out or "") and rt.notes.recall("where did I park") is None and not model, f"{out!r}")
    say(rt, "Ash, remember that the plumber comes on Friday")
    say(rt, "Ash, when does the plumber come")
    out, model = say(rt, "where did you get that")
    check("'where did you get that': from what you asked me to remember", "remember" in (out or "") and not model, f"{out!r}")
    check("status counts them", rt.status()["notes"]["notes"] == 1 and rt.status()["notes"]["kept"] == 3, str(rt.status()["notes"]))


def n8_not_overheard_guest_stranger():
    print("\n=== N8. Not overheard, not a guest, held for a stranger ===")
    from src.senses.base import Modality, SensorEvent
    rt = runtime()
    out, _ = say(rt, "remember that the door code is 1234", addressed=False)
    check("overheard: nothing kept", rt.notes.items == [], f"{out!r}")
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="Ash, remember that the safe code is 9999",
                          addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest: nothing kept", rt.notes.items == [])
    say(rt, "Ash, remember that the safe code is 4321")
    rt._stranger_hold = lambda now: True
    out, model = say(rt, "Ash, what's the safe code")
    check("a stranger in view: the answer is held", out == rt.HOLD_LINE and "4321" not in out and not model, f"{out!r}")


def n9_people_and_spelling():
    print("\n=== N9. People still go to remember.py; a misspelled command still works ===")
    rt = runtime(wonder=True)
    out, model = say(rt, "Ash, remember Lieth is my friend")
    check("'remember Lieth is my friend' is a person, not a note", rt.notes.items == [] and "Lieth" in (out or "") and not model, f"{out!r}")
    out, model = say(rt, "Ash, remmeber that I parked on level 2")
    check("'remmeber that ...' is still kept", len(rt.notes.items) == 1 and "level 2" in (out or "") and not model, f"{out!r}")
    out, model = say(rt, "Ash, wher did I park")
    check("'wher did I park' is still found", "level 2" in (out or "") and not model, f"{out!r}")


def n10_about_me_is_what_you_remember():
    print("\n=== N10. About me, the user, my friend: the same answer as 'what do you remember' ===")
    rt = runtime(wonder=True)
    for s in ("Ash, remember that lieth is a friend", "Ash, remember abd allah sonokut is my best friend",
              "Ash, remember baha natshah is one of my best friends",
              "Ash, im thinking of making a custom chip with a few computer engineer friends remember that"):
        say(rt, s)
    want, model = say(rt, "what do you remember")
    check("what do you remember lists what was kept", want is not None and "baha natshah" in want and "lieth" in want and not model, f"{want!r}")
    for q in ("what do you know about me", "what do you know about khalid", "what do you know about the user", "what do you know about Khalid?",
              "tell me about myself", "who is my friend", "who are my friends"):
        out, model = say(rt, q)
        about_me = q in ("what do you know about me", "tell me about myself")
        # About the person themselves it is who they are and who they know, THEN what they asked to keep (it was the notes alone).
        check(f"{q!r}: the same answer, no model", (out == want or (about_me and out is not None and out.startswith("You're Khalid.") and want in out
                                                                     and "I don't know much else" not in out)) and not model, f"{out!r}")
    out, model = say(rt, "what do you know about lieth")
    check("about somebody else is still its own question", out != want, f"{out!r}")
    out, _ = say(rt, "who am I")
    check("'who am I' still says who the user is", "Khalid" in (out or ""), f"{out!r}")
    rt2 = runtime(wonder=True)
    out, model = say(rt2, "what do you know about me")
    check("with nothing kept it is still answered by the identity rule", out is not None and "Khalid" in out and not model, f"{out!r}")
    n = Notes(None, clock=lambda: T)
    check("a role the people store already names is answered by it", n.about_user("who is my dad", (), lambda r: True) is None)


def n11_who_is_name():
    print("\n=== N11. Who is <the user> / <somebody noted> ===")
    n = Notes(None, clock=lambda: T)
    for b in ("lieth is a friend", "abd allah sonokut is my best friend"):
        n.keep_note(b, T)
    want = n.everything()
    for q in ("who is khalid", "who is Khalid?", "who's immortal", "who is me", "ash, who is the user"):
        check(f"{q!r} (the user) is what was kept", n.about_user(q, ("Khalid", "Immortal")) == want, str(n.about_user(q, ("Khalid", "Immortal"))))
    got = n.about_user("who is lieth", ("Khalid",))
    check("'who is lieth' is what was said about lieth", got == "You told me: lieth is a friend.", str(got))
    for q in ("who is the president", "who is Ada Lovelace", "who is that", "who is it"):
        check(f"{q!r} is left to the lookup", n.about_user(q, ("Khalid",)) is None, str(n.about_user(q, ("Khalid",))))
    check("with nothing kept it is left alone", Notes(None).about_user("who is khalid", ("Khalid",)) is None)


def n12_alias_learning():
    print("\n=== N12. Another name the person goes by ===")
    from src.daemon.identity import alias_in
    for said, name, whole in [("lieth is only a friend of khalid (me) not his best friend", "Khalid", False), ("khalid is me", "Khalid", True),
                              ("my real name is khalid", "Khalid", True), ("I'm also called Khalid", "Khalid", True),
                              ("Ash, my name is also khalid.", "Khalid", True)]:
        check(f"{said!r}", alias_in(said) == (name, whole), str(alias_in(said)))
    for said in ("I'm tired", "my name is the best", "who is khalid", "that is me", "it is me", "call me later", "remember that lieth is a friend"):
        check(f"{said!r} is not a name", alias_in(said) is None, str(alias_in(said)))
    rt = runtime(wonder=True)
    kept = []
    rt.save_aliases = kept.append
    rt.brain.user = "Immortal"
    rt.ash.user_aliases = set()
    for s in ("Ash, remember that lieth is a friend", "Ash, remember abd allah sonokut is my best friend"):
        say(rt, s)
    out, model = say(rt, "who is khalid")
    check("before it is told, 'who is khalid' is not taken for the user (it is left to the lookup)", out != rt.notes.everything(), f"{out!r}")
    out, model = say(rt, "Ash, khalid is me")
    check("told once, answered and kept", out is not None and "Khalid" in out and "Khalid" in rt.ash.user_aliases and kept and kept[-1] == ["Khalid"], f"{out!r} {kept}")
    out, model = say(rt, "who is khalid")
    check("afterwards 'who is khalid' is what was kept, with no model", out == rt.notes.everything() and not model, f"{out!r}")
    rt2 = runtime(wonder=True)
    rt2.brain.user = "Immortal"
    rt2.ash.user_aliases = set()
    out, model = say(rt2, "Ash, remember that lieth is a friend of sami (me)")
    check("a name in passing is learned and the sentence still goes on (here: kept as a note)",
          "Sami" in rt2.ash.user_aliases and any("lieth" in n.said for n in rt2.notes.items), f"{out!r} {rt2.ash.user_aliases}")


def n13_same_person_and_spelling():
    print("\n=== N13. 'Khalid and Immortal are the same person'; 'Khaled' is 'Khalid' ===")
    from src.daemon.identity import same_person_in, sounds_like
    check("the pair is read from a long line", same_person_in("im telling you that khalid and immortal are the same person .. khalid just likes to "
                                                              "be called immortal as a nickname") == ("Khalid", "Immortal"))
    check("'X likes to be called Y'", same_person_in("khalid just likes to be called immortal") == ("Khalid", "Immortal"))
    check("not two things that are the same size", same_person_in("the cat and the dog are the same size") is None)
    for a, b, want in [("khaled", "khalid", True), ("mohamed", "muhammad", True), ("lieth", "leith", True), ("sara", "omar", False), ("khalid", "kind", False)]:
        check(f"{a} ~ {b}: {want}", sounds_like(a, b) == want)
    rt = runtime(wonder=True)
    kept = []
    rt.save_aliases = kept.append
    rt.brain.user = "Immortal"
    rt.ash.user_aliases = set()
    out, model = say(rt, "Ash, im telling you that khalid and immortal are the same person .. khalid just likes to be called immortal as a nickname")
    check("told once, Khalid is kept as the person's name too, with no model", "Khalid" in rt.ash.user_aliases and kept and not model and "Khalid" in (out or ""), f"{out!r}")
    say(rt, "Ash, remember that lieth is a friend")
    out, model = say(rt, "who is Khaled?")
    check("'who is Khaled' (whisper's spelling) is the person, from what was kept -- not Wikipedia", out == rt.notes.everything() and not model, f"{out!r}")


def n14_mom_mother_and_a_bad_file():
    print("\n=== N14. 'mom' is 'mother'; a notes file that will not load is kept, not wiped ===")
    from src.daemon.notes import _ALL
    for t in ("what do you reme", "what do you rem?", "what do you reme.", "what do you remember."):
        check(f"{t!r} (whisper cuts the word short) is 'what do you remember'", bool(_ALL.match(t)))
    check("...but 'what do you recall about it' is not", not _ALL.match("what do you recall about it"))
    n = Notes(None, clock=lambda: T)
    n.keep_note("my mother's birthday is May 3rd", T)
    found = n.find("when is my mom's birthday?")
    check("a note about 'my mother' is found by a question about 'my mom'", len(found) == 1, str([x.said for x in found]))
    check("...and not for another person's", n.find("when is my dad's birthday?") == [])
    p = os.path.join(tempfile.mkdtemp(), "notes.json")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("{ not json")
    n = Notes(p, clock=lambda: T)
    check("a bad file is set aside as .bad", os.path.exists(p + ".bad") and not os.path.exists(p) and n.items == [])
    import json
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"notes": [{"id": 1, "said": "I parked on level 3", "back": "you parked on level 3", "keys": [], "at": T, "colour": "red"}]}, fh)
    n = Notes(p, clock=lambda: T)
    check("a field this version does not know does not lose the notes", len(n.items) == 1 and n.find("where did I park")[0].id == 1)
    n = Notes(None, clock=lambda: T)
    n.keep_note("I like jazz", T)
    check("'what do you know about Khaled?' is the user (Khalid) by sound",
          n.about_user("what do you know about Khaled?", ["Immortal", "Khalid"]) == n.everything())


def n15_pages():
    print("\n=== N15. 'What do you remember' is four short notes, and 'what else' is the next ===")
    n = Notes(None, clock=lambda: T)
    for i, b in enumerate(["you parked on level 3", "lieth is a friend", "a YouTube video is a type of video that is on YouTube that can teach you a lot of things and you watch them every single day to learn", "your dad's clinic is near the hospital", "samsung note is your favorite app", "your sister is Linda"]):
        n.keep_note(b, T + i)
    first = n.recall("what do you remember")
    check("four notes, the newest first, and how many are left", first.count(";") == 3 and "2 more" in first and "what else" in first, first)
    check("a long note is cut short", "..." in first or len(first) < 330, first)
    nxt = n.recall("what else?")
    check("'what else' gives the rest", nxt.startswith("Also: ") and "parked" in nxt and "lieth" in nxt, nxt)
    check("then it says that was everything", n.recall("what else") == "That's everything I've kept.")
    check("'what else' with no list just shown is left alone", n.recall("what else") is None)


if __name__ == "__main__":
    for fn in (n1_what_is_a_note, n2_turned_round, n3_finding, n4_never_the_wrong_note, n5_told_and_all, n6_disk_and_forget,
               n7_runtime, n8_not_overheard_guest_stranger, n9_people_and_spelling, n10_about_me_is_what_you_remember, n11_who_is_name, n12_alias_learning, n13_same_person_and_spelling, n14_mom_mother_and_a_bad_file):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
