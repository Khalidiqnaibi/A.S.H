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
                       ("make a note: the plumber comes on Friday", "the plumber comes on Friday")]:
        check(f"{said!r} -> {body!r}", Notes.wants_note(said) == body, str(Notes.wants_note(said)))
    for said in ("remember to call mom", "remember in 10 minutes to check the oven", "remember this", "remember that",
                 "remember me", "do you remember where I parked", "remember when we went to Paris", "what time is it",
                 "I remember the old house", "remember it"):
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
              "is it going to rain", "who is the president"):
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


if __name__ == "__main__":
    for fn in (n1_what_is_a_note, n2_turned_round, n3_finding, n4_never_the_wrong_note, n5_told_and_all, n6_disk_and_forget,
               n7_runtime, n8_not_overheard_guest_stranger, n9_people_and_spelling):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
