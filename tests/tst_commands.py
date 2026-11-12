"""
tst_commands.py -- a command said loosely still runs by rule, fast, and never by the model.

  C1  cleaning: the wake word, politeness and filler come off; the command stays
  C2  spelling: words that are not words are put right; real words are never changed
  C3  the nearest command: sure -> run, close -> ask, vague / negated / a question about it -> nothing
  C4  every catalog phrase is taken by the daemon's rules exactly as written (the catalog cannot drift)
  C5  in the runtime: loose forms run the right action (fake executors), with no model call
  C6  "Did you mean ...?" -- yes runs it, no drops it, anything else is a new line
  C7  never for overheard speech; never for "don't ..." or "why did you ..."
  C8  a typo in a question for the brain is passed on spelled right; a real word is left alone
  C9  fast: well under a few milliseconds a line; and the switch turns it off
  C10 weather / calendar / smart home not set up: said plainly, not handed to the model
  C11 an instruction without the name that no rule took is taken in, not answered (a question still is)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_commands.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_intents as ti  # noqa: E402  (fake media/desktop executors; the real ones are trapped)
import tst_wondering as tw  # noqa: E402
from src.daemon.commands import Catalog, CATALOG, Commands, Speller, clean, edit_distance  # noqa: E402

check = tw.check
CM = Commands()


def c1_clean():
    print("\n=== C1. Cleaning ===")
    for said, want in [("Ash, pause the music", "pause the music"), ("pause the music, Ash", "pause the music"),
                       ("hey ash can you please pause the music for me", "pause the music"),
                       ("um next song thanks", "next song"), ("could you open notepad real quick", "open notepad"),
                       ("ok so I need you to turn the volume up please", "turn the volume up"),
                       ("Ash! go to sleep.", "go to sleep"), ("whats the time", "what's the time"),
                       ("can u turn teh volume up pls", "turn the volume up")]:
        check(f"{said!r} -> {want!r}", clean(said) == want, clean(said))
    check("a line with nothing to take off is left as it is", clean("what should I cook tonight") == "what should I cook tonight")
    check("a name that only starts like the wake word is kept", clean("Ashley called") == "Ashley called", clean("Ashley called"))


def c2_spelling():
    print("\n=== C2. Spelling ===")
    for said, want in [("pasue the musci", "pause the music"), ("opne notepad", "open notepad"), ("go to slep", "go to sleep"),
                       ("check my calender", "check my calendar"), ("how's the wether", "how's the weather"),
                       ("tunr the volumme up", "turn the volume up")]:
        got = CM.speller.correct(said)
        check(f"{said!r} -> {want!r}", got == want, got)
    for said in ("what should I cook tonight", "I took the book", "call mom", "the meeting got moved to three",
                 "Omar is my brother", "play some jazz", "open the readme", "who is messi"):
        check(f"real words are never changed: {said!r}", CM.speller.correct(said) == said, CM.speller.correct(said))
    check("two command words equally close: no guess", Speller(["cat", "car"]).word("caz") == "caz")
    check("swapped letters count once", edit_distance("pasue", "pause") == 1 and edit_distance("musci", "music") == 1)
    check("with no word list at all, only what is close to a command word changes",
          Commands(known=()).speller.correct("pasue the musci") == "pause the music")


def c3_nearest():
    print("\n=== C3. The nearest command ===")
    sure = [("pause the music that's playing", "pause the music"), ("start a focus session for 45 minutes", "focus for 45 minutes"),
            ("who is ths", "who is this"), ("the volume turn up", "turn the volume up")]
    for said, want in sure:
        n = CM.nearest(said)
        check(f"sure: {said!r} -> {want!r}", n is not None and n.sure and n.phrase == want, str(n))
    n = CM.nearest("turn the volume up a little bit more")
    check("close but not sure: asked about", n is not None and not n.sure and n.phrase == "turn the volume up", str(n))
    for said in ("don't pause the music", "do not turn the volume up", "why did you pause the music",
                 "did you turn off the mic", "what should I cook tonight", "pass me the salt", "turn the volume",
                 "play that again", "I love this song", "who is messi", "who is the president"):
        check(f"nothing for {said!r}", CM.nearest(said) is None, str(CM.nearest(said)))
    n = CM.nearest("what time is it in Tokyo")
    check("a brain tool is matched as the brain's", n is not None and n.kind == "brain", str(n))


def c4_catalog_is_real():
    print("\n=== C4. Every catalog phrase is a real command, exactly as written ===")
    camera = {"who is this", "look around", "what am I holding"}
    for phrase, kind in CATALOG:
        if kind != "rule":
            continue
        if phrase in camera:
            rt = camera_runtime()
            n0 = len(rt.ash.calls)
            rt.say(phrase)
            check(f"{phrase!r} is taken by the camera", rt.looking.stats["looks"] == 1 and len(rt.ash.calls) == n0,
                  f"{rt.looking.stats} {rt.out[-1:]}")
            continue
        if phrase in OWN_RUNTIME:                    # a feature the standard test runtime does not switch on
            rt = OWN_RUNTIME[phrase]()
            n0 = len(rt.ash.calls)
            rt.say(phrase)
            check(f"{phrase!r} is taken by its feature, no model", len(rt.ash.calls) == n0 and rt.out, f"{rt.out[-1:]}")
            continue
        rt = ti.fresh()                              # each on its own: "go to sleep" must not colour the next one
        rt.cfg.loose_commands = False                # as written, with no help from this module
        rt._commands = None
        who, brain, _v = ti.hear(rt, phrase, force=True)
        check(f"{phrase!r} is taken by a rule ({who})", who is not None and not brain, f"claimed={who} brain={brain}")


def pcdoctor_runtime():
    import tst_effects as te
    import tst_pcdoctor as tp
    return te.rig(pcdoctor=True, pcdoctor_reader=lambda: tp.snap(), loose_commands=False)


OWN_RUNTIME = {"why is my computer slow": pcdoctor_runtime, "what's using my memory": pcdoctor_runtime}


def camera_runtime():
    """The camera with a fake eye and fake models (tst_vision's), and loose matching off."""
    import tst_effects as te
    import tst_vision as tv
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    mem = VisualMemory(None)
    mem.enroll("Omar", "person", tv.vec(1))
    io = dict(eye=Eye(grab=lambda: tv.frame()), faces=tv.FakeFaces([tv.face((200, 100, 160, 200))], [tv.vec(1)]),
              objects=tv.FakeObjects([tv.thing("cup", 0.9)]), memory=mem)
    rt = te.rig(vision=True, vision_io=io, vision_cfg=dict(enrol_checks=False), loose_commands=False)
    return rt


def runtime():
    rt = ti.fresh()
    rt.fake_desktop.procs |= {"notepad.exe"}
    return rt


def say(rt, text, force=False):
    n0 = len(rt.ash.calls)
    rt.claimed.clear()
    keys0, opened0 = len(rt.fake_media.keys), len(rt.fake_desktop.opened)
    who, brain, v = ti.hear(rt, text, force=force)
    return dict(who=who, brain=len(rt.ash.calls) > n0, keys=rt.fake_media.keys[keys0:], opened=rt.fake_desktop.opened[opened0:],
                last=(rt.ash.calls[-1] if len(rt.ash.calls) > n0 else None))


def c5_runtime():
    print("\n=== C5. In the runtime: loose forms run the right thing, with no model call ===")
    rt = runtime()
    cases = [("Ash, pause the music", "media", "play_pause"), ("hey Ash, can you please pause the music for me", "media", "play_pause"),
             ("pause the music, Ash", "media", "play_pause"), ("Ash, pasue the musci", "media", "play_pause"),
             ("Ash, pause the music that's playing", "media", "play_pause"), ("Ash, turn teh volume up pls", "media", "volume_up"),
             ("Ash, next song thanks", "media", "next")]
    for text, who, key in cases:
        r = say(rt, text)
        check(f"{text!r} -> {who} {key}, no model", r["who"] == who and key in r["keys"] and not r["brain"], str(r))
    r = say(rt, "Ash, opne notepad")
    check("'Ash, opne notepad' opens notepad, no model", r["who"] == "desktop" and r["opened"] == ["notepad"] and not r["brain"], str(r))
    r = say(rt, "Ash, can you open notepad for me please")
    check("'Ash, can you open notepad for me please' opens it", r["opened"] == ["notepad"] and not r["brain"], str(r))
    r = say(rt, "Ash, go to slep")
    check("'Ash, go to slep' goes to sleep", r["who"] == "voice_command" and not r["brain"], str(r))
    say(rt, "wake up", force=True)
    r = say(rt, "Ash, start a focus session for 45 minutes")
    check("'start a focus session for 45 minutes' -> focus, for 45", r["who"] == "focus" and not r["brain"]
          and "until" in (rt.out[-1] or ""), str(r) + " " + rt.out[-1])
    r = say(rt, "Ash, can you help me")
    check("'can you help me' is not cleaned into a command 'help me' (only handlers that DO things see a loose line)",
          r["who"] is None and r["brain"], str(r))
    r = say(rt, "Ash, who is messi")
    check("'who is messi' is not the camera's 'who is this'", r["who"] is None and r["brain"], str(r))
    check("the counts say what was done", rt.stats.get("loose_cleaned", 0) >= 4 and rt.stats.get("loose_corrected", 0) >= 2
          and rt.stats.get("loose_nearest", 0) >= 2, str({k: v for k, v in rt.stats.items() if k.startswith("loose")}))


def c6_did_you_mean():
    print("\n=== C6. Did you mean ...? ===")
    rt = runtime()
    r = say(rt, "Ash, turn the volume up a little bit more")
    check("close but not sure: asked, nothing pressed, no model", "Did you mean" in rt.out[-1] and "turn the volume up" in rt.out[-1]
          and not r["keys"] and not r["brain"], rt.out[-1])
    r = say(rt, "yes", force=True)
    check("'yes' runs it", "volume_up" in r["keys"] and not r["brain"], str(r))
    say(rt, "Ash, turn the volume up a little bit more")
    r = say(rt, "no", force=True)
    check("'no' drops it", not r["keys"] and "never mind" in rt.out[-1].lower(), rt.out[-1])
    say(rt, "Ash, turn the volume up a little bit more")
    r = say(rt, "Ash, what's 2 plus 2", force=True)
    check("anything else is a new line (the question is not dropped, nothing pressed)", r["brain"] and not r["keys"], str(r))
    r = say(rt, "yes", force=True)
    check("...and a later 'yes' does not run the old one", not r["keys"], str(r))
    say(rt, "Ash, turn the volume up a little bit more")
    rt.t[0] += 120
    r = say(rt, "yes", force=True)
    check("an old question is not answered by a much later 'yes'", not r["keys"], str(r))


def c7_never_overheard():
    print("\n=== C7. Never for overheard speech, a negation, or a question about a command ===")
    rt = runtime()
    # Lines the address scorer does not count as said to ASH ("pause the music that's playing" is only "probably"
    # for it -- an instruction, no name): never stretched into a command.
    from src.senses import address
    for text in ("pasue the musci", "pause the music that's playing", "opne notepad", "turn teh volume up",
                 "next song thanks"):
        v = address.score(text)
        r = say(rt, text)
        check(f"overheard {text!r} (addressed={v.addressed}): nothing pressed or opened", not v.addressed and not r["keys"]
              and not r["opened"], str(r))
    for text in ("Ash, don't pause the music", "Ash, why did you pause the music", "Ash, did you turn the volume up"):
        r = say(rt, text)
        check(f"{text!r}: nothing pressed", not r["keys"], str(r))
    check("no executor outside the fakes was reached", not ti.REAL_TOUCHES, str(ti.REAL_TOUCHES))


def c8_brain_spelling():
    print("\n=== C8. The brain gets a typo put right, and real words as said ===")
    rt = runtime()
    r = say(rt, "Ash, whats the tiem")
    check("'whats the tiem' reaches the brain as 'what's the time'", r["brain"] and r["last"] == "what's the time", str(r))
    r = say(rt, "Ash, what should I cook tonight")
    check("'what should I cook tonight' reaches it unchanged", r["brain"] and "cook" in (r["last"] or ""), str(r))
    r = say(rt, "Ash, tell me about Omar's new job")
    check("a name is not 'corrected'", r["brain"] and "Omar" in (r["last"] or ""), str(r))


def c9_fast_and_switch():
    print("\n=== C9. Fast; and it can be switched off ===")
    lines = ["Ash, pasue the musci", "hey Ash can you please turn teh volume up for me", "Ash, what should I cook tonight",
             "Ash, pause the music that's playing", "Ash, why did you pause the music"] * 40
    t0 = time.perf_counter()
    for s in lines:
        CM.variants(s)
        CM.nearest(s)
    per = (time.perf_counter() - t0) * 1000 / len(lines)
    check(f"matching a line takes well under 5 ms ({per:.2f} ms)", per < 5.0)
    rt = runtime()
    t0 = time.perf_counter()
    r = say(rt, "Ash, pasue the musci")
    ms = (time.perf_counter() - t0) * 1000
    check(f"a misspelled command is done by rule in the runtime ({ms:.0f} ms), with no model", "play_pause" in r["keys"]
          and not r["brain"] and ms < 500, str(r))
    off = runtime()
    off.cfg.loose_commands = False
    off._commands = None
    r = say(off, "Ash, pasue the musci")
    check("loose_commands off: as before, a misspelled command is not run", not r["keys"], str(r))


def c10_not_set_up():
    print("\n=== C10. Not set up: said plainly, not handed to the model ===")
    rt = runtime()
    for text, who, word in [("Ash, what's the weather like", "weather", "weather.city"),
                            ("Ash, what's on my calendar today", "calendar", "calendar.files"),
                            ("Ash, turn on the living room lights", "smarthome", "smarthome.url")]:
        r = say(rt, text)
        check(f"{text!r} -> {who}: says how to set it up, no model", r["who"] == who and not r["brain"] and word in rt.out[-1],
              f"{r} {rt.out[-1]!r}")
    r = say(rt, "Ash, what's the weather in Paris")
    check("another place still goes to the brain's weather tool", r["brain"], str(r))


def c11_overheard_instructions():
    print("\n=== C11. An instruction without ASH's name that no rule took is taken in, not answered ===")
    for text in ("turn left at the next light", "play that again", "put it on the table"):
        rt = runtime()
        r = say(rt, text)
        check(f"{text!r}: not answered, nothing pressed", not r["brain"] and not r["keys"] and not rt.out,
              f"{r} {rt.out}")
    rt = runtime()
    check("counted", say(rt, "turn left at the next light") and rt.stats.get("instructions_overheard") == 1, str(rt.stats))
    for text in ("what time does the store close", "what time is it"):
        r = say(runtime(), text)
        check(f"a question in the same band is still answered: {text!r}", r["brain"], str(r))
    r = say(runtime(), "Ash, turn left at the next light")
    check("with the name it is answered", r["brain"], str(r))
    rt = runtime()
    r = say(rt, "pause the music")
    check("an instruction a rule does take still works without the name", "play_pause" in r["keys"], str(r))


if __name__ == "__main__":
    for fn in (c1_clean, c2_spelling, c3_nearest, c4_catalog_is_real, c5_runtime, c6_did_you_mean, c7_never_overheard,
               c8_brain_spelling, c9_fast_and_switch, c10_not_set_up, c11_overheard_instructions):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
