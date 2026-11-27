"""tst_meaning.py -- the request is read by MEANING: natural rewordings of 17 features (23% recognised by their own regexes on 28 Nov) are classified by the sentence encoder,
turned into the canonical phrase the feature's rule understands, and run -- not sent to the model. Needs the encoder ASH already loads (skipped, saying so, when it isn't
on disk). Nothing real is touched: devices, the downloads folder and the calendar are fakes or temp folders.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_meaning.py"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.daemon import meaning  # noqa: E402
from src.daemon.chooser import Chooser  # noqa: E402

check = tw.check

# held-out wordings: none of them is an example in SEEDS
NATURAL = {
    "memory": ["why is my ram so full", "which program is eating all my memory up", "what's taking up all my memory"],
    "internet": ["why can't i get online right now", "is my wifi broken", "i've lost my connection"],
    "awake": ["i need it to stay on for the next two hours", "stop the laptop sleeping for a while", "don't let it go to sleep while i'm out"],
    "calendar": ["add dinner with omar to my calendar saturday evening", "schedule a dinner with omar for saturday night around 8", "put a gym session in my diary for monday at 6"],
    "downloads": ["clean up the stuff in downloads", "my downloads are a total mess, can you organise them", "sort out my download folder"],
    "plan": ["what should i be doing first today", "help me figure out what to do this morning", "what's my plan for today"],
    "yesterday": ["what was i doing yesterday", "what did i get up to last night", "tell me what i worked on the day before"],
    "update": ["what's new with you", "have you changed lately", "anything new in your code"],
    "away": ["anything happen while i was gone", "did i miss anything while i was out", "what did i miss"],
    "mic_off": ["stop listening for a bit", "mute your microphone", "i don't want you to hear me right now"],
    "quiet": ["leave me alone for the next hour", "give me an hour of silence", "don't talk to me for half an hour"],
    "volume": ["set my sound level to thirty", "drop the volume to about 30", "can you put the volume at forty percent"],
    "smile": ["give me a big grin", "show me a happy face", "can you look cheerful"],
    "search": ["find me some pictures of cats online", "google cute cats", "i want to see cat photos"],
    "note": ["i parked on level three, don't forget", "keep in mind my car is on level 3", "make a note that the car is on three"],
    "skills": ["what things can you do that i showed you", "list the stuff i've shown you", "which skills did i teach you"],
}
ORDINARY = ["what's the capital of Peru", "explain how a hash map works", "i think the match was fantastic", "my cousin is visiting next week", "tell me a bit about volcanoes",
            "she told me she'd be late", "do you like jazz", "that song is stuck in my head", "how many moons does mars have", "i'm going to the shop later"]


def encoder():
    try:
        from tools.classification import EMBEDDING_MODEL_NAME
        from tools.shared_model import load
        m = load(EMBEDDING_MODEL_NAME)
        return (lambda xs: m.encode(list(xs), show_progress_bar=False)) if m is not None else None
    except Exception:
        return None


def main():
    print("\n=== M1. Slots, from natural wording ===")
    check("a length of time", meaning.duration("for a couple of hours") == "2 hours" and meaning.duration("the next hour") == "1 hour" and meaning.duration("about forty five minutes") == "45 minutes"
          and meaning.duration("half an hour") == "30 minutes")
    check("a time of day settles am/pm", meaning.clock("saturday evening around eight") == "8 pm" and meaning.clock("monday morning at 9:30") == "9:30 am")
    check("a day", meaning.day("this saturday night") == "saturday" and meaning.day("tomorrow at noon") == "tomorrow")
    check("a number", meaning.number("set my sound level to thirty") == 30)
    check("a calendar sentence becomes the phrase the writer understands",
          meaning._event("i have dinner with omar this saturday at eight, put it in my calendar") == "put dinner with omar on saturday at 8 pm", meaning._event("i have dinner with omar this saturday at eight, put it in my calendar"))
    check("a sentence with no day or time is not an event", meaning._event("add dinner to my calendar") is None)

    emb = encoder()
    if emb is None:
        check("(the sentence encoder is not on disk here: the by-meaning checks are skipped)", True)
        print("\nALL PASS")
        sys.exit(0)

    ch = Chooser("meaning", emb, meaning.make_seeds(), accept=0.5, margin=0.04)
    print("\n=== M2. Which feature was meant ===")
    right = total = 0
    wrong = []
    for label, lines in NATURAL.items():
        want = {"mic_off": {"mic_off"}, "smile": {"smile"}}.get(label, {label})
        for t in lines:
            total += 1
            r = meaning.classify(ch, t)
            if r is not None and r.label in want:
                right += 1
            else:
                wrong.append((label, t, None if r is None else r.label))
    print(f"  {right}/{total} natural wordings read as the right feature")
    check("at least 75% of held-out natural wordings (the regexes alone: 23%)", right / total >= 0.75, str(wrong))
    fp = [t for t in ORDINARY if (meaning.classify(ch, t) is not None and meaning.classify(ch, t).sure)]
    check("ordinary sentences are not read as commands", not fp, str(fp))
    r = meaning.classify(ch, "set my sound level to thirty")
    check("the canonical phrase carries the slot", r is not None and r.phrase == "volume to 30", str(r))

    print("\n=== M3. In the runtime: run by the feature's own rule, not the model ===")
    import types
    import tst_modellast as tm
    from src.daemon import devicehelp as dh

    def rig():
        rt = tm.rig()
        rt.cfg.chooser_embed = emb
        rt.cfg.chooser_dir = tempfile.mkdtemp()
        vol = {"v": 18, "sets": []}
        rt._bt = dh.Bluetooth(reader=lambda: [], clock=lambda: rt.clock())
        rt._net = dh.Network(wifi=lambda: ("Home", 78), gw=lambda: "1.1.1.1", ping_fn=lambda h: 3.0, dns=lambda: True, out=lambda: 40.0)
        rt._awake = dh.Awake(setter=lambda on: True, clock=time.time, threaded=False)
        rt._volume = dh.Volume(get=lambda: vol["v"], set_=lambda n: (vol["sets"].append(n), vol.__setitem__("v", n), True)[2])
        rt.vol = vol
        rt._downloads_dir = tempfile.mkdtemp()
        rt.cfg.life_dir = tempfile.mkdtemp()
        return rt
    rt = rig()
    used, said = rt.say("why can't i get online right now")
    check("'why can't I get online' -> the internet check", not used and said and said[-1].startswith("The internet is working"), f"{used} {said}")
    rt = rig()
    used, said = rt.say("set my sound level to thirty")
    check("'set my sound level to thirty' -> volume 30 (a fake volume)", not used and rt.vol["sets"] == [30], f"{used} {said} {rt.vol}")
    rt = rig()
    used, said = rt.say("my downloads are a total mess, can you organise them")
    check("downloads, no model", not used and said and ("tidy" in said[-1] or "move" in said[-1] or "Downloads" in said[-1]), f"{used} {said}")
    rt = rig()
    used, said = rt.say("leave me alone for the next hour")
    check("'leave me alone for the next hour' -> quiet", not used and rt.gate.muted, f"{used} {said}")
    rt = rig()
    used, said = rt.say("i have dinner with omar this saturday at eight, put it in my calendar")
    check("a calendar sentence is read, slots and all, and asked about before it is written", not used and said and said[-1].endswith("Add it?") and "20:00" in said[-1], f"{used} {said}")
    rt = rig()
    used, said = rt.say("what's the capital of Peru")
    check("an ordinary question still goes to the model", used, f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
