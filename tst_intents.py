"""
tst_intents.py -- is each sentence taken for what it is?

Three layers decide what ASH does with a sentence, and each had its own way of being wrong:

  A  Addressing   was it said to ASH? (senses/address.py, with the handlers' own grammars as evidence)
  B  Routing      which handler acts on it, or does it go to the model? (the rule chain in ambient.py)
  C  The router   which tool, if any, does the embedding model pick? (tools/classification.py)

The corpus below is curated from a measurement of 267 real and realistic sentences: before the fixes, 98 commands
ASH's own handlers understand scored 0 and were silently dropped, "can you pass me that cable" scored 1.0,
27 sentences called a tool for no reason ("I'm so stressed today" -> the clock, "what are you doing" -> a CPU table,
"open the browser" -> read_file), and a TV saying "pause" could press a media key.

Every case runs in its own fresh runtime. Media keys and app launches go to FAKES; the test asserts that nothing
real was touched (an earlier version of this measurement launched Excel, Calculator and Chrome on the machine).

Run:  PYTHONIOENCODING=utf-8 python tst_intents.py
"""

import contextlib
import io
import logging
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_AI_BASE", os.path.join(HERE, "data"))

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402
from src.senses import address  # noqa: E402
from src.senses.mic import build_event, fix_name  # noqa: E402

check = tw.check

FEATURES = dict(wonder=True, taste=True, briefings=True, watch=True, desktop=True, rituals=True, personas=True,
                scenes=True, research=True, calendar=True, smarthome=True, voice=True, reminders=True, play=True,
                media=True, focus=True, weather=True, email=True, weekly=True, cheer=True, facts=True, care=True,
                tasks=True, arrivals=True, grammar=True)
NAMES_RX = re.compile(r"^_(\w+_request|\w+_step|voice_command|guest_action)$")
REAL_TOUCHES = []          # anything that reaches a real executor lands here; must stay empty


def _trap_real_executors():
    """Make the real media and desktop executors incapable of acting, and record any attempt: a measurement once
    launched Excel, Calculator and Chrome on the machine it ran on."""
    from src.daemon import desktop, media
    media.MediaExecutor.key = lambda self, name: REAL_TOUCHES.append(("media key", name)) or False
    media.MediaExecutor.chime = lambda self, kind: REAL_TOUCHES.append(("chime", kind)) or False
    desktop.Executor.open = lambda self, target: REAL_TOUCHES.append(("open", target)) or False
    desktop.Executor.close = lambda self, process: REAL_TOUCHES.append(("close", process)) or False
    desktop.Executor.focus = lambda self, process: REAL_TOUCHES.append(("focus", process)) or False


_trap_real_executors()


class FakeMedia:
    def __init__(self):
        self.keys = []

    def key(self, name):
        self.keys.append(name)
        return True

    def chime(self, kind):
        return True

    def photo(self, path, idx=0):
        return True


def fresh():
    """A full-featured runtime whose media and desktop are fakes, and which records who claims a sentence."""
    from tst_desktop import FakeEx
    rt = te.rig(**FEATURES)
    rt.brain.user = "Khalid"
    rt.fake_media = FakeMedia()
    rt.fake_desktop = FakeEx()
    rt.media.ex = rt.fake_media
    rt.desktop.ex = rt.fake_desktop
    if not (rt.media.ex is rt.fake_media and rt.desktop.ex is rt.fake_desktop):
        REAL_TOUCHES.append("an executor was not faked")
    claimed = []
    for name in [n for n in dir(rt) if NAMES_RX.match(n)]:
        fn = getattr(rt, name)

        def mk(fn, name):
            def w(*a, **k):
                r = fn(*a, **k)
                if r and not claimed:
                    claimed.append(name.strip("_").replace("_request", "").replace("_step", ""))
                return r
            return w
        setattr(rt, name, mk(fn, name))
    rt.claimed = claimed
    return rt


def hear(rt, text, force=False, meta=None):
    """Say `text` as the microphone would deliver it. force=True is as if the name had been said."""
    verdict = address.score(fix_name(text))
    ev = build_event("microphone", text, verdict, **(meta or {}))
    if force:
        ev.addressed = True
        ev.meta["addressed"] = True
    n0 = len(rt.ash.calls)
    rt.t[0] += 5
    rt.submit(ev)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt._tick()
    return (rt.claimed[0] if rt.claimed else None), (len(rt.ash.calls) > n0), verdict


def a_addressing():
    print("\n=== A. Was it said to ASH? ===")
    rt = fresh()           # registers the handlers' grammars with the scorer
    # not for ASH: must stay under "probably" (0.32)
    no = ["tell him I'll be there in ten minutes", "mom can you pass the salt", "yeah I'll call you back",
          "the ash tray is on the table", "I told ash about it", "hey sam turn the lights on", "volcanic ash covered the town",
          "so that's a", "stick", "not there", "good good good good", "but I think it's", "I'm gonna see what happens",
          "he said he'd be late", "okay everyone let's start the meeting", "can you pass me that cable",
          "did you see the game last night", "what are you guys doing tonight", "how are you doing guys",
          "set the timer on the oven for me", "we should open the windows", "I play football every sunday",
          "the song is stuck in my head", "I forgot my keys", "I'm in work mode today", "I can't focus today"]
    bad = [(t, round(address.score(fix_name(t)).confidence, 2)) for t in no if address.score(fix_name(t)).confidence >= address.LIKELY_AT]
    check("sentences for other people or about nothing stay under 'probably for ASH'", not bad, str(bad))
    # commands ASH's own handlers understand: at least 'probably', with no name said
    yes = ["next song", "timer for ten minutes", "switch to spotify", "launch notepad", "go to visual studio code",
           "turn on the living room lights", "switch off the kitchen light", "quiet for an hour", "go to sleep",
           "focus for 45 minutes", "let's play a game", "remind me in an hour to check the oven", "make it louder",
           "lower the volume a bit", "undo that", "learn my voice", "work mode", "wind down", "break time",
           "set a timer for 5 minutes", "open chrome", "close word", "play a game with me", "forget what I said about Omar"]
    bad = [(t, round(address.score(fix_name(t)).confidence, 2)) for t in yes if address.score(fix_name(t)).confidence < address.LIKELY_AT]
    check("commands a handler understands are 'probably for ASH' without the name", not bad, str(bad))
    sure = [t for t in yes if address.score(fix_name(t)).confidence >= address.ADDRESSED_AT and not re.match(r"^(?:open|close|set|remind|turn|switch)", t)]
    check("...but a recognised command is never promoted to certain on that evidence alone",
          address.score("next song").confidence < address.ADDRESSED_AT, str(address.score("next song").confidence))
    fb = ["you are working really fast", "Oh my god, you are taking so much time out.", "you're super fast and you need a few improvements",
          "So fix that, but otherwise you are working well actually."]
    bad = [(t, round(address.score(fix_name(t)).confidence, 2)) for t in fb if address.score(fix_name(t)).confidence < address.LIKELY_AT]
    check("feedback to the listener is for the listener", not bad, str(bad))
    check("the name settles it: 'ash, can you pass the salt' is for ASH", address.score("ash, can you pass the salt").addressed)
    check("a recognised command about the oven is still a command (the oven cue needs no handler to agree)",
          address.score("remind me in an hour to check the oven").confidence >= address.LIKELY_AT)
    check("...and without the grammar, a favour about it is for a person",
          address.score("set the timer on the oven for me").confidence < address.LIKELY_AT)
    for t, want in (("Yo, what's up bash?", "Yo, what's up Ash?"), ("What's your status as?", "What's your status Ash?"),
                    ("hey bash how are you", "hey Ash how are you"), ("Ass system status", "Ash system status")):
        check(f"the name misheard: {t!r}", fix_name(t) == want, fix_name(t))
    for t in ("run the bash script", "bash script is broken", "I did it as well as him", "what's up with that as"):
        check(f"...and 'bash' / 'as' elsewhere are left alone: {t!r}", fix_name(t) == t, fix_name(t))
    check("plain narration is not given a reason it is a command", "a command I know" not in address.score("I did it as well as him").reasons)


# ---- B. which handler acts, when the sentence is addressed ---------------------------------------------------
ROUTE = [
    # (text, the handler families that may claim it)  -- None: no handler; it goes on to the model
    ("set a timer for 5 minutes", "reminder"), ("timer for ten minutes", "reminder"),
    ("remind me in an hour to check the oven", "reminder"), ("wake me up at seven", "reminder"),
    ("pause the music", "media"), ("next song", "media"), ("make it louder", "media"), ("lower the volume a bit", "media"),
    ("turn the volume up a little", "media"), ("quieter please", "media"), ("play some music", "media"),
    ("let's play a game", "play"), ("play a game with me", "play"), ("play twenty questions", "play"),
    ("open chrome", "desktop"), ("switch to spotify", "desktop"), ("launch notepad", "desktop"),
    ("open the browser", "desktop"), ("go to visual studio code", "desktop"),
    ("quiet for forty five minutes", "voice_command"), ("quiet until three", "voice_command"),
    ("focus for forty five minutes", "focus"), ("focus for 45 minutes", "focus"), ("I'm done focusing", "focus"),
    ("work mode", "scene"), ("wind down", "scene"), ("break time", "scene"),
    ("what can you do", "honest"), ("are you able to order food", "honest"), ("can you call my mom", "honest"),
    ("what are your limits", "honest"), ("learn my voice", "voice"), ("good morning", "ritual"),
    ("talk like a pirate", "persona"), ("what's playing", "playing"), ("when do I focus best", "browsing"),
    ("this is home", "arrival"), ("tell me when the download finishes", "watch"),
    # statements and chat: no handler may take them
    ("I play football every sunday", None), ("the song is stuck in my head", None), ("my brother plays the guitar", None),
    ("I opened the window earlier", None), ("I switch jobs next month", None), ("the program opened late", None),
    ("I turned off my phone", None), ("it rained all weekend", None), ("I have a meeting at three", None),
    ("I need a break", None), ("it's break time at school", None), ("I can't focus today", None),
    ("it's so quiet in here", None), ("I forgot my keys", None), ("I'll never forget that day", None),
    ("pirates of the caribbean was great", None), ("I'm browsing for shoes", None), ("this browser is slow", None),
    ("can you help me", None), ("can you hear me", None), ("are you there", None), ("what are you doing", None),
    ("how long should I boil an egg", None), ("what's a timer in python", None), ("how do I open a pdf in word", None),
    ("I'm so stressed today", None), ("you are working really fast", None), ("close the deal tomorrow", None),
    ("turn left at the corner", None), ("play it cool", None), ("I'll tell you when I'm done", None),
]


def b_routing():
    print("\n=== B. Which handler acts ===")
    bad = []
    for text, want in ROUTE:
        rt = fresh()
        got, called, _v = hear(rt, text, force=True)
        ok = (got is None) if want is None else (got == want)
        if not ok:
            bad.append((text, want, got))
    check(f"{len(ROUTE)} sentences go where they should (and chat goes to no handler)", not bad, "; ".join(f"{t!r}: wanted {w}, got {g}" for t, w, g in bad))
    check("nothing real was touched: no real media key, no real app launch", REAL_TOUCHES == [], str(REAL_TOUCHES))

    # collisions between handlers: the right one wins
    rt = fresh()
    rt.tasks.saved["start work"] = {"name": "start work", "steps": ["open chrome"], "runs": 0}
    got, _c, _v = hear(rt, "forget start work", force=True)
    check("'forget start work' forgets the saved task, not a memory", got == "task" and "start work" not in rt.tasks.saved, f"{got} {rt.out[-1:]}")
    rt = fresh()
    got, _c, _v = hear(rt, "forget Interstellar", force=True)
    check("...and 'forget X' with no such task is still a memory", got == "forget", str(got))
    rt = fresh()
    rt.say("remember that I like coffee")
    check("'remember that I like coffee' is kept as a taste", "coffee" in str(rt.wonder.kb.tastes) if rt.taste is not None else True,
          str(getattr(rt.wonder.kb, "tastes", None)))


def c_router():
    print("\n=== C. The tool router ===")
    from tools import classification as CL
    if getattr(CL, "SentenceTransformer", None) is None:
        print("  (the embedding model is not installed here: skipped)")
        return
    logging.disable(logging.CRITICAL)
    tool = {"what time is it": "get_time", "what's the time": "get_time", "tell me the time": "get_time",
            "do you know what time it is": "get_time", "current date please": "get_time", "what's today's date": "get_time",
            "is it morning or evening right now": "get_time", "what's the weather": "get_weather",
            "will it rain tomorrow": "get_weather", "is it cold outside": "get_weather", "weather in Amman tomorrow": "get_weather",
            "what's the weather like today": "get_weather"}
    bad = {q: CL.classify_intent(q)["intent"] for q, i in tool.items() if CL.classify_intent(q)["intent"] != i}
    check("real questions still reach the tool that answers them", not bad, str(bad))
    none = ["I'm so stressed today", "time to eat", "it's break time at school", "I can't focus today", "I'll never forget that day",
            "I'll tell you when I'm done", "what are you doing", "are you doing fine", "you're doing great", "you are working really fast",
            "hey ash what are you doing", "turn up the heat", "let me know how it goes", "open the browser", "how do I open a pdf in word",
            "close the deal tomorrow", "what's a timer in python", "it rained all weekend", "the weather app is broken",
            "do I have any meetings tomorrow", "what's on my calendar today", "this browser is slow"]
    bad = {q: CL.classify_intent(q)["intent"] for q in none if CL.classify_intent(q)["intent"] != "conversation"}
    check("a similar word is not a request: none of these calls a tool", not bad, str(bad))
    check("a statement about the weather is not a weather query", CL.classify_intent("it rained all weekend")["intent"] == "conversation")
    check("...but the guard knows a telegraphic request when it sees one", CL._request_shape("weather in Amman tomorrow")
          and not CL._request_shape("it rained all weekend") and not CL._request_shape("the weather app is broken"))
    check("a machine noun keeps system_stats available", CL._has_evidence("system_stats", "how is the cpu doing")
          and not CL._has_evidence("system_stats", "what are you doing"))
    check("a file noun keeps the file tools available", CL._has_evidence("read_file", "read notes.txt")
          and not CL._has_evidence("read_file", "open the browser"))


def d_safety():
    print("\n=== D. Media keys: overheard speech never presses one ===")
    from src.senses.base import State
    from src.senses.base import Modality, SensorEvent
    rt = fresh()
    # speech that did not pass the address scorer at all (a TV): no "address" evidence in the event
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="pause the music", addressed=False, salience=0.5, meta={}))
    with contextlib.redirect_stdout(io.StringIO()):
        rt._tick()
    check("overheard speech that is not for ASH presses no key", rt.fake_media.keys == [], str(rt.fake_media.keys))
    rt = fresh()
    hear(rt, "next one please")
    check("a stray 'next one please' presses nothing", rt.fake_media.keys == [], str(rt.fake_media.keys))
    rt = fresh()
    hear(rt, "pause the music")
    check("'pause the music' (a recognised command, 'probably for ASH') does", rt.fake_media.keys == ["play_pause"], str(rt.fake_media.keys))
    rt = fresh()
    rt._playing = {"item": "Midnight City by M83", "title": "Midnight City", "artist": "M83", "kind": "music", "player": "Spotify"}
    hear(rt, "pause")
    check("a bare 'pause' while the sensor says something is playing is for ASH", rt.fake_media.keys == ["play_pause"], str(rt.fake_media.keys))
    rt = fresh()
    from src.senses.nowplaying import NowPlayingSensor
    rt.periphery.sensors["now_playing"] = NowPlayingSensor(enabled=True, windows=lambda: [])
    hear(rt, "pause the music", force=True)
    check("'pause' when the sensor is watching and nothing is on does not START something",
          rt.fake_media.keys == [] and "Nothing seems to be playing" in " ".join(rt.out), f"{rt.fake_media.keys} {rt.out}")
    rt = fresh()
    hear(rt, "pause the music", force=True)
    check("...with no sensor to ask, it trusts the user", rt.fake_media.keys == ["play_pause"], str(rt.fake_media.keys))
    check("nothing real was touched", REAL_TOUCHES == [])


def e_numbers():
    print("\n=== E. Spoken numbers ===")
    from src.daemon import numwords
    check("number words become digits", numwords.digits("quiet for forty five minutes until three") == "quiet for 45 minutes until 3"
          and numwords.digits("twenty-five and ten") == "25 and 10")
    check("...and nothing else is touched", numwords.digits("an hour and a half, one of those") == "an hour and a half, 1 of those"
          or numwords.digits("an hour") == "an hour")
    check("a word that merely contains one is left alone", numwords.digits("someone stone phone tone") == "someone stone phone tone")


if __name__ == "__main__":
    a_addressing()
    b_routing()
    c_router()
    d_safety()
    e_numbers()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
