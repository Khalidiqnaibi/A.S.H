"""
tst_speech.py -- interruptible speech.

  P1  Hearing "stop": while ASH talks, a plain "stop", "shh", "enough"; when
      it is quiet, only words that can mean nothing else; never a sentence
      that merely has "stop" in it
  P2  "The short version" and "keep it short": the phrases, and shortening
  P3  The preference and the last answer: brief mode cuts long answers only,
      kept across a restart; the short version of what was just said
  P4  The speaker: a new line replaces the old one, stop() cuts it off
      mid-sound, says whether anything was playing
  P5  The runtime: "stop" while speaking cuts it and says nothing back;
      "stop" while quiet goes on to the rest; "the short version"; brief
      mode speaks short and shows it all; the console's speaking cue

Run:  PYTHONIOENCODING=utf-8 python tests/tst_speech.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.speech import SpeechControl, shorten, wants_brief, wants_short, wants_stop, words  # noqa: E402

check = tw.check
T0 = tw.T0

LONG = ("The mitochondria is the powerhouse of the cell. It turns food into energy the cell can use, "
        "through a process called cellular respiration. That happens in several stages: glycolysis, "
        "the Krebs cycle, and the electron transport chain, each in a different part of the cell.")


def p1_stop():
    print("\n=== P1. Hearing 'stop' ===")
    for t in ("stop", "Stop.", "stop!", "shh", "shhhh", "hush", "quiet", "be quiet", "enough", "that's enough",
              "stop talking", "ok stop", "okay, stop please", "ash, stop", "hey ash stop", "shut up", "ok ok",
              "okay okay okay", "alright alright", "got it", "I get it", "stop it", "pause", "stop there ash"):
        check(f"while talking: {t!r}", wants_stop(t, speaking=True))
    for t in ("I'll stop asking about that", "stop the music", "stop the timer", "don't stop", "quiet hours",
              "is it quiet", "enough about that, tell me about Paris", "stop watching the download",
              "how do I stop a process", "", None, "stop " * 30):
        check(f"not a stop: {str(t)[:30]!r}", not wants_stop(t, speaking=True))
    check("a long run of 'please' is not a quick stop", not wants_stop("stop" + ", please" * 10, speaking=True))
    for t in ("stop", "enough", "pause", "got it", "quiet", "ok ok"):
        check(f"while quiet, {t!r} is left alone", not wants_stop(t, speaking=False))
    for t in ("shh", "hush", "stop talking", "shut up", "ash, shush"):
        check(f"while quiet, {t!r} still counts", wants_stop(t, speaking=False))


def p2_phrases():
    print("\n=== P2. Short version, keep it short ===")
    for t in ("the short version", "short version please", "give me the short version", "what's the short version",
              "tl;dr", "TLDR", "tl dr", "the gist", "just the gist", "too long", "in short", "summarize that",
              "can you summarise it", "shorter", "briefly", "say that shorter", "ash, the quick version"):
        check(f"short: {t!r}", wants_short(t))
    for t in ("tell me the short story of Rome", "a short version of the song", "that's too long a walk",
              "summarize the meeting notes in my documents", "", None):
        check(f"not short: {str(t)[:30]!r}", not wants_short(t))
    for t, want in (("keep it short", True), ("keep your answers short from now on", True), ("be brief", True),
                    ("shorter answers please", True), ("talk less", True), ("please keep it brief", True),
                    ("you can talk more", False), ("full answers", False), ("back to normal answers", False),
                    ("no need to keep it short", False), ("talk more", False), ("give me longer answers", False)):
        check(f"brief {t!r} -> {want}", wants_brief(t) is want)
    for t in ("keep it short and simple, what is DNS", "be brief in the email to Sara", "the short version", "",
              None):
        check(f"not a preference: {str(t)[:30]!r}", wants_brief(t) is None)
    s = shorten(LONG)
    check("the first sentence", s == "The mitochondria is the powerhouse of the cell.", s)
    s = shorten("Yes. It is open until nine tonight, and on weekends until eleven, and closed on holidays.")
    check("a very short first sentence takes the next too", s.startswith("Yes. It is open until nine"), s)
    s = shorten("word " * 60 + ".")
    check("cut at 25 words, with an ellipsis", words(s) == 25 and s.endswith("…"), s)
    s = shorten("word " * 40 + ".")
    check("forty words are cut too", words(s) == 25, s)
    s = shorten("Use ```ls -la``` to list every file in this folder, hidden ones included.")
    check("inline code dropped", "ls -la" not in s and s.startswith("Use"), s)
    s = shorten("Here's how to do it, in three steps:\n- open the settings\n- pick Network\n- turn it off")
    check("a list reads as one sentence", "steps: open the settings; pick Network" in s and "\n" not in s, s)
    s = shorten("Run this command to see it in full detail right now:\n\n```\nls -la\n```\n\nThen look at the sizes.")
    check("code blocks dropped", "ls -la" not in s and s.startswith("Run this command"), s)
    check("short text kept as it is", shorten("Sure.") == "Sure.")
    check("nothing in, nothing out", shorten("") == "" and shorten(None) == "")


def p3_control():
    print("\n=== P3. The preference and the last answer ===")
    t = [T0]
    p = os.path.join(tempfile.mkdtemp(), "speech.json")
    c = SpeechControl(path=p, clock=lambda: t[0])
    check("off at first", not c.brief and c.spoken_form(LONG, {"addressed": True}) is None)
    c.set_brief(True)
    check("on: a long answer is spoken short", c.spoken_form(LONG, {"addressed": True})
          == "The mitochondria is the powerhouse of the cell.")
    check("a short answer is not touched", c.spoken_form("It's 5 pm.", {"addressed": True}) is None)
    check("a few short sentences are not cut", c.spoken_form("It opens at nine in the morning today. It closes at "
                                                           "five. Sundays it is shut.", {"addressed": True}) is None)
    check("a remark is not touched", c.spoken_form(LONG, {"addressed": False}) is None)
    check("the short version itself is not cut again", c.spoken_form(LONG, {"addressed": True, "short_version": True})
          is None)
    check("kept across a restart", SpeechControl(path=p).brief is True)
    c.set_brief(False)
    check("and off again", SpeechControl(path=p).brief is False)
    open(p, "w").write("{broken")
    check("a broken file: off, no crash", SpeechControl(path=p).brief is False)
    c = SpeechControl(clock=lambda: t[0])
    check("nothing said yet", c.short_version() == "I haven't said anything long just now.")
    c.said(LONG, {"addressed": True})
    c.said("By the way, it's sunny.", {"addressed": False})
    check("the short version of the last answer, not the remark",
          c.short_version() == "The mitochondria is the powerhouse of the cell.")
    c.said("The mitochondria is the powerhouse of the cell.", {"addressed": True, "short_version": True})
    check("saying the short version does not replace the answer", c.last["text"] == LONG)
    t[0] += 301
    check("only for a while", c.short_version() == "I haven't said anything long just now.")
    c.said("It's 5 pm.", {"addressed": True})
    check("already short: said so", c.short_version() == "That was the short version.")


class Tts:
    engine = True

    def __init__(self):
        self.texts = []

    def stream_audio(self, text):
        self.texts.append(text)
        yield b"RIFF" + text.encode()


class Player:
    """Plays for `secs` unless stopped; records what happened."""

    def __init__(self, secs=0.5):
        self.secs = secs
        self.log = []

    def __call__(self, wav, stop):
        self.log.append(("start", wav[4:].decode()))
        stopped = stop.wait(self.secs)
        self.log.append(("stopped" if stopped else "done", wav[4:].decode()))


def p4_speaker():
    print("\n=== P4. The speaker ===")
    from src.senses.actuators import SpeakActuator
    tts, player = Tts(), Player(0.4)
    sp = SpeakActuator(tts_engine=tts, player=player, dry_run=False)
    sp.enable()
    check("quiet at first", not sp.speaking)
    r = sp.act("hello there")
    time.sleep(0.05)
    check("speaking", r["ok"] and sp.speaking)
    check("stop says it cut something", sp.stop() is True)
    time.sleep(0.05)
    check("cut off mid-sound", ("stopped", "hello there") in player.log and not sp.speaking, str(player.log))
    check("and nothing to stop now", sp.stop() is False)
    check("it knows when it went quiet", 0 < time.time() - sp.ended_at < 1.0)
    player.log.clear()
    sp.act("first line")
    time.sleep(0.05)
    sp.act("second line")
    time.sleep(0.15)
    check("the old line ending does not mark the new one quiet", sp.speaking)
    time.sleep(0.45)
    check("a new line replaces the old one", player.log[:2] == [("start", "first line"), ("stopped", "first line")]
          and ("done", "second line") in player.log, str(player.log))
    check("quiet after", not sp.speaking)


class Ash(tw.Ash):
    reply = "Sure."

    def think(self, query, image=None, context=""):
        self.calls.append(query)
        return tw.Resp(self.reply)


def p4b_windows_player():
    print("\n=== P4b. Windows playback ===")
    import io
    import wave
    from src.senses import actuators as A
    calls = []

    class Fake:
        SND_FILENAME, SND_ASYNC, SND_NODEFAULT = 1, 2, 4

        @staticmethod
        def PlaySound(sound, flags):
            calls.append((sound, flags))

    old = (A._HAS_WINSOUND, getattr(A, "winsound", None))
    A._HAS_WINSOUND, A.winsound = True, Fake
    try:
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000 * 3)
        stop = threading.Event()
        threading.Timer(0.2, stop.set).start()
        t0 = time.time()
        A._play_wav(b.getvalue(), stop)
        check("played as a file, asynchronously", calls[0][1] == 7 and str(calls[0][0]).endswith(".wav"))
        check("stopped early, and the sound cut", time.time() - t0 < 1.0 and calls[-1] == (None, 0), str(calls))
        calls.clear()
        A._play_wav(b.getvalue()[:100], threading.Event())
        check("played to the end: nothing cut", calls and calls[-1][0] is not None, str(calls))
    finally:
        A._HAS_WINSOUND, A.winsound = old


def rig(brief=False):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.actuators import SpeakActuator
    from src.senses.base import PeripheryRegistry
    clock = [T0]
    out, metas = [], []
    per = PeripheryRegistry()
    tts, player = Tts(), Player(5.0)
    sp = SpeakActuator(tts_engine=tts, player=player, dry_run=False)
    sp.enable()
    per.actuators["speak"] = sp
    d = tempfile.mkdtemp()
    ash = Ash()
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(),
                        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9,
                                             speech_path=os.path.join(d, "speech.json")),
                        response_sink=lambda t, m: (out.append(t), metas.append(dict(m))))
    rt.clock = lambda: clock[0]
    rt.speech.clock = lambda: clock[0]
    if brief:
        rt.speech.set_brief(True)
    rt.t, rt.out, rt.metas, rt.sp, rt.player, rt.tts_ = clock, out, metas, sp, player, tts

    def say(text, source="microphone"):
        from src.senses.base import Modality, SensorEvent
        clock[0] += 3
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()
    rt.say = say
    return rt


def p5_runtime():
    print("\n=== P5. The runtime ===")
    rt = rig()
    rt.sp.act(LONG)
    time.sleep(0.05)
    n = len(rt.out)
    rt.say("stop")
    time.sleep(0.05)
    check("'stop' while speaking cuts it off", not rt.sp.speaking and any(x[0] == "stopped" for x in rt.player.log))
    check("and says nothing back, asks no model", len(rt.out) == n and rt.ash.calls == [], str(rt.out[n:]))
    check("counted", rt.speech.stats["stopped"] == 1)
    rt.sp.ended_at = time.time() - 30                # quiet for a while now
    rt.say("stop")
    check("'stop' while quiet goes on to the rest", rt.ash.calls == ["stop"], str(rt.ash.calls))
    rt.ash.calls.clear()
    rt.sp.act("just now")
    time.sleep(0.05)
    rt.sp.stop()
    rt.sp.ended_at = time.time()                     # it went quiet a moment ago; the "shh" was late
    rt.say("enough")
    check("just after speaking (transcription lags): still a stop", rt.ash.calls == [], str(rt.ash.calls))
    rt.sp.ended_at = time.time() - 30
    rt.say("shh")
    check("'shh' when long quiet: nothing to stop, nothing said", rt.ash.calls == [] and rt.out[-1:] != ["stop"])
    rt.say("shh", source="console")
    check("typed in the console it works too", rt.ash.calls == [])

    rt = rig()
    rt.ash.reply = LONG
    rt.say("what do mitochondria do")
    check("(a long answer)", rt.out[-1] == LONG, str(rt.out[-1:]))
    rt.say("the short version")
    check("the short version", rt.out[-1] == "The mitochondria is the powerhouse of the cell.", str(rt.out[-1:]))
    check("not from the model", rt.ash.calls == ["what do mitochondria do"], str(rt.ash.calls))
    check("marked, so it is not cut or kept again", rt.metas[-1].get("short_version") is True)
    rt.say("keep it short from now on")
    check("brief mode on", rt.speech.brief and "short" in rt.out[-1].lower(), str(rt.out[-1:]))
    rt.say("what do mitochondria do")
    check("the console still gets all of it", rt.out[-1] == LONG)
    check("but it is spoken short", rt.metas[-1].get("spoken") == "The mitochondria is the powerhouse of the cell.",
          str(rt.metas[-1]))
    rt.say("you can talk more")
    check("brief mode off", not rt.speech.brief)
    check("status carries it", rt.status()["speech"]["brief"] is False)

    from src.ui import UIBus, attach_ui
    rt = rig()
    bus = UIBus()
    attach_ui(rt, bus, opener=None, status_every_s=10 ** 9)
    rt.sp.act(LONG)
    time.sleep(0.05)
    check("(the console shows it speaking)", bus.state.get("speaking"))
    rt.say("stop")
    check("and not speaking once stopped", not bus.state.get("speaking"), str(bus.state))


if __name__ == "__main__":
    for fn in (p1_stop, p2_phrases, p3_control, p4_speaker, p4b_windows_player, p5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
