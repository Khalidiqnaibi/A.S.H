"""
tst_bmo.py -- caring nudges, a story that continues, voices, face sounds.

  B1  Caring nudges: time at the machine adds up and a ten-minute break
      resets it; a stretch at 90 minutes, water at 2 hours, a break at 3;
      each once a stretch; only when the mood is fine; "no more nudges"
  B2  The story: kept where it got to; offered as "the next part"; the brain
      asked to continue from there; fresh after a week, or when asked;
      "what happens next?"; across a restart
  B3  Voices: a game line carries the game voice; the speaker uses it, and
      falls back to the usual voice if it fails
  B4  Face sounds: off by default; only for lines ASH volunteers, not in
      quiet hours, not more than one every 20 seconds; the tone is a real WAV
  B5  The runtime: a nudge at a free moment; a story told, then continued
      the next evening; the setting for face sounds

Run:  PYTHONIOENCODING=utf-8 python tests/tst_bmo.py
"""

import io
import os
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.care import Care  # noqa: E402
from src.daemon.storybook import NEW_PROMPT, StoryBook  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402

check = tw.check
T0 = tw.T0
EVE = time.mktime((2026, 3, 10, 20, 30, 0, 0, 0, -1))
DAY = 86400.0
FREE = dict(idle_s=60.0, free_moment=True)


def b1_care():
    print("\n=== B1. Caring nudges ===")
    c = Care(clock=lambda: T0)
    fine = Environment(mood=0.4, **FREE)
    c.tick(T0, 0.0)
    check("just started: nothing", c.propose(fine, T0) is None)
    for m in range(1, 71):
        c.tick(T0 + m * 60, 5.0)
    check("70 minutes: not yet", c.propose(fine, T0 + 70 * 60) is None)
    for m in range(71, 95):
        c.tick(T0 + m * 60, 5.0)
    p = c.propose(fine, T0 + 94 * 60)
    check("a stretch at 90 minutes", p is not None and "stretch" in p.text and "an hour and a half" in p.text,
          str(p and p.text))
    c.spoken(T0 + 94 * 60)
    check("said once a stretch", c.propose(fine, T0 + 95 * 60) is None)
    check("not when the mood is unknown", c.propose(Environment(**FREE), T0 + 2.1 * 3600) is None)
    check("not when low", c.propose(Environment(mood=-0.2, **FREE), T0 + 2.1 * 3600) is None)
    p = c.propose(fine, T0 + 2.1 * 3600)
    check("water at two hours", p is not None and "water" in p.text)
    c.spoken(T0 + 2.1 * 3600)
    p = c.propose(fine, T0 + 3.1 * 3600)
    check("a break at three", p is not None and "3 hours straight" in p.text, str(p and p.text))
    c.spoken(T0 + 3.1 * 3600)
    check("and that once too", c.propose(fine, T0 + 3.15 * 3600) is None)
    c.tick(T0 + 3.2 * 3600, 30.0)
    check("a short pause is not a break", c.at_it(T0 + 3.2 * 3600) > 3 * 3600)
    c.tick(T0 + 3.4 * 3600, 700.0)
    check("ten minutes away is", c.since is None and c.said == set() and c.stats["breaks"] == 1)
    check("no idle reading: nothing changes", (c.tick(T0, None), c.since)[1] is None)
    check("'no more nudges'", c.hear("no more nudges", T0) == "Okay, no more nudges." and c.off)
    c.since = T0
    check("off: nothing", c.propose(fine, T0 + 5 * 3600) is None)
    check("other words are not ours", Care().hear("stop", T0) is None)


def b2_story():
    print("\n=== B2. The story ===")
    p = os.path.join(tempfile.mkdtemp(), "story.json")
    t = [EVE]
    b = StoryBook(path=p, clock=lambda: t[0])
    check("no story: a new one is offered", b.offer_text(t[0]) == "Want a short story before you wrap up for the night?")
    check("and asked for fresh", b.prompt() == NEW_PROMPT and b.waiting)
    b.told("Once upon a time, a fox found a lantern. " * 30)
    check("kept where it got to", b.story["part"] == 1 and len(b.story["tail"]) <= 700 and not b.waiting)
    b.told("something else said later")
    check("only the answer right after the prompt", b.story["tail"].endswith("lantern."))
    t[0] += DAY
    check("next evening: the next part", b.offer_text(t[0]) == "Want the next part of last night's story?")
    q = b.prompt()
    check("the brain is asked to carry on from there", q.startswith("Continue the story") and "part 2" in q
          and "found a lantern" in q, q[:120])
    b.told("The fox followed the light into the woods.")
    check("part two", b.story["part"] == 2)
    check("kept across a restart", StoryBook(path=p, clock=lambda: t[0]).story["part"] == 2)
    t[0] += 3 * DAY
    check("a few days on: 'our story'", b.offer_text(t[0]) == "Want the next part of our story?")
    t[0] += 5 * DAY
    check("after a week: fresh", not b.ongoing() and b.prompt() == NEW_PROMPT)
    b = StoryBook(clock=lambda: t[0])
    b.prompt()
    b.told("A tale.")
    check("'what happens next?'", (b.hear("what happens next?") or "").startswith("Continue the story"))
    check("'tell me a new story'", b.hear("tell me a new story") == NEW_PROMPT and b.story is None)
    check("no story: 'what happens next' is not ours", StoryBook().hear("what happens next") is None)
    for text in ("tell me a story about dragons", "continue", "", None):
        check(f"not ours: {str(text)[:30]!r}", b.hear(text) is None)
    open(p, "w").write("{bad")
    check("a broken file: no story, no crash", StoryBook(path=p).story is None)


class Tts:
    engine = True

    def __init__(self, bad_voice=False):
        self.calls = []
        self.bad = bad_voice

    def stream_audio(self, text, voice=None):
        self.calls.append((text, voice))
        if voice and self.bad:
            raise ValueError("no such voice")
        yield b"RIFF"


def b3_voices():
    print("\n=== B3. Voices ===")
    from src.senses.actuators import SpeakActuator
    tts = Tts()
    sp = SpeakActuator(tts_engine=tts, player=lambda w, s: None, dry_run=False)
    sp.enable()
    sp.act({"text": "Higher.", "voice": "bm_george"})
    time.sleep(0.1)
    check("spoken in the voice asked for", tts.calls == [("Higher.", "bm_george")], str(tts.calls))
    sp.act("Plain.")
    time.sleep(0.1)
    check("a plain line in the usual voice", tts.calls[-1] == ("Plain.", None))
    tts = Tts(bad_voice=True)
    sp = SpeakActuator(tts_engine=tts, player=lambda w, s: None, dry_run=False)
    sp.enable()
    sp.act({"text": "Lower.", "voice": "nope"})
    time.sleep(0.1)
    check("a voice that fails: the usual one", tts.calls == [("Lower.", "nope"), ("Lower.", None)], str(tts.calls))


def b4_sounds():
    print("\n=== B4. Face sounds ===")
    from src.face import sounds as S
    played = []
    t = [T0]
    fs = S.FaceSounds(play=lambda p: (played.append(p), True)[1], clock=lambda: t[0], folder=tempfile.mkdtemp())
    check("off by default", not fs.enabled and not fs.maybe("happy", True, False))
    fs.enabled = True
    check("an answer you asked for: no sound", not fs.maybe("happy", False, False))
    check("in quiet hours: no sound", not fs.maybe("happy", True, True))
    check("an expression without a sound: none", not fs.maybe("skeptical", True, False))
    check("a volunteered line: the sound", fs.maybe("curious", True, False) and played[-1].endswith("curious.wav"))
    t[0] += 5
    check("not again within 20 seconds", not fs.maybe("alert", True, False))
    t[0] += 20
    check("after that, yes", fs.maybe("alert", True, False))
    with wave.open(io.BytesIO(S.make_wav(S.TONES["very_happy"]))) as w:
        n, rate = w.getnframes(), w.getframerate()
    check("a real, short WAV", rate == S.RATE and 0.2 < n / rate < 0.4, str(n / rate))
    fs2 = S.FaceSounds(enabled=True, play=lambda p: False, clock=lambda: t[0], folder=tempfile.mkdtemp())
    check("no sound device: quietly nothing", not fs2.maybe("happy", True, False) and fs2.stats["played"] == 0)


def b5_runtime():
    print("\n=== B5. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [EVE]
    out = []
    d = tempfile.mkdtemp()

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            self.calls.append(query)
            return tw.Resp("The lantern glowed brighter as the fox stepped into the dark forest, and then...")
    ash = Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=99),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, play=True, care=True,
                             story_path=os.path.join(d, "story.json"), game_voice="bm_george", moments=True,
                             moments_cfg={"min_gap_s": 0, "max_per_hour": 99}),
        response_sink=lambda t, m: out.append((t, dict(m))))
    rt.clock = lambda: clock[0]

    def say(text, source="console", addressed=True):
        clock[0] += 3
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    rt.play.offer("story", clock[0])
    say("sure")
    check("a story told by the brain, asked for fresh", ash.calls and ash.calls[-1] == NEW_PROMPT, str(ash.calls))
    check("and kept", rt.storybook.story and "lantern glowed" in rt.storybook.story["tail"])
    clock[0] += DAY
    say("what happens next?")
    check("the next evening, continued", ash.calls[-1].startswith("Continue the story") and "part 2" in ash.calls[-1])
    check("part two kept", rt.storybook.story["part"] == 2)
    check("the story offer knows", rt.moments.activities["story"].propose(Environment(**FREE), clock[0] + 600)
          .text.startswith("Want the next part"))
    say("let's play the number game")
    check("a game line in the game voice", out[-1][1].get("voice") == "bm_george", str(out[-1][1]))
    say("stop")
    say("what time is it")
    check("not other lines", "voice" not in out[-1][1])
    check("care is a free-moment activity", "care" in rt.moments.activities)
    idle.value = 0.0
    rt._tick()
    check("and is ticked with the idle time", rt.care.since is not None)
    rt.care.since = clock[0] - 2.5 * 3600
    say("no more nudges")
    check("'no more nudges' heard any time", out[-1][0] == "Okay, no more nudges." and rt.care.off)
    check("status", rt.status()["story"]["part"] == 2 and rt.status()["care"]["off"] is True)

    from src.face.bridge import FaceBridge
    from src.ui import settings as st

    class Player:
        def play(self, name, force=False):
            return True
    rt.face = FaceBridge(Player())
    check("face sounds off by default", st.current(rt, "face.sounds") is False)
    r = st.change(rt, os.path.join(d, "daemon.json"), "face.sounds", "on")
    check("turned on from the console, live", r["ok"] and r["live"] and rt.face.sounds.enabled)


if __name__ == "__main__":
    for fn in (b1_care, b2_story, b3_voices, b4_sounds, b5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
