"""
tst_play.py -- quick games, ASH wanting to play, and stories.

  P1  Starting a game, and what is not a request
  P2  The number game: higher, lower, words for numbers, out of range,
      winning, another round
  P3  The people quiz: from what ASH was told; agreeing and disagreeing;
      not offered with too few people
  P4  Stopping, and nothing else is read while playing
  P5  Wanting to play (only when things go well) and the evening story offer;
      yes / no to each
  P6  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tst_play.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.play import Bored, NumberGame, PeopleQuiz, Play, Story  # noqa: E402
from src.daemon.wondering import Environment, Knowledge  # noqa: E402

check = tw.check
T0 = tw.T0
FREE = dict(idle_s=60.0, free_moment=True)


def kb_with(*people):
    k = Knowledge()
    for name, rel in people:
        k.store_relation(name, role=rel)
    return k


def p1_start():
    print("\n=== P1. Starting ===")
    p = Play(seed=1)
    for text in ("let's play a game", "can we play something?", "play the number game", "let's play guess the number",
                 "hey ash, let's play"):
        p.game = None
        r = p.hear(text, T0)
        check(f"starts: {text!r}", r is not None and p.playing, str(r))
    p.game = None
    for text in ("I played a game yesterday", "what game is on tonight", "play some music", "", None):
        check(f"not a request: {str(text)[:30]!r}", p.hear(text, T0) is None and not p.playing)
    check("off: nothing", Play(enabled=False).hear("let's play a game", T0) is None)


def p2_number():
    print("\n=== P2. The number game ===")
    p = Play(seed=3)
    r = p.hear("let's play the number game", T0)
    g = p.game
    check("it starts", isinstance(g, NumberGame) and "1 to 50" in r["say"])
    g.secret = s = 20
    check("lower", p.hear("30", T0)["say"] == "Lower.")
    check("higher", p.hear("is it 19?", T0)["say"] == "Higher.")
    check("out of range", p.hear("is it 99", T0)["say"] == "It's between 1 and 50.")
    check("not a number", p.hear("hmm let me think", T0)["say"] == "Give me a number from 1 to 50.")
    r = p.hear(f"{s}", T0)
    check("winning ends it, and offers another", r["say"].startswith(f"Yes! {s}") and r["say"].endswith("Another one?")
          and not p.playing, r["say"])
    r = p.hear("sure", T0 + 5)
    check("and yes starts another", p.playing and r is not None)
    g = NumberGame(__import__("random").Random(0))
    g.secret = 7
    check("number words count", g.hear("seven")[0].startswith("Yes! 7"))
    g = NumberGame(__import__("random").Random(0))
    g.secret = 25
    check("'twenty five' is 25, not 20", g.hear("twenty five")[0].startswith("Yes! 25"))


def p3_quiz():
    print("\n=== P3. The people quiz ===")
    kb = kb_with(("Omar", "cousin"), ("Sara", "colleague"), ("Lina", "sister"))
    p = Play(knowledge=kb, seed=2)
    r = p.hear("let's play the people quiz", T0)
    check("a quiz from what ASH was told", isinstance(p.game, PeopleQuiz) and "Who is" in r["say"], r["say"])
    answers = {"Omar": "my cousin", "Sara": "a coworker", "Lina": "she's my brother"}
    said = []
    while p.playing:
        name = p.game.qs[p.game.i][0]
        said.append(p.hear(answers[name], T0)["say"])
    check("agreeing is said", any("That's what I have too" in s for s in said), str(said))
    check("'coworker' counts for colleague", sum("That's what I have too" in s for s in said) == 2, str(said))
    check("a mismatch is said, not argued", any("I have Lina down as your sister" in s for s in said))
    check("and the score at the end", said[-1].endswith("we agreed on 2 of 3."), said[-1])
    p = Play(knowledge=kb_with(("Omar", "cousin")), seed=2)
    p.hear("let's play the people quiz", T0)
    check("with fewer than two people, the number game instead", isinstance(p.game, NumberGame))


def p4_stopping():
    print("\n=== P4. Stopping ===")
    p = Play(seed=1)
    p.hear("let's play a game", T0)
    check("stop ends it", p.hear("stop", T0)["say"] == "Okay, game over. That was fun." and not p.playing)
    p.hear("let's play a game", T0)
    check("while playing, everything goes to the game", p.hear("what's the weather", T0) is not None)
    p.hear("I give up", T0)
    check("'I give up' ends it", not p.playing)


def p5_offers():
    print("\n=== P5. Wanting to play, and stories ===")
    p = Play(seed=1)
    b = Bored(p, curiosity=lambda: 0.8)
    check("bored only when things go well", b.propose(Environment(mood=0.6, **FREE), T0) is not None)
    check("not when the mood is unknown", b.propose(Environment(**FREE), T0) is None)
    check("not when low", b.propose(Environment(mood=-0.4, **FREE), T0) is None)
    check("not when ASH is not curious", Bored(p, curiosity=lambda: 0.3).propose(Environment(mood=0.6, **FREE), T0) is None)
    c = b.propose(Environment(mood=0.6, **FREE), T0)
    b.spoken(T0)
    check("said, then a yes starts a game", c.text == "I'm a bit bored. Want a quick game?" and
          p.hear("yeah ok", T0 + 10) is not None and p.playing)
    p = Play(seed=1)
    Bored(p).spoken(T0)
    check("a no is taken kindly", p.hear("not now", T0 + 5)["say"] == "Okay, another time." and not p.playing)
    p = Play(seed=1)
    Bored(p).spoken(T0)
    check("a yes long after is not taken", p.hear("yes", T0 + 600) is None)

    eve = time.mktime((2026, 3, 10, 20, 30, 0, 0, 0, -1))
    p = Play(seed=1)
    s = Story(p)
    check("a story in the evening", s.propose(Environment(**FREE), eve) is not None)
    check("not in the afternoon", s.propose(Environment(**FREE), T0) is None)
    check("not to someone who is low", s.propose(Environment(mood=-0.5, **FREE), eve) is None)
    s.spoken(eve)
    check("a yes is handed to the brain to tell", p.hear("sure", eve + 5) == {"story": True})


def p6_runtime():
    print("\n=== P6. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, play=True, wonder=True,
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    rt.play.rng.seed(5)

    def say(text, source="console", addressed=True):
        clock[0] += 2
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("let's play the number game")
    check("a game starts without the model", out[-1][0].startswith("Okay -- I'm thinking") and ash.calls == [])
    s = rt.play.game.secret
    say(str(s), source="microphone", addressed=False)
    check("a guess heard without the name is played", out[-1][0].startswith(f"Yes! {s}"), str(out[-1]))
    rt.play.offer("story", clock[0])
    say("sure, go on", source="microphone", addressed=False)
    check("an accepted story is told by the brain", ash.calls and "story" in ash.calls[-1].lower(), str(ash.calls))
    check("status carries play", rt.status()["play"]["games"] == 1)


if __name__ == "__main__":
    for fn in (p1_start, p2_number, p3_quiz, p4_stopping, p5_offers, p6_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
