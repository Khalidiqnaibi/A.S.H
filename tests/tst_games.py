"""
tst_games.py -- more games: word association, twenty questions, riddles, trivia.

  G1  Word association: ASH answers each word with one it brings to mind,
      never repeating; an unknown word still gets an answer; the chain at the
      end
  G2  Twenty questions: yes/no from what the thing is; a right guess wins, a
      wrong one counts; unclear questions don't count; twenty at most;
      giving up says what it was
  G3  Riddles: the answer (plural too), a hint, giving up, three tries
  G4  Trivia: the answer in words or digits, a second try, the same daily
      question all day
  G5  Starting them: by name, "tell me a riddle", any game; the daily trivia
      and the idle riddle as free-moment offers, and a yes starts that game
  G6  The runtime: a riddle start to finish without the model

Run:  PYTHONIOENCODING=utf-8 python tests/tst_games.py
"""

import os
import random
import sys
import tempfile
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import games as G  # noqa: E402
from src.daemon.play import NumberGame, Play  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402

check = tw.check
T0 = tw.T0
FREE = dict(idle_s=60.0, free_moment=True)


def g1_words():
    print("\n=== G1. Word association ===")
    g = G.WordAssociation(random.Random(1), rounds=3)
    first = g.word
    check("opens with a word", first in G.SEEDS and f"“{first}”" in g.start())
    say, over = g.hear("beach")
    mine = say.rstrip(".").lower()
    check("answers with one it brings to mind", mine in G.ASSOC["beach"] and not over, say)
    say2, _ = g.hear("um, the zorblax")
    check("an unknown word still gets an answer", say2.rstrip(".").lower() in G.ASSOC, say2)
    check("never repeats a word", len(g.chain) == len(set(g.chain)), str(g.chain))
    g.used = set(g.chain) - {"sun"} | {"sand", "waves"}   # two of "beach"'s three are taken
    check("an answer not used yet", g._answer("beach") == "sun")
    say3, over = g.hear("sand")
    check("the chain at the end", over and "Great chain:" in say3 and " → " in say3, say3)
    g = G.WordAssociation(random.Random(1))
    check("no word: asks for one", g.hear("...") == ("Just one word -- whatever pops up.", False))
    check("the table points at itself where it can", sum(1 for v in G.ASSOC.values() for w in v if w in G.ASSOC)
          > len(G.ASSOC))


def g2_twenty():
    print("\n=== G2. Twenty questions ===")
    g = G.TwentyQuestions(random.Random(0))
    g.names, g.feats = ("cat", "kitten"), {"alive", "animal", "legs", "fur", "pet", "house", "small"}
    check("opens", "Twenty questions" in g.start())
    check("alive: yes", g.hear("is it alive?") == ("Yes.", False))
    check("bigger than a breadbox: no", g.hear("is it bigger than a breadbox") == ("No.", False))
    check("can you eat it: no", g.hear("can you eat it?") == ("No.", False))
    check("does it have fur: yes", g.hear("does it have fur") == ("Yes.", False))
    say, over = g.hear("what colour is your hat")
    check("unclear: not counted", not over and "can't tell" in say and g.asked == 4)
    check("a wrong guess counts", g.hear("is it a dog?") == ("No, it's not a dog.", False) and g.asked == 5)
    say, over = g.hear("is it a cat?")
    check("a right guess wins", over and say == "Yes! It's a cat -- you got it in 6.", say)
    g = G.TwentyQuestions(random.Random(0), limit=3)
    g.names, g.feats = ("moon",), {"big", "sky"}
    g.hear("is it big")
    g.hear("is it alive")
    say, _ = g.hear("is it in the sky")
    check("twenty at most (here three)", say == "Yes. That's twenty -- it was a moon!", say)
    g = G.TwentyQuestions(random.Random(0))
    g.names = ("key",)
    check("giving up says what it was", g.hear("I give up") == ("It was a key! Another time.", True))
    check("'is it electric' is a question, not a guess at 'electric'",
          G.TwentyQuestions._feature("is it electric?") == "electric")
    check("every thing has a feature the questions can find",
          all(any(f in feats for _, f in G.FEATURES) for _, feats in G.THINGS))


def g3_riddles():
    print("\n=== G3. Riddles ===")
    g = G.Riddle(random.Random(0), which=0)
    check("asks it", g.start() == "Here's a riddle: What has keys but can't open locks?")
    check("a wrong answer: offers a hint", g.hear("a door") == ("Not quite. Want a hint?", False))
    check("the hint", g.hear("hint please") == ("Hint: You play it.", False))
    check("then: try again", g.hear("a lock") == ("Not quite -- try again.", False))
    check("right: done", g.hear("a piano!") == ("Got it -- a piano!", True))
    g = G.Riddle(random.Random(0), which=9)
    check("first time, said with no article", g.hear("footsteps") == ("Yes! It's footsteps.", True))
    g = G.Riddle(random.Random(0), which=5)
    check("plural counts too", g.hear("combs") == ("Yes! It's a comb.", True))
    g = G.Riddle(random.Random(0), which=1)
    check("giving up", g.hear("I don't know") == ("It's a towel.", True))
    g = G.Riddle(random.Random(0), which=1)
    g.hear("a")
    g.hear("b")
    check("three tries", g.hear("c") == ("Not quite -- it's a towel.", True))
    check("every riddle has a hint and an answer", all(q and a and h for q, a, h in G.RIDDLES))


def g4_trivia():
    print("\n=== G4. Trivia ===")
    g = G.Trivia(random.Random(0), which=1)                  # spider legs
    check("asks", g.start() == "Trivia: How many legs does a spider have?")
    check("in digits", g.hear("8") == ("Correct! 8.", True))
    g = G.Trivia(random.Random(0), which=1)
    check("in words", g.hear("I think eight")[1])
    g = G.Trivia(random.Random(0), which=0)
    check("a second try", g.hear("saturn") == ("Not quite. One more try?", False))
    check("then the answer", g.hear("neptune") == ("Not quite -- it's jupiter.", True))
    g = G.Trivia(random.Random(0), which=0)
    check("giving up", g.hear("no idea") == ("It's jupiter.", True))
    g = G.Trivia(random.Random(0), which=4)                   # pacific
    check("not a part of a longer word", g.hear("pacificism")[0].startswith("Not quite"))
    d = date(2026, 3, 10)
    check("the same question all day, another tomorrow",
          G.daily_index(d) == G.daily_index(d) and G.daily_index(d) != G.daily_index(date(2026, 3, 11)))


def g5_starting():
    print("\n=== G5. Starting them ===")
    p = Play(seed=1)
    for text, kind in (("let's play word association", G.WordAssociation), ("let's play twenty questions", G.TwentyQuestions),
                       ("play 20 questions", G.TwentyQuestions), ("tell me a riddle", G.Riddle),
                       ("give me a riddle", G.Riddle), ("let's play a riddle", G.Riddle),
                       ("ask me a trivia question", G.Trivia), ("let's play trivia", G.Trivia),
                       ("play the number game", NumberGame)):
        p.game = None
        r = p.hear(text, T0)
        check(f"{text!r} -> {kind.__name__}", isinstance(p.game, kind), str(r))
    kinds = set()
    for s in range(30):
        q = Play(seed=s)
        q.hear("let's play a game", T0)
        kinds.add(type(q.game).__name__)
    check("any game: a few different ones", {"NumberGame", "WordAssociation", "TwentyQuestions"} <= kinds, str(kinds))
    p = Play(seed=1)
    dt = G.DailyTrivia(p)
    c = dt.propose(Environment(**FREE), T0)
    check("the daily trivia is offered", c is not None and c.text == "Want today's trivia question?")
    dt.spoken(T0)
    r = p.hear("sure", T0 + 5)
    want = G.TRIVIA[G.daily_index(date.fromtimestamp(T0))][0]
    check("a yes asks today's question", isinstance(p.game, G.Trivia) and want in r["say"], str(r))
    p.game = None
    check("once a day", dt.propose(Environment(**FREE), T0 + 3600) is None)
    p.game = None
    check("again tomorrow", dt.propose(Environment(**FREE), T0 + 86400) is not None)
    check("not to someone who is low", G.DailyTrivia(p).propose(Environment(mood=-0.4, **FREE), T0) is None)
    p.game = None
    ir = G.IdleRiddle(p)
    check("a riddle offered when things go well", ir.propose(Environment(mood=0.5, **FREE), T0) is not None)
    check("not when the mood is unknown", ir.propose(Environment(**FREE), T0) is None)
    ir.spoken(T0)
    r = p.hear("go on", T0 + 5)
    check("a yes asks a riddle", isinstance(p.game, G.Riddle) and r["say"].startswith("Here's a riddle"), str(r))
    check("not while playing", ir.propose(Environment(mood=0.5, **FREE), T0) is None)


def g6_runtime():
    print("\n=== G6. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, play=True,
                                                      moments=True),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]

    def say(text, source="console", addressed=True):
        clock[0] += 2
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("tell me a riddle")
    check("a riddle, no model", out[-1].startswith("Here's a riddle") and ash.calls == [], str(out[-1:]))
    rt.play.game.answers = ("a towel", "towel")
    say("hint", source="microphone", addressed=False)
    check("a hint heard without the name", out[-1].startswith("Hint:"))
    say("a towel", source="microphone", addressed=False)
    check("solved (a hint is not a try), and it offers another", out[-1] == "Yes! It's a towel. Another one?", str(out[-1:]))
    check("the free-moment offers are there", {"daily_trivia", "riddle"} <= set(rt.moments.activities))
    check("no model asked at all", ash.calls == [])


if __name__ == "__main__":
    for fn in (g1_words, g2_twenty, g3_riddles, g4_trivia, g5_starting, g6_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
