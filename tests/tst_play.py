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
  P7  A late yes to an offer, and asking for today's trivia in the words people use
  P8  A request to play worded unlike the patterns is understood by meaning (the chooser); what the patterns catch teaches it

Run:  PYTHONIOENCODING=utf-8 python tests/tst_play.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def p7_late_yes_and_trivia_words():
    print("\n=== P7. A late yes; today's trivia in plain words ===")
    from src.daemon.games import DailyTrivia
    spoke = [0.0]
    p = Play(seed=1)
    p.ash_last = lambda: spoke[0]
    DailyTrivia(p).spoken(T0)
    spoke[0] = T0 + 1                                        # ASH said the offer itself
    got = p.hear("yes", T0 + 23 * 60)
    check("a plain yes 23 minutes after the offer still starts today's trivia, since nothing was said in between",
          got is not None and p.playing and "?" in got["say"], str(got))
    p = Play(seed=1)
    p.ash_last = lambda: spoke[0]
    DailyTrivia(p).spoken(T0)
    spoke[0] = T0 + 400                                      # ASH said something else meanwhile
    check("but not when ASH said something else since", p.hear("yes", T0 + 23 * 60) is None and not p.playing)
    p = Play(seed=1)
    p.ash_last = lambda: 0.0
    DailyTrivia(p).spoken(T0)
    check("and not after more than half an hour", p.hear("yes", T0 + 35 * 60) is None)
    p = Play(seed=1)
    DailyTrivia(p).spoken(T0)
    check("with no way to know, it is the old two minutes", p.hear("yes", T0 + 600) is None)
    for q in ("whats todays trivia", "what's today's trivia", "no i was saying yes about that i want the todays trivia", "give me a riddle",
              "can i have today's trivia", "i want a riddle", "show me the trivia"):
        p = Play(seed=1)
        got = p.hear(q, T0)
        check(f"{q!r} starts it", got is not None and p.playing, str(got))
    a, b = Play(seed=1), Play(seed=2)
    ga, gb = a.hear("whats todays trivia", T0), b.hear("whats todays trivia", T0)
    check("today's trivia is the same question whoever asks", ga["say"] == gb["say"], f"{ga} {gb}")
    for q in ("what is the capital of France", "what's the trivia about", "i want a sandwich", "what is a riddle"):
        p = Play(seed=1)
        check(f"{q!r} is left alone", p.hear(q, T0) is None and not p.playing)


def p8_by_meaning():
    print("\n=== P8. By meaning ===")
    import semantic_fake as sf
    from src.daemon.chooser import Chooser
    from src.daemon.play import SEEDS
    ch = Chooser("play", sf.embed, SEEDS, accept=0.62, margin=0.1)
    p = Play(seed=1)
    check("without a chooser 'trivia question todays' is not a request", p.hear("trivia question todays", T0) is None and not p.playing)
    p = Play(seed=1)
    p.chooser = ch
    got = p.hear("trivia question todays", T0)
    check("with one it is, and it is today's trivia", got is not None and p.playing and "?" in got["say"], f"{got} {ch.classify('trivia question todays').scores}")
    p2 = Play(seed=2)
    p2.chooser = Chooser("play", sf.embed, SEEDS, accept=0.62, margin=0.1)
    check("the same question whoever asks", p2.hear("trivia question todays", T0)["say"] == got["say"])
    p3 = Play(seed=1)
    p3.chooser = ch
    check("a riddle asked for in other words", p3.hear("riddle puzzle for me", T0) is not None and p3.playing)
    for q in ("play some music", "what is a riddle", "play the next song", "thanks", "tell me a joke", "who won the game last night"):
        p4 = Play(seed=1)
        p4.chooser = ch
        check(f"{q!r} is not a request to play", p4.hear(q, T0) is None and not p4.playing, str(p4.hear(q, T0)))
    n = len(ch.learned.get("trivia", []))
    p5 = Play(seed=1)
    p5.chooser = ch
    p5.hear("whats todays trivia", T0)
    check("what a pattern caught teaches the chooser", len(ch.learned.get("trivia", [])) >= n, str(ch.status()))
    import tst_effects as te
    rt = te.rig(play=True, chooser_embed=sf.embed, chooser_dir=tempfile.mkdtemp())
    n0 = len(rt.ash.calls)
    rt.say("trivia question todays")
    check("in the runtime it starts a game with no model call", rt.play.playing and len(rt.ash.calls) == n0, f"{rt.out[-1:]}")


if __name__ == "__main__":
    for fn in (p1_start, p2_number, p3_quiz, p4_stopping, p5_offers, p6_runtime, p7_late_yes_and_trivia_words, p8_by_meaning):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
