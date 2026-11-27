"""tst_pocket.py -- BMO's side by rule: coin, dice, a number, rock-paper-scissors, the 8 ball; "what should I play?"; "what did you do while I was
away?" from the real counters; "hype me up" with facts only; "quiz me on NoSQL" from the person's own notes; "play my coding music" from what they
listen to while coding. Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_pocket.py"""
import os
import random
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon import pocket  # noqa: E402

check = tw.check


def main():
    print("\n=== B1. Pocket games ===")
    p = pocket.Pocket(rng=random.Random(4))
    check("a coin", p.hear("flip a coin") in ("Heads.", "Tails."))
    r = p.hear("roll a die")
    check("a die", r.rstrip(".").isdigit() and 1 <= int(r.rstrip(".")) <= 6, r)
    r = p.hear("roll 2d6")
    check("2d6: both and the sum", r.endswith("in all.") and len(r.split(",")) == 2, r)
    r = p.hear("roll three dice")
    check("three dice", r.count(",") == 2, r)
    r = p.hear("pick a number between 1 and 10")
    check("a number in range", 1 <= int(r.rstrip(".")) <= 10, r)
    check("rock paper scissors: it waits for yours", p.hear("let's play rock paper scissors") == "Rock, paper, scissors -- go! Say yours.")
    r = p.hear("rock")
    check("...and answers it", r.split(".")[0] in ("Rock", "Paper", "Scissors") and ("win" in r or "draw" in r), r)
    check("a bare 'rock' later is nothing", p.hear("rock") is None)
    check("the 8 ball", p.hear("magic 8 ball, will I pass my exam?") in pocket._EIGHT_SAYS)
    check("...with no question, asks for one", p.hear("magic eight ball").startswith("Ask it something"))
    rt = tm.rig()
    used, said = rt.say("flip a coin")
    check("in the runtime, no model", not used and said and said[-1] in ("Heads.", "Tails."), f"{used} {said}")

    print("\n=== B2. What should I play? ===")
    now = time.time()
    s = pocket.what_to_play(["Elden Ring", "Valorant", "Hades"], {"Elden Ring": now - 15 * 86400, "Valorant": now - 3600}, now, random.Random(1))
    check("the one left longest, with how long", s.startswith("You haven't touched Elden Ring in 2 weeks.") and "Hades" in s, s)
    check("no games: says where it looks", pocket.what_to_play([], {}, now).startswith("I don't see any games"))

    print("\n=== B3. ASH's own day ===")
    s = pocket.my_day({"wonders_asked": 1, "observed": 303}, 2, {"trained": 276}, True)
    check("only what happened", s == "I slept and sorted 276 memories, asked you 1 question, saved 2 more for a good moment, noticed 303 things on the "
                                    "computer and kept the ones worth keeping and watched the memory sit above 90%.", s)
    check("just started", pocket.my_day({}, 0, None, False) == "Not much -- I've only just started.")
    rt = tm.rig()
    used, said = rt.say("what did you do while I was away?")
    check("asked in the runtime: by rule", not used and said, f"{used} {said}")

    print("\n=== B4. Hype: facts, not flattery ===")
    s = pocket.hype([("A.S.H", 9 * 3600)], ("A.S.H", 5), "your interview went well")
    check("hours, a streak, a good moment", s == "Here's what's true: 9 hours on A.S.H this week; 5 days in a row on A.S.H; your interview went well. Keep that going.", s)
    check("nothing on record: no made-up praise", "don't have your wins" in pocket.hype([], None, None))

    print("\n=== B5. A quiz from the person's own notes ===")
    texts = ["Cassandra is a wide-column NoSQL database built for writes.", "MongoDB is a document database that stores JSON-like documents.",
             "Sharding means splitting data across machines. It is used for scale.", "this is a note without a definition"]
    cards = pocket.cards_from(texts)
    check("cards from 'X is Y' lines only", [c[0] for c in cards] == ["Cassandra", "MongoDB", "Sharding"], str(cards))
    q = pocket.Quiz(cards, "NoSQL", rng=random.Random(0))
    first = q.ask()
    x = first[len("What is "):-1]
    want = dict(cards)[x]
    r = q.hear(want)
    check("a right answer", r.startswith("Right. Next: What is"), r)
    r = q.hear("hint")
    check("a hint is the start of the answer", r.startswith("It starts:"), r)
    r = q.hear("I give up")
    check("giving up says it", " is " in r and "Next" in r, r)
    r = q.hear("bananas")
    check("the end, with the score", q.done and r.endswith("1 right."), r)
    rt = tm.rig()
    rt.say("remember that Cassandra is a wide-column NoSQL database built for writes")
    rt.say("remember that MongoDB is a NoSQL document database")
    rt.samsung = None
    used, said = rt.say("quiz me on NoSQL")
    check("started from what was noted", not used and said and said[-1].startswith("2 questions from your notes on NoSQL.") and said[-1].endswith("?"), str(said))
    used, said = rt.say("stop")
    check("stopped", said and said[-1].startswith("Stopped."), str(said))
    used, said = rt.say("quiz me on astrophysics")
    check("not enough in the notes: said so", said and "couldn't find enough" in said[-1], str(said))

    print("\n=== B6. The music that fits ===")
    check("parsed", pocket.music_ask("play my coding music") == "coding" and pocket.music_ask("play something I like") == "" and pocket.music_ask("play the video") is None)
    items = {"lofi": {"item": "Lofi Girl", "kind": "music", "score": 0.5, "last": 1, "seen_when": {"coding": 4}, "n": 4},
             "tenet": {"item": "Tenet", "kind": "movie", "score": 0.6, "last": 2}}
    from src.daemon.taste import occasion_of
    check("what goes with coding", pocket.pick_music(items, "coding", occasion_of) == "Lofi Girl")
    check("a film is not music", pocket.pick_music(items, "", occasion_of) == "Lofi Girl")
    rt = tm.rig()
    rt.desktop = td.drig()
    rt.taste.kb.tastes[rt.taste.user_key] = {"name": "", "user": True, "items": items}
    used, said = rt.say("play my coding music")
    check("opened in YouTube, by rule", not used and rt.desktop.ex_.opened and "youtube.com/results?search_query=Lofi%20Girl" in rt.desktop.ex_.opened[-1]
          and said[-1] == "Putting on Lofi Girl.", f"{rt.desktop.ex_.opened} {said}")
    rt = tm.rig()
    rt.desktop = td.drig()
    used, said = rt.say("play my studying music")
    check("nothing known: says how it learns", said and "I don't know what you play while studying" in said[-1], str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
