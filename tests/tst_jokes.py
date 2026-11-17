"""
tst_jokes.py -- the same joke told again and again, and a denial that was made up.

Today the bread joke came back on every restart (the free-moment joke was picked by an in-memory counter that began at the first joke each
time and only moved when the person answered), and when the person said "you said the same joke 6 times already" the model answered
"I haven't told it six times yet" with nothing to check.

  J1  the joke book: the one told least, the oldest first; kept across a restart; a retired one never comes back
  J2  the free-moment joke: written down when told, not repeated until the rest have been, not the first one again after a restart
  J3  "you told me that already" is answered from the record and retires the joke
  J4  in the runtime: a joke asked for is recorded; the complaint is answered by rule, no model

Run:  PYTHONIOENCODING=utf-8 python tests/tst_jokes.py
"""

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.fun import Fun  # noqa: E402
from src.daemon.freemoment import Joke, _JOKES  # noqa: E402
from src.daemon.jokebook import JokeBook  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402

check = tw.check
POOL = ["a", "b", "c", "d"]


def j1_book():
    print("\n=== J1. The joke book ===")
    p = os.path.join(tempfile.mkdtemp(), "jokes.json")
    b = JokeBook(p, rng=random.Random(1))
    told = []
    for i in range(8):
        j = b.pick(POOL)
        told.append(j)
        b.record(j, 100.0 + i)
    check("all four are told before any is told twice", sorted(told[:4]) == POOL and sorted(told[4:]) == POOL, str(told))
    again = JokeBook(p, rng=random.Random(2))
    check("what was told survives a restart", again.times(told[0]) == 2 and again.last == told[-1])
    nxt = again.pick(POOL)
    check("and the next is the one told longest ago", nxt == told[4], f"{nxt} {told}")
    b.complaint()
    check("a retired one is never picked", b.last not in {b.pick(POOL) for _ in range(30)})
    check("with everything retired there is nothing to tell", JokeBook(None).pick([]) is None)


def j2_free_moment_joke():
    print("\n=== J2. The free-moment joke ===")
    p = os.path.join(tempfile.mkdtemp(), "jokes.json")
    seen = []
    for run in range(len(_JOKES)):
        j = Joke(_JOKES, book=JokeBook(p, rng=random.Random(run)))            # a fresh object each time: a restart
        c = j.propose(Environment(mood=0.5), 1000.0 + run)
        j.spoken(1000.0 + run)
        seen.append(c.text)
    check("across restarts every joke comes once before any comes twice", sorted(seen) == sorted(_JOKES), str(len(set(seen))))
    check("the first joke is not the first every time", seen.count(_JOKES[0]) == 1)
    low = Joke(_JOKES, book=JokeBook(None)).propose(Environment(mood=-0.5), 1.0)
    check("still never when the person is low", low is None)
    old = Joke(_JOKES)
    check("without a book it works as before", old.propose(Environment(mood=0.5), 1.0).text == _JOKES[0])


def j3_complaint():
    print("\n=== J3. 'You told me that already' ===")
    b = JokeBook(None)
    f = Fun(None)
    f.book = b
    b.record("the bread joke", 1.0)
    b.record("the bread joke", 2.0)
    b.record("the bread joke", 3.0)
    for said in ("you said the same joke 6 times already", "you told me that one already", "that joke again", "I've heard that one",
                 "you keep telling the same joke", "you've already told me that joke"):
        b2 = JokeBook(None)
        for t in (1.0, 2.0, 3.0):
            b2.record("the bread joke", t)
        f.book = b2
        got = f.hear(said)
        check(f"{said!r}: from the record", got is not None and "3 times" in got and "retired" in got, str(got))
        check("...and it is retired", "the bread joke" in b2.retired)
    f.book = JokeBook(None)
    check("with nothing on record it says so, not 'I haven't told it'", "on record" in f.hear("you said the same joke again"))
    f.book = None
    check("with no book, left alone", f.hear("you told me that one already") is None)
    f.book = JokeBook(None)
    for said in ("tell me a joke", "I like that joke", "tell me another one", "what is the same thing"):
        got = f.hear(said)
        check(f"{said!r} is not a complaint", got is None or "retired" not in got and "won't tell" not in got, str(got))


def j4_runtime():
    print("\n=== J4. In the runtime ===")
    import tst_effects as te
    d = tempfile.mkdtemp()
    rt = te.rig(fun=True, fun_path=os.path.join(d, "fun.json"), jokes_path=os.path.join(d, "jokes.json"))
    n = len(rt.ash.calls)
    rt.say("tell me a joke")
    told = rt.out[-1]
    check("a joke asked for is told by rule and written down", rt.jokebook.times(told) == 1 and len(rt.ash.calls) == n, f"{told!r}")
    rt.say("you said the same joke 6 times already")
    check("the complaint is answered from the record, no model", "retired" in rt.out[-1] or "won't tell" in rt.out[-1] and len(rt.ash.calls) == n, rt.out[-1])
    check("and that joke is retired", told in rt.jokebook.retired)
    seen = set()
    for _ in range(12):
        rt.say("tell me a joke")
        seen.add(rt.out[-1])
    check("it is never told again", told not in seen, str(seen))


if __name__ == "__main__":
    for fn in (j1_book, j2_free_moment_joke, j3_complaint, j4_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
