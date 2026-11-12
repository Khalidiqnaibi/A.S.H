"""
tst_fun.py -- jokes, made-up songs and inside jokes, by rule.

  J1  the ways people ask; "another one" only right after a joke
  J2  jokes: none repeated until all are told; kept across a restart
  J3  songs: about anything, "my cat" sung as "your cat"; something if nothing was named
  J4  inside jokes: a line you laughed at (soon after) is kept, and comes back; listed; forgotten on request
  J5  in the runtime: no model; the laugh is matched to what ASH just said

Run:  PYTHONIOENCODING=utf-8 python tests/tst_fun.py
"""

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.fun import Fun, JOKES  # noqa: E402

check = tw.check


def fun(path=None, seed=1):
    t = [1_800_000_000.0]
    f = Fun(path, clock=lambda: t[0], rng=random.Random(seed))
    f.t = t
    return f


def j1_asking():
    print("\n=== J1. Asking ===")
    f = fun()
    for said in ("tell me a joke", "Ash, make me laugh", "can you tell me another joke", "say something funny", "sing me a song",
                 "sing a song about my cat", "what are our inside jokes?"):
        check(f"{said!r}", f.hear(said) is not None, str(f.hear(said)))
    for said in ("this is a joke", "I'm singing", "the song is good", "what's the joke about"):
        check(f"not for us: {said!r}", f.hear(said) is None)
    check("'another one' after something that wasn't a joke is left alone", f.hear("another one") is None)
    check("...right after a joke it's another joke", f.hear("another one", last_was_joke=True) in JOKES)


def j2_jokes():
    print("\n=== J2. Jokes ===")
    path = os.path.join(tempfile.mkdtemp(), "fun.json")
    f = fun(path)
    seen = [f.joke() for _ in range(len(JOKES))]
    check("all of them before any repeats", len(set(seen)) == len(JOKES))
    g = fun(path)
    check("kept across a restart: the round goes on", len(g.told) == len(JOKES))
    nxt = g.joke()
    check("then a new round starts", nxt in JOKES and len(g.told) == 1)
    check("original jokes, short", all(10 < len(j) < 160 for j in JOKES))


def j3_songs():
    print("\n=== J3. Songs ===")
    f = fun()
    s = f.hear("sing a song about my cat")
    check("about it, turned round: 'your cat'", "your cat" in s and s.startswith("♪"), s)
    s = f.hear("sing me a song")
    check("nothing named: something anyway", s.startswith("♪") and len(s) > 60, s)
    check("several different songs", len({fun(seed=i).song("x") for i in range(12)}) >= 3)


def j4_inside():
    print("\n=== J4. Inside jokes ===")
    path = os.path.join(tempfile.mkdtemp(), "fun.json")
    f = fun(path)
    line = "I tried to make coffee but I don't have hands, so I made an excuse instead."
    check("a laugh soon after a line keeps it", f.heard("hahaha", line, f.t[0] - 10) and f.inside[0]["line"] == line)
    check("not twice", f.heard("lol", line, f.t[0] - 5) is False and len(f.inside) == 1)
    check("a laugh long after: not kept", f.heard("haha", "Something else entirely, said long ago.", f.t[0] - 600) is False)
    check("not a laugh: not kept", f.heard("ok", "Another thing ASH said, a while back.", f.t[0] - 5) is False)
    check("a line too short to be a joke: not kept", f.heard("haha", "Sure.", f.t[0]) is False)
    g = fun(path, seed=3)
    recalled = [g.joke() for _ in range(20)]
    check("it comes back now and then as a joke", any(r.startswith("Remember when I said “I tried to make coffee") for r in recalled), str(recalled[:3]))
    check("listed", line in g.hear("what are our inside jokes?"))
    check("forgotten on request", g.hear("forget our inside jokes") == "Forgotten -- 1 inside joke." and g.inside == [])
    check("...and on disk", Fun(path).inside == [])


def j5_runtime():
    print("\n=== J5. In the runtime ===")
    import tst_effects as te
    rt = te.rig(fun=True, fun_path=os.path.join(tempfile.mkdtemp(), "fun.json"))
    n = len(rt.ash.calls)
    rt.say("Ash, tell me a joke")
    joke = rt.out[-1]
    check("a joke, by rule", joke in JOKES and len(rt.ash.calls) == n, joke)
    rt.say("Ash, hahaha")
    check("laughing at it keeps it as an inside joke", rt.fun.inside and rt.fun.inside[-1]["line"] == joke, str(rt.fun.inside))
    rt.say("Ash, another one")
    check("'another one' right after a joke: another joke", rt.out[-1] in JOKES or rt.out[-1].startswith("Remember when"), rt.out[-1])
    rt.say("Ash, sing me a song about the rain")
    check("a song", "the rain" in rt.out[-1] and rt.out[-1].startswith("♪"), rt.out[-1])
    check("status", rt.status()["fun"]["songs"] == 1)


if __name__ == "__main__":
    for fn in (j1_asking, j2_jokes, j3_songs, j4_inside, j5_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
