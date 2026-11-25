"""
tst_courtesy.py -- "Later", "will do thanks", "good night", "kk": a short line by rule, never a 20-second model call.

  Y1  what counts: thanks, goodbyes, good night, brb, short acknowledgements (the real lines from today among them)
  Y2  what does not: anything with a request in it, a question, "yes"/"ok"/"sure" (they answer a question), long lines
  Y3  never the same line twice running
  Y4  in the runtime: answered with no model call; a pending offer still gets "later"/"yes" first; not for a guest or overheard speech
  Y5  what the rules do not list is understood by meaning (the chooser), what the rules catch teaches it, and the rules stand without it

Run:  PYTHONIOENCODING=utf-8 python tests/tst_courtesy.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.courtesy import Courtesy  # noqa: E402

check = tw.check
from datetime import datetime  # noqa: E402

T = datetime(2027, 1, 15, 14, 0).timestamp()        # an afternoon, whatever the time zone
NIGHT = datetime(2027, 1, 15, 23, 0).timestamp()


def y1_what_counts():
    print("\n=== Y1. What counts ===")
    c = Courtesy()
    for said, kind in [("Later", "bye"), ("later.", "bye"), ("will do thanks", "thanks"), ("thanks", "thanks"), ("Thank you!", "thanks"),
                       ("ok thanks ash", "thanks"), ("thanks a lot man", "thanks"), ("cheers", "thanks"), ("thx", "thanks"),
                       ("see you", "bye"), ("see ya later", "bye"), ("bye", "bye"), ("ok bye", "bye"), ("gotta go", "bye"),
                       ("good night", "night"), ("gn", "night"), ("brb", "brb"), ("one sec", "brb"), ("kk", "ack"), ("k", "ack"),
                       ("got it", "ack"), ("will do", "ack"), ("noted", "ack"), ("Ash, thanks", "thanks")]:
        check(f"{said!r} is {kind}", c.kind(said) == kind, str(c.kind(said)))


def y2_what_does_not():
    print("\n=== Y2. What does not ===")
    c = Courtesy()
    for said in ("thanks, now what's the weather", "thanks but that's wrong", "later today remind me to call mom", "yes", "ok", "okay",
                 "sure", "alright", "yeah", "no", "what time is it", "tell me a joke", "thank you for the trivia, what's the answer",
                 "see you in the morning briefing at 8", "play something", "who is khalid", "kk what", ""):
        check(f"{said!r} is left alone", c.kind(said) is None, str(c.kind(said)))


def y3_not_the_same_twice():
    print("\n=== Y3. Not the same line twice running ===")
    c = Courtesy()
    seen = [c.hear("thanks", T) for _ in range(6)]
    check("six thanks, never the same line back to back", all(a != b for a, b in zip(seen, seen[1:])), str(seen))
    check("a noon 'night' is a goodbye, not a bedtime", Courtesy().hear("night", T) == "Alright, later.")
    check("at night it is a bedtime", Courtesy().hear("night", NIGHT) in ("Night. Sleep well.", "Good night.", "Night."))


def y4_runtime():
    print("\n=== Y4. In the runtime ===")
    import tst_effects as te
    rt = te.rig(courtesy=True)
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say("Later")
    check("'Later' is answered with no model call", len(rt.out) > k and len(rt.ash.calls) == n and rt.out[-1] in ("Later.", "See you.", "Catch you later.", "Alright, later."),
          f"{rt.out[-1:]} calls={len(rt.ash.calls) - n}")
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say("will do thanks")
    check("'will do thanks' is answered with no model call", len(rt.out) > k and len(rt.ash.calls) == n, f"{rt.out[-1:]}")
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say("what should I cook tonight")
    check("a real question still goes to the model", len(rt.ash.calls) > n)
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say("You are so dumb today.")
    check("a complaint about ASH is owned in a line, no model (it answered an insult with a paragraph)", len(rt.out) > k and len(rt.ash.calls) == n
          and ("wrong" in rt.out[-1] or "off" in rt.out[-1] or "should I have done" in rt.out[-1]), f"{rt.out[-1:]}")
    rt._last_turn = {"took_s": 63.0, "pathway": "slow", "tool": "", "timings": {}}
    rt.say("You are hallucinating, boy.")
    check("...and says what it knows went wrong: a model answer with nothing checked behind it", rt.out[-1].startswith("Fair -- that last answer came from the language model"), rt.out[-1])
    for t in ("you are right", "you are wrong about that", "you are so kind"):
        n = len(rt.ash.calls)
        rt.say(t)
        check(f"{t!r} is not a complaint", "get wrong" not in rt.out[-1] and "off" not in rt.out[-1].split()[-3:])
    bare = te.rig()
    n, k = len(bare.ash.calls), len(bare.out)
    bare.say("Later")
    check("with the switch off (a bare runtime) it goes on as before, to the model", len(bare.ash.calls) > n)


def y4b_nothing_asked():
    print("\n=== Y4b. A bare 'no thanks' / 'ok' with nothing asked ===")
    c = Courtesy()
    for t in ("No thanks.", "nope", "never mind", "I'm good", "not now", "it's fine", "nah"):
        check(f"{t!r}, nothing asked: courtesy", c.kind(t, open_q=False) == "decline", str(c.kind(t, open_q=False)))
        check(f"{t!r} after a question: not courtesy (it answers it)", c.kind(t) is None, str(c.kind(t)))
    for t in ("ok", "yes", "sure", "yeah"):
        check(f"{t!r}, nothing asked: an ack", c.kind(t, open_q=False) == "ack")
        check(f"{t!r} after a question: left alone", c.kind(t) is None)
    check("a request is never courtesy", c.kind("no thanks, open notepad", open_q=False) is None and c.kind("no what is the weather", open_q=False) is None)
    import tst_effects as te
    rt = te.rig(courtesy=True)
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say("No thanks.")
    check("in the runtime: no model call, a short line (it took 62 s)", len(rt.out) > k and len(rt.ash.calls) == n and rt.out[-1] in ("Okay.", "No problem.", "Alright.", "Sure."), f"{rt.out[-1:]} {len(rt.ash.calls) - n}")
    rt._last_out, rt._last_out_at = "Want me to open it?", rt.clock()
    n = len(rt.ash.calls)
    rt.say("nope")
    check("right after ASH asked something, 'nope' is not taken as courtesy", rt.out[-1] not in ("Okay.", "No problem.", "Alright.", "Sure.") or len(rt.ash.calls) > n, f"{rt.out[-1:]}")


def y5_by_meaning():
    print("\n=== Y5. By meaning ===")
    import semantic_fake as sf
    from src.daemon.chooser import Chooser
    from src.daemon.courtesy import SEEDS
    ch = Chooser("courtesy", sf.embed, SEEDS, accept=0.6, margin=0.12)
    c = Courtesy(chooser=ch)
    check("'much obliged' is not in the rules, so without the chooser it is left alone", Courtesy().kind("much obliged") is None)
    check("with the chooser it is thanks", c.kind("much obliged") == "thanks", str(ch.classify("much obliged").scores))
    check("'catch you tomorrow' is a goodbye", c.kind("catch you tomorrow") == "bye", str(ch.classify("catch you tomorrow").scores))
    check("a question with courtesy words in it is not", c.kind("what is the weather tomorrow") is None and c.kind("thanks, what time is it?") is None)
    check("a longer sentence is not", c.kind("thanks for all the help with the project yesterday morning") is None)
    n = len(ch.learned.get("thanks", []))
    c.kind("thanks")
    check("what the rules catch is added to what the chooser knows", len(ch.learned.get("thanks", [])) >= n)
    check("it counts what it did by meaning", c.stats["by_meaning"] >= 2)
    import tst_effects as te
    d = tempfile.mkdtemp()
    rt = te.rig(courtesy=True, chooser_embed=sf.embed, chooser_dir=d)
    n0, k = len(rt.ash.calls), len(rt.out)
    rt.say("much obliged")
    check("in the runtime 'much obliged' is answered at once, no model", len(rt.out) > k and len(rt.ash.calls) == n0, f"{rt.out[-1:]}")
    rt.say("what should I cook tonight")
    check("a real question still goes to the model", len(rt.ash.calls) > n0)
    plain = te.rig(courtesy=True)
    n0 = len(plain.ash.calls)
    plain.say("much obliged")
    check("without an encoder the rules stand and it goes to the model as before", len(plain.ash.calls) > n0)


if __name__ == "__main__":
    for fn in (y1_what_counts, y2_what_does_not, y3_not_the_same_twice, y4_runtime, y4b_nothing_asked, y5_by_meaning):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
