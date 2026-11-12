"""
tst_kids.py -- kids mode: simpler, kinder, nothing grown-up, a time limit.

  K1  on (with a time), off; a passphrase if one is set; never by a guest
  K2  grown-up requests refused kindly, before any feature sees them; play still works
  K3  the model is told it is talking with a child; a not-for-kids word in anything said is replaced
  K4  the time limit: said once, then only "kids mode off" works
  K5  in the runtime: refused by rule (no email, no files, no apps), jokes still there, the model's context

Run:  PYTHONIOENCODING=utf-8 python tests/tst_kids.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.kids import CONTEXT, KidsMode, REFUSE, SAFE  # noqa: E402

check = tw.check


def kids(passphrase="", minutes=60):
    t = [1_800_000_000.0]
    k = KidsMode(passphrase=passphrase, default_minutes=minutes, clock=lambda: t[0])
    k.t = t
    return k


def k1_on_off():
    print("\n=== K1. On and off ===")
    k = kids()
    for said in ("kids mode on", "Ash, turn on kids mode", "my daughter is using you", "kids mode for 30 minutes", "a kid is here"):
        check(f"on: {said!r}", k.parse(said) == "on")
    for said in ("kids mode off", "turn off kids mode", "end kids mode"):
        check(f"off: {said!r}", k.parse(said) == "off")
    check("not about it", k.parse("my kids are at school") is None and k.parse("kid friendly recipes") is None)
    out = k.start("kids mode for 30 minutes")
    check("on, for the time said", k.active and out.startswith("Kids mode on for 30 minutes") and k.until - k.t[0] == 1800)
    k2 = kids()
    k2.start("kids mode for 2 hours")
    check("hours too", k2.until - k2.t[0] == 7200)
    check("a guest can't turn it off", k.stop_asked(guest=True).startswith("Only a grown-up") and k.active)
    check("no passphrase set: off", k.stop_asked(guest=False) == "Kids mode off." and not k.active)
    p = kids(passphrase="blue banana")
    p.start("kids mode on")
    check("with a passphrase: asked for it", p.stop_asked(False) == "What's the passphrase?" and p.active)
    check("the wrong one: stays on", p.passphrase_heard("red apple", False) == "That's not it -- kids mode stays on." and p.active)
    p.stop_asked(False)
    check("the right one: off", p.passphrase_heard("Blue banana!", False) == "Kids mode off." and not p.active)
    p.start("kids mode on")
    p.stop_asked(False)
    check("...but not in a guest's voice", p.passphrase_heard("blue banana", True).startswith("That's not it") and p.active)


def k2_refused():
    print("\n=== K2. Grown-up things ===")
    k = kids()
    k.start("kids mode on")
    for said in ("any new email?", "reply to Omar saying hi", "text mom saying hi", "what's on my calendar", "find my tax return",
                 "remember that the code is 1234", "open notepad", "close chrome", "turn off the lights", "guard mode on",
                 "buy some robux", "delete my files", "what's my password", "turn off the camera", "privacy mode"):
        check(f"refused: {said!r}", k.refuse(said) == REFUSE, str(k.refuse(said)))
    for said in ("tell me a joke", "sing a song about dogs", "let's play a game", "why is the sky blue", "what's 7 times 8",
                 "tell me about dinosaurs"):
        check(f"still fine: {said!r}", k.refuse(said) is None)
    check("off: nothing refused", kids().refuse("any new email?") is None)


def k3_words():
    print("\n=== K3. What is said ===")
    k = kids()
    k.start("kids mode on")
    check("a not-for-kids word: replaced", k.filter("The soldier was killed in the war.") == SAFE)
    check("a kind answer: as it is", k.filter("The sky is blue because of the way sunlight scatters!") == "The sky is blue because of the way sunlight scatters!")
    check("words inside other words are fine (shell, chocolate, toward)", k.filter("A snail's shell, chocolate, toward home.").startswith("A snail"))
    check("off: nothing filtered", kids().filter("The war ended.") == "The war ended.")
    check("the model is told who it is talking to", "child" in CONTEXT and "simple" in CONTEXT)
    from src.daemon.fun import JOKES, SONGS
    check("every joke and song is kid-safe as written", all(k.filter(j) == j for j in JOKES) and all(k.filter(s) == s for s in SONGS))


def k4_time():
    print("\n=== K4. The time limit ===")
    k = kids(minutes=30)
    k.start("kids mode on")
    k.t[0] += 29 * 60
    check("not yet", k.time_up() is False)
    k.t[0] += 120
    check("then once", k.time_up() is True and k.time_up() is False)
    check("after that only 'kids mode off' works", k.refuse("tell me a joke").startswith("Kids mode time is up"))
    check("...which still works", k.stop_asked(False) == "Kids mode off.")


def k5_runtime():
    print("\n=== K5. In the runtime ===")
    import tst_effects as te
    rt = te.rig(kids=True, fun=True, fun_path=os.path.join(tempfile.mkdtemp(), "fun.json"), email=True,
                email_cfg={"host": "", "user": "", "password": ""})
    seen = {}
    real = rt.ash.think

    def think(query, image=None, context=""):
        seen["context"] = context
        return real(query, image=image, context=context)
    rt.ash.think = think
    rt.say("Ash, kids mode for 20 minutes")
    check("on", rt.kids.active and "20 minutes" in rt.out[-1], rt.out[-1])
    n = len(rt.ash.calls)
    rt.say("Ash, any new email?")
    check("email refused by rule, before the email feature", rt.out[-1] == REFUSE and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("Ash, tell me a joke")
    from src.daemon.fun import JOKES
    check("jokes still work", rt.out[-1] in JOKES, rt.out[-1])
    rt.say("Ash, why do cats purr")
    check("the model is told it is talking with a child", "KIDS MODE" in seen.get("context", ""), seen.get("context", "")[:80])
    rt.response_sink("Some people hate Mondays.", {"addressed": True, "pathway": "slow"})
    check("anything said with a not-for-kids word is replaced", rt.out[-1] == SAFE, rt.out[-1])
    rt.t[0] += 21 * 60
    rt._tick()
    check("time up: said once", any("time is up" in o for o in rt.out[-3:]), str(rt.out[-3:]))
    rt.say("Ash, kids mode off")
    check("off", not rt.kids.active and rt.out[-1] == "Kids mode off.")
    rt.say("Ash, tell me something")
    check("off: the model is not told about a child any more", "KIDS MODE" not in seen.get("context", ""))


if __name__ == "__main__":
    for fn in (k1_on_off, k2_refused, k3_words, k4_time, k5_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
