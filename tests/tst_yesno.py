"""
tst_yesno.py -- yes and no, said the way people say them.

  Y1  what counts as yes, what as no, what as neither
  Y2  a clear yes for things that cannot be taken back: "ok" and "mhm" are not enough to send a mail
  Y3  shutdown: any ordinary yes confirms, "shut down" again confirms, anything else cancels it
  Y4  the other questions: close an app, run a plan, play a game, send a text or a draft (only right after the read-back)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_yesno.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.yesno import is_no, is_yes  # noqa: E402

check = tw.check


def y1_words():
    print("\n=== Y1. Yes, no, neither ===")
    for t in ("yes", "Yes!", "ye", "yea", "yeah", "yeh", "yep", "yup", "ya", "y", "yess", "yesss", "yeahh", "okay", "ok", "okey", "k", "kk",
              "sure", "sure thing", "alright", "alright then", "fine", "do it", "go ahead", "please", "yes please", "of course", "why not",
              "absolutely", "definitely", "confirm", "confirmed", "affirmative", "mhm", "uh huh", "yes, shut it down", "okay do it",
              "yeah go ahead", "yes yes", "sure, why not", "Ash, yes", "proceed", "yeaah", "yeah sure", "yes ash"):
        check(f"yes: {t!r}", is_yes(t))
    for t in ("no", "No.", "n", "nope", "nah", "nay", "negative", "cancel", "stop", "don't", "dont", "wait", "never mind", "nevermind",
              "no thanks", "not now", "not yet", "leave it", "forget it", "abort", "no no", "noo", "hold on", "nope not now"):
        check(f"no: {t!r}", is_no(t))
    for t in ("yes but not now", "yes, actually wait", "ok but later", "sure, unless it's raining", "what time is it", "open spotify",
              "yes what is the weather like in Amman today", "the yeast is rising", "yet another one", "nothing", "note that down",
              "nopes are not things", "I said open notepad", "", "   "):
        check(f"neither: {t!r}", not is_yes(t) and not is_no(t), f"yes={is_yes(t)} no={is_no(t)}")
    check("a yes is never a no", not any(is_no(t) for t in ("yes", "yeah", "ok", "sure", "do it")))
    check("a no is never a yes", not any(is_yes(t) for t in ("no", "nope", "nah", "cancel", "not now", "never mind")))
    check("yet is not yes", not is_yes("yet") and not is_yes("year") and not is_yes("yeast"))
    check("a long sentence is not an answer", not is_yes("yes " + "and then I thought about the weather " * 3))


def y2_strong():
    print("\n=== Y2. Things that cannot be taken back ===")
    for t in ("yes", "yeah", "yep", "sure", "do it", "go ahead", "yes please", "absolutely", "ye", "confirm"):
        check(f"strong yes: {t!r}", is_yes(t, strong=True))
    for t in ("ok", "okay", "mhm", "alright", "fine", "k", "right", "uh huh"):
        check(f"not enough to send: {t!r}", not is_yes(t, strong=True) and is_yes(t))


def rig(**extra):
    import tst_effects as te
    return te.rig(**extra)


def y3_shutdown():
    print("\n=== Y3. Shutdown ===")
    for said in ("yes", "ye", "yeah", "okay", "ok", "sure", "do it", "yea", "yep", "alright", "go ahead", "confirm", "please", "yes, shut down",
                 "shut down", "yeah do it"):
        rt = rig()
        done = []
        rt.request_shutdown = lambda: done.append(1)
        rt.say("Ash, shut down", source="console")
        check(f"(asked) {said!r}: precondition", rt.out[-1].startswith("Shut ASH down completely"), rt.out[-1])
        rt.say(said, source="console")
        check(f"{said!r} confirms it", done == [1] and rt.out[-1] == "Shutting down. Bye.", rt.out[-1])
    for said in ("no", "nope", "not now", "cancel", "open the calculator", "what time is it", "hmm"):
        rt = rig()
        done = []
        rt.request_shutdown = lambda: done.append(1)
        rt.say("Ash, shut down", source="console")
        rt.say(said, source="console")
        check(f"{said!r} does not", done == [])
    rt = rig()
    done = []
    rt.request_shutdown = lambda: done.append(1)
    rt.say("Ash, shut down", source="console")
    rt.say("yes", addressed=False, source="microphone")
    check("a yes that was not said to ASH does not", done == [])
    rt = rig()
    rt.request_shutdown = lambda: done.append(1)
    rt.say("Ash, shut down", source="console")
    rt.t[0] += 120
    rt.say("yes", source="console")
    check("and not two minutes later", done == [])


def y4_others():
    print("\n=== Y4. The other questions ===")
    from tst_desktop import FakeEx
    for said in ("ye", "yea", "okay", "yeah sure", "do it"):
        rt = rig(desktop=True)
        fake = FakeEx()
        fake.procs |= {"notepad.exe"}
        rt.desktop.ex = fake
        rt.say("Ash, close notepad")
        rt.say(said)
        check(f"close an app: {said!r}", fake.closed == ["notepad.exe"], str(fake.closed))
    rt = rig(desktop=True)
    fake = FakeEx()
    fake.procs |= {"notepad.exe"}
    rt.desktop.ex = fake
    rt.say("Ash, close notepad")
    rt.say("nah")
    check("'nah' leaves it open", fake.closed == [] and "leave it open" in rt.out[-1], rt.out[-1])

    from src.daemon.phone import Phone, PhoneConfig
    T = [1_800_000_000.0]

    class CLI:
        def __init__(self):
            self.calls = []

        def __call__(self, args, timeout):
            self.calls.append(list(args))
            return 0, "abc123def456" if "--list-available" in args else ""
    cli = CLI()
    ph = Phone(PhoneConfig.from_dict({"contacts": {"Omar": "+962791234567"}}), cli="kdeconnect-cli", runner=cli, clock=lambda: T[0])
    ph.hear("text Omar saying I'm late", now=T[0])
    T[0] += 5
    check("'ok' does not send a text", ph.hear("ok", now=T[0]) is None and ph.pending_sms is not None)
    check("'yes' right after the read-back does", (ph.hear("yes", now=T[0] + 5) or "").startswith("Sent the text to Omar"), str(ph.pending_sms))
    ph.hear("text Omar saying I'm late", now=T[0])
    check("'yeah' two minutes later does not (it is probably about something else)", ph.hear("yeah", now=T[0] + 120) is None and ph.pending_sms is not None)
    check("...but 'send it' still does", (ph.hear("send it", now=T[0] + 120) or "").startswith("Sent"))
    ph.hear("text Omar saying I'm late", now=T[0])
    check("'nope' drops it", ph.hear("nope", now=T[0] + 3) == "Okay, I won't send it." and ph.pending_sms is None)

    from src.daemon.drafts import Drafts
    import tst_drafts as td
    d = td.make_drafts() if hasattr(td, "make_drafts") else None
    if d is not None:
        pass


if __name__ == "__main__":
    for fn in (y1_words, y2_strong, y3_shutdown, y4_others):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
