"""
tst_phone.py -- the phone through KDE Connect. The tool is faked: nothing is run and nothing is sent anywhere.

  P1  the ways people ask, and what is not about the phone
  P2  ring, note, share (the file just found), "is my phone connected?" -- the exact tool arguments
  P3  texts: a number said or from contacts, read back, sent only on "send it", not after five minutes
  P4  not set up: says how; alerts may also go to the ntfy address, notes never do
  P5  in the runtime: no model; only for lines said to ASH; not for a guest; "send it to my phone" after a file search

Run:  PYTHONIOENCODING=utf-8 python tests/tst_phone.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.phone import Phone, PhoneConfig  # noqa: E402

check = tw.check
T = [1_800_000_000.0]


class FakeCLI:
    def __init__(self, devices=("abc123def456",), fail=False):
        self.calls, self.devices, self.fail = [], list(devices), fail

    def __call__(self, args, timeout):
        self.calls.append(list(args[1:]))
        if "--list-available" in args:
            return 0, "\n".join(self.devices)
        return (1, "error") if self.fail else (0, "")


def phone(cli=None, cfg=None, poster=None):
    cli = cli or FakeCLI()
    posted = []
    p = Phone(PhoneConfig.from_dict(cfg or {"contacts": {"Omar Haddad": "+962791234567"}}), cli="kdeconnect-cli", runner=cli,
              poster=poster or (lambda url, text: posted.append((url, text)) or True), clock=lambda: T[0])
    return p, cli, posted


def p1_asking():
    print("\n=== P1. Asking ===")
    p, cli, _ = phone()
    for said in ("find my phone", "Ash, ring my phone", "where's my phone?", "can you call my phone please", "send \"buy milk\" to my phone",
                 "send a note to my phone: buy milk", "is my phone connected?", "text Omar saying I'm late"):
        check(f"{said!r} is the phone's", p.hear(said) is not None, str(p.hear(said)))
        p.pending_sms = None
    for said in ("call mom", "my phone is new", "what's on my phone", "send it", "text me later", "find my keys"):
        check(f"not the phone's: {said!r}", p.hear(said) is None, str(p.hear(said)))


def p2_actions():
    print("\n=== P2. Ring, note, share, connected ===")
    p, cli, _ = phone()
    check("ring", p.hear("find my phone") == "Ringing your phone." and cli.calls[-1] == ["-d", "abc123def456", "--ring"], str(cli.calls))
    check("note", p.hear("send a note to my phone: buy milk") == "Sent to your phone." and cli.calls[-1][-2:] == ["--ping-msg", "buy milk"])
    check("note, quoted", p.hear('send "call the bank" to my phone') == "Sent to your phone." and cli.calls[-1][-1] == "call the bank")
    out = p.hear("send it to my phone", last_thing=r"C:\Users\me\Documents\contract.docx")
    check("share the file just found", out == "Sent contract.docx to your phone." and cli.calls[-1][-2:] == ["--share", r"C:\Users\me\Documents\contract.docx"], out)
    check("share with nothing found yet: says so", "nothing I've just found" in p.hear("send it to my phone"))
    check("connected", p.hear("is my phone connected?") == "Yes, your phone is connected.")
    gone, _c, _ = phone(cli=FakeCLI(devices=()))
    check("not reachable", "can't see your phone" in gone.hear("is my phone connected?") and "couldn't reach" in gone.hear("ring my phone"))
    named, ncli, _ = phone(cfg={"device": "Pixel 8"})
    named.hear("ring my phone")
    check("a device set by name is used by name", ncli.calls[-1] == ["-n", "Pixel 8", "--ring"], str(ncli.calls))
    bad, bcli, _ = phone(cli=FakeCLI(fail=True))
    check("a failing tool is said, and counted", "couldn't reach" in bad.hear("ring my phone") and bad.stats["failed"] == 1)


def p3_texts():
    print("\n=== P3. Texts ===")
    p, cli, _ = phone()
    out = p.hear("text Omar saying I'm running late")
    check("a contact: read back with the number, not sent yet", out == "Text to Omar Haddad (+962791234567): \u201cI'm running late\u201d. Say \u201csend it\u201d to send it."
          and not any("--send-sms" in c for c in cli.calls), out)
    out = p.hear("send it")
    check("'send it' sends exactly that", out == "Sent the text to Omar Haddad." and cli.calls[-1][-4:] == ["--send-sms", "I'm running late", "--destination", "+962791234567"], str(cli.calls[-1]))
    check("once", p.hear("send it") is None)
    out = p.hear("text 079 123 4567 saying on my way")
    check("a number said", "0791234567" in out, out)
    check("'don't send it' drops it", p.hear("don't send it") == "Okay, I won't send it." and p.pending_sms is None)
    p.hear("text Omar saying hi")
    T[0] += 400
    check("five minutes later 'send it' sends nothing", p.hear("send it") is None and not any(c[-1:] == ["+962791234567"] and "hi" in c for c in cli.calls))
    check("no number known: says how", "don't have a number for Zed" in p.hear("text Zed saying hi"))
    off, _c, _ = phone(cfg={"allow_sms": False})
    check("texting switched off", "switched off" in off.hear("text 0791234567 saying hi"))


def p4_not_set_up():
    print("\n=== P4. Not set up; alerts ===")
    none = Phone(PhoneConfig.from_dict({"ntfy_url": "https://ntfy.example/ash"}), cli=None, runner=lambda a, t: (_ for _ in ()).throw(AssertionError("ran")),
                 poster=lambda url, text: True)
    check("says how to set it up", none.hear("ring my phone") == Phone.NOT_SET_UP)
    posted = []
    none.poster = lambda url, text: posted.append((url, text)) or True
    check("an alert goes to the ntfy address", none.notify("ASH guard: movement", alert=True) and posted == [("https://ntfy.example/ash", "ASH guard: movement")])
    check("a note never goes there", none.notify("my private note") is False and len(posted) == 1)
    check("can alert with ntfy alone", none.can_alert() and not Phone(PhoneConfig.from_dict({}), cli=None).can_alert())


def p5_runtime():
    print("\n=== P5. In the runtime ===")
    import tempfile
    import tst_effects as te
    cli = FakeCLI()
    rt = te.rig(phone=True, phone_background=False, phone_cfg={"contacts": {"Omar": "+962791234567"}}, phone_io={"cli": "kdeconnect-cli", "runner": cli},
                filesearch=True, filesearch_folders=[tempfile.mkdtemp()])
    n = len(rt.ash.calls)
    rt.say("Ash, find my phone")
    check("rings, by rule, no model", "--ring" in cli.calls[-1] and len(rt.ash.calls) == n, rt.out[-1])
    k = len(cli.calls)
    rt.say("find my phone", addressed=False)
    check("overheard: nothing rings", len(cli.calls) == k)
    rt.filesearch.last_path = (r"C:\docs\lease.txt", rt.t[0])
    rt.say("Ash, send it to my phone")
    check("'send it to my phone' after a file search shares that file", cli.calls[-1][-2:] == ["--share", r"C:\docs\lease.txt"], str(cli.calls[-1]))
    from src.senses.base import Modality, SensorEvent
    k = len(cli.calls)
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="Ash, ring my phone", addressed=True, salience=0.9,
                          urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest cannot", len(cli.calls) == k)
    rt._stranger_hold = lambda now: True
    rt.say("Ash, text Omar saying the code is 4321")
    check("a text's read-back waits for a stranger to go", rt.out[-1] == rt.HOLD_LINE)


def p6_off_the_loop():
    print("\n=== P6. A slow phone never stalls the loop ===")
    import threading
    import time as _t
    import tst_effects as te
    gate = threading.Event()

    class Slow(FakeCLI):
        def __call__(self, args, timeout):
            if "--ring" in args:
                gate.wait(5)
            return super().__call__(args, timeout)
    cli = Slow()
    jobs = []
    rt = te.rig(phone=True, phone_io={"cli": "kdeconnect-cli", "runner": cli}, phone_spawn=jobs.append)
    t0 = _t.time()
    rt.say("Ash, find my phone")
    check("the loop is free at once (the call waits on its own job)", _t.time() - t0 < 1.0 and len(jobs) == 1 and not rt.out)
    rt.say("Ash, ring my phone")
    check("a second ask while one is running is told to wait", rt.out and rt.out[-1].startswith("Still talking"), str(rt.out))
    gate.set()
    jobs.pop(0)()
    rt._tick()
    check("the result is said on the next tick", any("Ringing your phone" in o for o in rt.out), str(rt.out))
    rt.say("Ash, any new email?")
    check("the loop answers other things meanwhile", len(rt.out) >= 3)
    real = threading.Thread
    th = []
    rt2 = te.rig(phone=True, phone_io={"cli": "kdeconnect-cli", "runner": Slow()})
    gate.clear()
    t1 = _t.time()
    rt2.say("Ash, find my phone")
    check("with real threads too: returns at once", _t.time() - t1 < 1.0 and not rt2.out, str(rt2.out))
    gate.set()
    for _ in range(100):
        rt2.t[0] += 1
        rt2._tick()
        if rt2.out:
            break
        _t.sleep(0.05)
    check("...and says it when the phone answers", any("Ringing" in o for o in rt2.out), str(rt2.out))


if __name__ == "__main__":
    for fn in (p1_asking, p2_actions, p3_texts, p4_not_set_up, p5_runtime, p6_off_the_loop):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
